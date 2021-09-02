# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
from copy import deepcopy

import numpy as np
from datasets import load_dataset, concatenate_datasets

from evaluator import MoreEvaluator, prepro
from tempfile import NamedTemporaryFile

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from model_span import DiscernForSpanExtraction
from trainer_span import DiscernDataCollatorWithPaddingEvaluation, DiscernSpanTrainer, postprocess_span_predictions

ENTAILMENT_CLASSES = ['yes', 'no', 'unknown']

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # train_qa_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the training qa data."}
    # )
    validation_qa_prediction_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation qa prediction data."}
    )
    test_qa_prediction_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test qa prediction data."}
    )
    validation_qa_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation qa data."}
    )
    validation_qa_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation qa data."}
    )
    test_qa_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test qa data."}
    )
    test_qa_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test qa data."}
    )
    # train_retrieval_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the training retrieval (dpr|tfidf) data."}
    # )
    validation_retrieval_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation (dpr|tfidf) ata."}
    )
    validation_retrieval_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation (dpr|tfidf) data."}
    )
    test_retrieval_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test (dpr|tfidf) data."}
    )
    test_retrieval_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test (dpr|tfidf) data."}
    )
    snippet_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the snippet data."}
    )
    top_k_snippets: Optional[int] = field(
        default=1, metadata={"help": "Use top-k retrieved snippets for training"}
    )
    tokenized_file: Optional[str] = field(
        default=None, metadata={"help": "tokenized sharc data"}
    )
    tree_mapping_file: Optional[str] = field(
        default=None, metadata={"help": "map tree_id to its own reasoning structure such as follow_up questions / clauses"}
    )
    debug_sharc: bool = field(
        default=False,
        metadata={"help": "debug model, load less data"},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    lambda_entailment: Optional[float] = field(
        default=3.0, metadata={"help": "lambda entailment for entailment loss"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {
        "validation_seen": data_args.validation_qa_seen_file,
        "validation_unseen": data_args.validation_qa_unseen_file,
        "test_seen": data_args.test_qa_seen_file,
        "test_unseen": data_args.test_qa_unseen_file,
                  }

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.validation_qa_seen_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # get retrieval results
    def _load_json(path):
        with open(path) as f:
            data = json.load(f)
        return data

    id2snippet = _load_json(data_args.snippet_file)
    # id2snippet_parsed = _load_json(data_args.snippet_file.replace(".json", "_parsed.json"))
    snippet2id = {v:k for k, v in id2snippet.items()}
    # retrieval_snippet_id_train = _load_json(data_args.train_retrieval_file)
    retrieval_snippet_id_dev_seen = _load_json(data_args.validation_retrieval_seen_file)
    retrieval_snippet_id_dev_unseen = _load_json(data_args.validation_retrieval_unseen_file)
    retrieval_snippet_id_test_seen = _load_json(data_args.test_retrieval_seen_file)
    retrieval_snippet_id_test_unseen = _load_json(data_args.test_retrieval_unseen_file)
    # assert len(retrieval_snippet_id_train) == len(datasets['train']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_dev_seen) == len(datasets['validation_seen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_dev_unseen) == len(datasets['validation_unseen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_test_seen) == len(datasets['test_seen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_test_unseen) == len(datasets['test_unseen']), 'Examples mismatch!'

    tokenized_sharc_data = _load_json(data_args.tokenized_file)
    tree_mapping_data = _load_json(data_args.tree_mapping_file)
    snippetid2snippetparsed = {}
    for v in tree_mapping_data.values():
        if snippet2id[v['snippet']] not in snippetid2snippetparsed:
            snippetid2snippetparsed[snippet2id[v['snippet']]] = v['processed_snippet']

    def _add_retrieval_psgs_ids(qa_dataset, retrieval_dataset):
        def _helper(example, idx):
            example['retrieval_psgs_ids'] = retrieval_dataset[idx][0]
            return example
        return qa_dataset.map(_helper, with_indices=True)

    # dataset_train = _add_retrieval_psgs_ids(datasets['train'], retrieval_snippet_id_train)
    dataset_validation_seen = _add_retrieval_psgs_ids(datasets['validation_seen'], retrieval_snippet_id_dev_seen)
    dataset_validation_unseen = _add_retrieval_psgs_ids(datasets['validation_unseen'], retrieval_snippet_id_dev_unseen)
    dataset_test_seen = _add_retrieval_psgs_ids(datasets['test_seen'], retrieval_snippet_id_test_seen)
    dataset_test_unseen = _add_retrieval_psgs_ids(datasets['test_unseen'], retrieval_snippet_id_test_unseen)

    # load decision predictions
    DECISION_CLASSES = ['more', 'no', 'yes',]
    decision_logits_dev = np.load(data_args.validation_qa_prediction_file)
    decision_logits_test = np.load(data_args.test_qa_prediction_file)
    decision_predictions_dev = np.argmax(decision_logits_dev, axis=1)
    decision_predictions_test = np.argmax(decision_logits_test, axis=1)

    def _add_decision_predictions(qa_dataset, decision_predictions):
        def _helper(example, idx):
            example['decision_cls'] = DECISION_CLASSES[decision_predictions[idx]]
            return example
        return qa_dataset.map(_helper, with_indices=True)

    dataset_validation_seen = _add_decision_predictions(dataset_validation_seen, decision_predictions_dev[:len(dataset_validation_seen)])
    dataset_validation_unseen = _add_decision_predictions(dataset_validation_unseen, decision_predictions_dev[len(dataset_validation_seen):])
    dataset_test_seen = _add_decision_predictions(dataset_test_seen, decision_predictions_test[:len(dataset_test_seen)])
    dataset_test_unseen = _add_decision_predictions(dataset_test_unseen, decision_predictions_test[len(dataset_test_seen):])

    # filter yes/no examples, here our task is span extraction
    # dataset_train = dataset_train.filter(lambda example: example['answer'].lower() not in ['yes', 'no', 'irrelevant'])
    # dataset_validation_seen = dataset_validation_seen.filter(lambda example: example['decision_cls'].lower() not in ['yes', 'no', 'irrelevant'])
    # dataset_validation_unseen = dataset_validation_unseen.filter(lambda example: example['decision_cls'].lower() not in ['yes', 'no', 'irrelevant'])
    # dataset_test_seen = dataset_test_seen.filter(lambda example: example['decision_cls'].lower() not in ['yes', 'no', 'irrelevant'])
    # dataset_test_unseen = dataset_test_unseen.filter(lambda example: example['decision_cls'].lower() not in ['yes', 'no', 'irrelevant'])

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        additional_special_tokens=['<qu>', '<sc>', '<sn>', '<fuq>', '<fua>', '<ssep>'],
    )
    model = DiscernForSpanExtraction.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # lambda_entailment=model_args.lambda_entailment,
    )
    model.resize_token_embeddings(len(tokenizer))

    def _preprocess_function_user_info(example):
        # question, scenario, dialog history, snippet 1, snippet 2, snippet 3 ...
        result = {
            'user_idx': [],  # user relevant index
            'rule_idx': [],  # rule relevant index
            'input_ids': [],
            'label_entail': [],
            'start_position': -1,
            'end_position': -1,
        }

        ssep_token_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index('<ssep>')]
        cls_token_id = tokenizer.cls_token_id

        result['user_idx'].append(len(result['input_ids']))
        result['input_ids'] = [cls_token_id] + tokenized_sharc_data['questions']['<qu> ' + example['question']] + [ssep_token_id]

        result['user_idx'].append(len(result['input_ids']))
        result['input_ids'].extend([cls_token_id] + tokenized_sharc_data['scenarios']['<sc> ' + example['scenario']] + [ssep_token_id])

        for _fqa in example['history']:
            result['user_idx'].append(len(result['input_ids']))
            result['input_ids'].append(cls_token_id)
            result['input_ids'].extend(tokenized_sharc_data['follow_up_questions']['<fuq> ' + _fqa['follow_up_question']])
            if 'yes' in _fqa['follow_up_answer'].lower():
                result['input_ids'].extend(tokenized_sharc_data['follow_up_answers']["<fua> yes"])
            else:
                result['input_ids'].extend(tokenized_sharc_data['follow_up_answers']["<fua> no"])
            result['input_ids'].append(ssep_token_id)

        return result

    def _preprocess_function_snippet(result, top_k_snippet_ids, example):
        ssep_token_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index('<ssep>')]
        cls_token_id = tokenizer.cls_token_id
        sn_token_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index('<sn>')]
        result['span_mask'] = [0 for _ in result['input_ids']]
        result['snippet_idx'] = []

        # check the maximum top k ids under the embedding limit
        max_top_k_ids = 0
        temp_input_ids = deepcopy(result['input_ids'])
        for snippet_id in top_k_snippet_ids:
            # if snippet_id == snippet2id[example['snippet']]:
            #     m = tree_mapping_data[example['tree_id']]['processed_snippet']
            # else:
            m = snippetid2snippetparsed[snippet_id]
            temp_input_ids += [sn_token_id]
            for clause_id, clause in enumerate(m['clause_t']):
                temp_input_ids += [cls_token_id] + clause
            temp_input_ids += [ssep_token_id]
            if len(temp_input_ids) <= config.max_position_embeddings - 2:
                max_top_k_ids += 1
            else:
                break
        top_k_snippet_ids = top_k_snippet_ids[:max_top_k_ids]
        result['top_k_input_ids'] = max_top_k_ids

        fqs, fas = [], []
        for fqa in example['evidence'] + example['history']:
            fq, fa = fqa['follow_up_question'], fqa['follow_up_answer'].lower()
            fa = 'no' if 'no' in fa else 'yes'  # fix possible typos like 'noe'
            fqs.append(fq)
            fas.append(ENTAILMENT_CLASSES.index(fa))

        for snippet_id in top_k_snippet_ids:
            result['snippet_idx'].append(len(result['input_ids']))
            # if snippet_id == snippet2id[example['snippet']]:
            #     is_gold = True
            #     m = tree_mapping_data[example['tree_id']]['processed_snippet']
            #     span_clause_id, span_clause_start, span_clause_end, span_clause_text, span_clause_dist = m['q2clause'][example['answer']].values()
            # else:
            #     is_gold = False
            m = snippetid2snippetparsed[snippet_id]

            result['input_ids'] += [sn_token_id]
            result['span_mask'] += [0]
            for clause_id, clause in enumerate(m['clause_t']):
                result['rule_idx'].append(len(result['input_ids']))
                # if is_gold and clause_id == span_clause_id:
                #     span_offset = len(result['input_ids'])
                #     result['start_position'] = span_clause_start + span_offset + 1
                #     result['end_position'] = span_clause_end + span_offset + 1
                result['input_ids'] += [cls_token_id] + clause
                result['span_mask'] += [0] + [1 for _ in clause]
            result['input_ids'] += [ssep_token_id]
            result['span_mask'] += [0]

            # if is_gold:
            #     assert result['input_ids'][result['start_position']:result['end_position'] + 1] == \
            #            m['clause_t'][span_clause_id][span_clause_start:span_clause_end + 1]

            for clause_id, clause2q in enumerate(m['clause2q']):
                # if not is_gold:
                result['label_entail'].append(ENTAILMENT_CLASSES.index('unknown'))
            #     else:
            #         sentence_entail_states = []
            #         for clause2qj in clause2q:
            #             clause2aj = fas[fqs.index(clause2qj)] if clause2qj in fqs else ENTAILMENT_CLASSES.index('unknown')
            #             sentence_entail_states.append(clause2aj)
            #         if len(sentence_entail_states) > 1:
            #             if ENTAILMENT_CLASSES.index('yes') in sentence_entail_states:
            #                 result['label_entail'].append(ENTAILMENT_CLASSES.index('yes'))
            #             elif ENTAILMENT_CLASSES.index('no') in sentence_entail_states:
            #                 result['label_entail'].append(ENTAILMENT_CLASSES.index('no'))
            #             else:
            #                 result['label_entail'].append(ENTAILMENT_CLASSES.index('unknown'))
            #         elif len(sentence_entail_states) == 1:
            #             result['label_entail'].append(sentence_entail_states[0])
            #         else:
            #             result['label_entail'].append(ENTAILMENT_CLASSES.index('unknown'))
            #
            # assert len(result['label_entail']) == len(result['rule_idx'])

        # cropping to max_length
        assert len(result['span_mask']) == len(result['input_ids'])
        result['input_ids'] = result['input_ids'][:config.max_position_embeddings - 2]
        result['span_mask'] = result['span_mask'][:config.max_position_embeddings - 2]
        # ensure that all examples have <eos> in the end (required by bart classification)
        if len(result['input_ids']) == config.max_position_embeddings - 2:
            result['input_ids'][-1] = ssep_token_id
            result['span_mask'][-1] = 0
        num_rule_cls = result['input_ids'].count(cls_token_id) - len(result['user_idx'])
        result['rule_idx'] = result['rule_idx'][:num_rule_cls]
        result['label_entail'] = result['label_entail'][:num_rule_cls]
        result['attention_mask'] = [1 for _ in result['input_ids']]
        if result['end_position'] >= len(result['input_ids']):
            result['end_position'] = -1
            result['start_position'] = -1
        if result['start_position'] == -1 or result['end_position'] == -1:
            result['gold_span_text'] = ""
        else:
            result['gold_span_text'] = tokenizer.decode(result['input_ids'][result['start_position']:result['end_position'] + 1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return result

    def preprocess_function(example):
        result = _preprocess_function_user_info(example)
        top_k_snippet_ids = example['retrieval_psgs_ids'][:data_args.top_k_snippets]
        result = _preprocess_function_snippet(result, top_k_snippet_ids, example)
        return result

    # dataset_train = dataset_train.map(preprocess_function, load_from_cache_file=not data_args.overwrite_cache)

    # logger.info("Train Unfiltered: {}".format(len(dataset_train)))

    # dataset_train = dataset_train.filter(lambda example: example['start_position'] != -1 and example['end_position'] != -1)

    # logger.info("Train Filtered: {}".format(len(dataset_train)))

    # logger.info('Train: avg {:.2f} / max {} / min {} snippets'.format(
    #     np.mean(dataset_train['top_k_input_ids']),
    #     max(dataset_train['top_k_input_ids']),
    #     min(dataset_train['top_k_input_ids']),
    # ))

    dataset_validation_seen = dataset_validation_seen.map(preprocess_function, load_from_cache_file=not data_args.overwrite_cache)
    dataset_validation_unseen = dataset_validation_unseen.map(preprocess_function, load_from_cache_file=not data_args.overwrite_cache)
    dataset_test_seen = dataset_test_seen.map(preprocess_function, load_from_cache_file=not data_args.overwrite_cache)
    dataset_test_unseen = dataset_test_unseen.map(preprocess_function, load_from_cache_file=not data_args.overwrite_cache)
    dataset_validation = concatenate_datasets([dataset_validation_seen, dataset_validation_unseen])
    # dataset_validation = concatenate_datasets([dataset_test_seen, dataset_test_unseen])
    dataset_test = concatenate_datasets([dataset_test_seen, dataset_test_unseen])

    logger.info('Validation: avg {:.2f} / max {} / min {} snippets'.format(
        np.mean(dataset_validation['top_k_input_ids']),
        max(dataset_validation['top_k_input_ids']),
        min(dataset_validation['top_k_input_ids']),
    ))

    logger.info('Test: avg {:.2f} / max {} / min {} snippets'.format(
        np.mean(dataset_test['top_k_input_ids']),
        max(dataset_test['top_k_input_ids']),
        min(dataset_test['top_k_input_ids']),
    ))

    def compute_metrics(preds, data):
        import evaluator
        data.reset_format()
        predictions, labels = [], []
        for e_span, e_decision in zip(preds.predictions, data):
            assert e_span['id'] == e_decision['utterance_id']
            predictions.append({
                'utterance_id': e_span['id'],
                'answer': e_span['answer'] if e_decision['decision_cls'] == 'more' else e_decision['decision_cls']
            })
            labels.append({'utterance_id': e_decision['utterance_id'], 'answer': e_decision['answer']})
        with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
            json.dump(predictions, fp)
            fp.flush()
            json.dump(labels, fg)
            fg.flush()
            results = evaluator.evaluate(fg.name, fp.name, mode='combined')
            # results['combined'] = results['macro_accuracy'] * results['bleu_4']
        return results

    # Post-processing:
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_span_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            tokenizer=tokenizer,
        )
        references = [{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(examples['utterance_id'], examples['answer'])]
        return EvalPrediction(predictions=predictions, label_ids=references)

    if data_args.debug_sharc:
        # dataset_train = dataset_train.filter(lambda example, indice: indice < 32, with_indices=True)
        dataset_validation = dataset_validation.filter(lambda example, indice: indice < 16, with_indices=True)
        dataset_test = dataset_test.filter(lambda example, indice: indice < 16, with_indices=True)

    # evaluate gold spans
    # logger.info(compute_metrics(EvalPrediction(predictions=[{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(dataset_train['utterance_id'], dataset_train['gold_span_text'])],
    #                                            label_ids=[{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(dataset_train['utterance_id'], dataset_train['answer'])])))
    # logger.info(compute_metrics(EvalPrediction(predictions=[{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(dataset_validation['utterance_id'], dataset_validation['gold_span_text'])],
    #                                            label_ids=[{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(dataset_validation['utterance_id'], dataset_validation['answer'])])))
    # logger.info(compute_metrics(EvalPrediction(predictions=[{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(dataset_test['utterance_id'], dataset_test['gold_span_text'])],
    #                                            label_ids=[{"id": ex_id, "answer": ex_answer} for ex_id, ex_answer in zip(dataset_test['utterance_id'], dataset_test['answer'])])))


    # Initialize our Trainer
    trainer = DiscernSpanTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_validation,
        eval_dataset=dataset_validation,
        eval_examples=dataset_validation,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        # data_collator=default_data_collator if data_args.pad_to_max_length else None,
        post_process_function=post_processing_function,
        data_collator=DiscernDataCollatorWithPaddingEvaluation(tokenizer),
        model_args=model_args,
        config=config,
    )

    if not trainer.is_model_parallel:
        trainer.model = trainer.model.to(trainer.args.device)

    tasks = ['dev', 'test']
    eval_datasets = [dataset_validation, dataset_test]
    all_retrieval_ids = [retrieval_snippet_id_dev_seen+retrieval_snippet_id_dev_unseen, retrieval_snippet_id_test_seen+retrieval_snippet_id_test_unseen, ]
    # for seen /unseen snippets ablation
    # tasks = ['dev_seen', 'dev_unseen', 'test_seen', 'test_unseen']
    # eval_datasets = [dataset_validation_seen, dataset_validation_unseen, dataset_test_seen, dataset_test_unseen]
    # all_retrieval_ids = [retrieval_snippet_id_dev_seen, retrieval_snippet_id_dev_unseen,
    #                      retrieval_snippet_id_test_seen, retrieval_snippet_id_test_unseen, ]
    for eval_dataset, task, retrieval_ids in zip(eval_datasets, tasks, all_retrieval_ids):
        eval_predictions, eval_results = trainer.predict(eval_dataset=eval_dataset, eval_examples=eval_dataset)
        # add snippets for each prediction
        eval_predictions = eval_predictions.predictions
        for pred, example, retrieval_id in zip(eval_predictions, eval_dataset, retrieval_ids):
            snippet_idx = 0
            snippet_starts = example['snippet_idx']
            snippet_ends = snippet_starts[1:] + [512]
            for ss, se in zip(snippet_starts, snippet_ends):
                if ss <= pred['answer_span_start'] and pred['answer_span_end'] <= se:
                    pred['span_snippet_idx'] = snippet_idx
                    pred['span_snippet'] = id2snippet[retrieval_id[0][snippet_idx]]
                    break
                else:
                    snippet_idx += 1
        saving_file = data_args.validation_qa_prediction_file if task == 'dev' else data_args.test_qa_prediction_file
        # for seen /unseen snippets ablation
        # if 'dev' in task:
        #     saving_file = data_args.validation_qa_prediction_file.replace('dev', task)
        # else:
        #     saving_file = data_args.test_qa_prediction_file.replace('test', task)
        prediction_eval_file = os.path.join(saving_file.replace(".npy", "_span.json"))
        result_eval_file = os.path.join(saving_file.replace(f"predictions_{task}.npy", f"results_{task}_span.json"))
        if trainer.is_world_process_zero():
            logger.info(f"***** Eval results {task} *****")
            for key, value in sorted(eval_results.items()):
                logger.info(f"  {key} = {value}")
            logger.info(f"Saving {prediction_eval_file}")
            logger.info(f"Saving {result_eval_file}")
            with open(prediction_eval_file, "w") as f:
                json.dump(eval_predictions, f)
            with open(result_eval_file, "w") as f:
                json.dump(eval_results, f)


if __name__ == "__main__":
    main()
