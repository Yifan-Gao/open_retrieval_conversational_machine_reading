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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
from datasets import load_dataset, concatenate_datasets

from sklearn.metrics import accuracy_score, confusion_matrix

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process


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
    train_qa_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training qa data."}
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
    train_retrieval_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training retrieval (dpr|tfidf) data."}
    )
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
    debug_sharc: bool = field(
        default=False,
        metadata={"help": "debug model, load less data"},
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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if data_args.task_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     datasets = load_dataset("glue", data_args.task_name)
    # else:
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {
        "train": data_args.train_qa_file,
        "validation_seen": data_args.validation_qa_seen_file,
        "validation_unseen": data_args.validation_qa_unseen_file,
        "test_seen": data_args.test_qa_seen_file,
        "test_unseen": data_args.test_qa_unseen_file,
                  }

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_qa_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    def _get_label(example):
        if example['answer'].lower() in ['yes', 'no']:
            example['label'] = example['answer'].lower()
        else:
            example['label'] = 'more'
        return example
    datasets = datasets.map(_get_label)
    label_list = datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # get retrieval results
    def _load_json(path):
        with open(path) as f:
            data = json.load(f)
        return data

    id2snippet = _load_json(data_args.snippet_file)
    snippet2id = {v:k for k, v in id2snippet.items()}
    retrieval_snippet_id_train = _load_json(data_args.train_retrieval_file)
    retrieval_snippet_id_dev_seen = _load_json(data_args.validation_retrieval_seen_file)
    retrieval_snippet_id_dev_unseen = _load_json(data_args.validation_retrieval_unseen_file)
    retrieval_snippet_id_test_seen = _load_json(data_args.test_retrieval_seen_file)
    retrieval_snippet_id_test_unseen = _load_json(data_args.test_retrieval_unseen_file)
    assert len(retrieval_snippet_id_train) == len(datasets['train']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_dev_seen) == len(datasets['validation_seen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_dev_unseen) == len(datasets['validation_unseen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_test_seen) == len(datasets['test_seen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_test_unseen) == len(datasets['test_unseen']), 'Examples mismatch!'

    def _add_retrieval_psgs_ids(qa_dataset, retrieval_dataset):
        def _helper(example, idx):
            example['retrieval_psgs_ids'] = retrieval_dataset[idx]
            return example
        return qa_dataset.map(_helper, with_indices=True)

    dataset_train = _add_retrieval_psgs_ids(datasets['train'], retrieval_snippet_id_train)
    dataset_validation_seen = _add_retrieval_psgs_ids(datasets['validation_seen'], retrieval_snippet_id_dev_seen)
    dataset_validation_unseen = _add_retrieval_psgs_ids(datasets['validation_unseen'], retrieval_snippet_id_dev_unseen)
    dataset_test_seen = _add_retrieval_psgs_ids(datasets['test_seen'], retrieval_snippet_id_test_seen)
    dataset_test_unseen = _add_retrieval_psgs_ids(datasets['test_unseen'], retrieval_snippet_id_test_unseen)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
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
        # additional_special_tokens=['<qu>', '<sc>', '<sn>', '<fuq>', '<fua>', '<fqa>', '<ssep>'],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    # get tokenized passages
    def _get_tokenized_kv(original, tokenized):
        t_keys = tokenized.keys()
        out = {_q: {} for _q in original}
        for t_k in t_keys:
            for _q, _v in zip(original, tokenized[t_k]):
                out[_q][t_k] = _v
        return out

    tokenized_sharc_path = f'./data/{model_args.model_name_or_path}-tokenized-nospecialtoken.json'
    if os.path.exists(tokenized_sharc_path):
        with open(tokenized_sharc_path) as f:
            tokenized_sharc_data = json.load(f)
    else:
        os.makedirs('./data', exist_ok=True)
        # '<qu>', '<sc>', '<sn>', '<fuq>', '<fua>', '<ina>'
        questions = list(set([_ex for dataset_split in datasets.values() for _ex in dataset_split['question']]))
        scenarios = list(set([_ex for dataset_split in datasets.values() for _ex in dataset_split['scenario']]))
        snippets = list(set([_ex for dataset_split in datasets.values() for _ex in dataset_split['snippet']]))
        follow_up_questions = list(set(
            [fuqa['follow_up_question'] for dataset_split in datasets.values() for _ex in dataset_split['history'] for fuqa in _ex] +
            [fuqa['follow_up_question'] for dataset_split in datasets.values() for _ex in dataset_split['evidence'] for fuqa in _ex]))
        follow_up_answers = ["yes", "no",]
        inquire_answers = list(set([_ex for dataset_split in datasets.values() for _ex in dataset_split['answer']]))
        questions_tokenized = tokenizer(questions, add_special_tokens=False)
        scenarios_tokenized = tokenizer(scenarios, add_special_tokens=False)
        snippets_tokenized = tokenizer(snippets, add_special_tokens=False)
        follow_up_questions_tokenized = tokenizer(follow_up_questions, add_special_tokens=False)
        follow_up_answers_tokenized = tokenizer(follow_up_answers, add_special_tokens=False)
        inquire_answers_tokenized = tokenizer(inquire_answers, add_special_tokens=False)
        tokenized_sharc_data = {
            'questions': _get_tokenized_kv(questions, questions_tokenized),
            'scenarios': _get_tokenized_kv(scenarios, scenarios_tokenized),
            'snippets': _get_tokenized_kv(snippets, snippets_tokenized),
            'follow_up_questions': _get_tokenized_kv(follow_up_questions, follow_up_questions_tokenized),
            'follow_up_answers': _get_tokenized_kv(follow_up_answers, follow_up_answers_tokenized),
            'inquire_answers': _get_tokenized_kv(inquire_answers, inquire_answers_tokenized),
        }
        with open(tokenized_sharc_path, 'w') as f:
            json.dump(tokenized_sharc_data, f)
        logger.info(f"Saving tokenized sharc data {tokenized_sharc_path}")


    label_to_id = {v: i for i, v in enumerate(label_list)}

    def _preprocess_function_user_info(example):
        # question, scenario, dialog history, snippet 1, snippet 2, snippet 3 ...
        result = {'label': label_to_id[example['label']]}
        for k in ['input_ids', 'attention_mask']:
            # fqa_token_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index('<fqa>')] if k == 'input_ids' else 1
            # ssep_token_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index('<ssep>')] if k == 'input_ids' else 1
            cls_token_id = tokenizer.cls_token_id if k == 'input_ids' else 1
            sep_token_id = tokenizer.sep_token_id if k == 'input_ids' else 1
            result[k] = [cls_token_id] + \
                        tokenized_sharc_data['questions'][example['question']][k] + [sep_token_id] + \
                        tokenized_sharc_data['scenarios'][example['scenario']][k] + [sep_token_id]
            # result[k].append(fqa_token_id)
            for _fqa in example['history']:
                result[k].extend(tokenized_sharc_data['follow_up_questions'][_fqa['follow_up_question']][k])
                if 'yes' in _fqa['follow_up_answer'].lower():
                    result[k].extend(tokenized_sharc_data['follow_up_answers']["yes"][k])
                else:
                    result[k].extend(tokenized_sharc_data['follow_up_answers']["no"][k])
                result[k].append(sep_token_id)
        return result

    def _preprocess_function_snippet(result, top_k_snippet_ids):
        for k in ['input_ids', 'attention_mask']:
            # ssep_token_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index('<ssep>')] if k == 'input_ids' else 1
            eos_token_id = tokenizer.eos_token_id if k == 'input_ids' else 1
            sep_token_id = tokenizer.sep_token_id if k == 'input_ids' else 1
            for snippet_id in top_k_snippet_ids:
                result[k].extend(tokenized_sharc_data['snippets'][id2snippet[snippet_id]][k] + [sep_token_id])
            # replace the last ssep with eos
            result[k][-1] = eos_token_id
            # cropping to max_length
            result[k] = result[k][:config.max_position_embeddings - 2]
            # ensure that all examples have <eos> in the end (required by bart classification)
            if k == 'input_ids' and len(result[k]) == config.max_position_embeddings - 2:
                result[k][-1] = eos_token_id
        return result


    def preprocess_function_train(example):
        result = _preprocess_function_user_info(example)

        #  add gold snippet in training
        if snippet2id[example['snippet']] in example['retrieval_psgs_ids'][:data_args.top_k_snippets]:
            top_k_snippet_ids = example['retrieval_psgs_ids'][:data_args.top_k_snippets]
        else:
            top_k_snippet_ids = [snippet2id[example['snippet']]] + example['retrieval_psgs_ids'][:data_args.top_k_snippets - 1]

        result = _preprocess_function_snippet(result, top_k_snippet_ids)

        return result

    dataset_train = dataset_train.map(preprocess_function_train, load_from_cache_file=not data_args.overwrite_cache)

    def preprocess_function_eval(example):
        result = _preprocess_function_user_info(example)

        top_k_snippet_ids = example['retrieval_psgs_ids'][:data_args.top_k_snippets]

        result = _preprocess_function_snippet(result, top_k_snippet_ids)

        return result

    dataset_validation_seen = dataset_validation_seen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_validation_unseen = dataset_validation_unseen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_test_seen = dataset_test_seen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_test_unseen = dataset_test_unseen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_validation = concatenate_datasets([dataset_validation_seen, dataset_validation_unseen])
    dataset_test = concatenate_datasets([dataset_test_seen, dataset_test_unseen])


    def compute_metrics(p: EvalPrediction):
        metrics = {}
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        golds = p.label_ids
        micro_accuracy = accuracy_score(golds, preds)
        metrics["micro_accuracy"] = float("{0:.2f}".format(micro_accuracy * 100))  # int(100 * micro_accuracy) / 100
        conf_mat = confusion_matrix(golds, preds, labels=[0, 1, 2])
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat.shape[0])])
        metrics["macro_accuracy"] = float("{0:.2f}".format(macro_accuracy * 100))  # int(100 * macro_accuracy) / 100
        metrics["combined"] = float("{0:.2f}".format(macro_accuracy * micro_accuracy * 100))
        metrics["confmat"] = conf_mat.tolist()
        return metrics

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_validation_seen if data_args.debug_sharc else dataset_train,
        eval_dataset=dataset_validation_seen if data_args.debug_sharc else dataset_validation,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Test ***")
        # tasks = ['seen', 'unseen', 'all']
        # eval_datasets = [dataset_test_seen, dataset_test_unseen, dataset_test]
        tasks = ['all']
        eval_datasets = [dataset_test]
        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            output_eval_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
            eval_results.update(eval_result)

    return eval_results


if __name__ == "__main__":
    main()
