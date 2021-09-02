#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
from copy import deepcopy
from tempfile import NamedTemporaryFile
from evaluator import MoreEvaluator, prepro

import numpy as np
from datasets import load_dataset, load_metric, Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from trainer_seq2seq import DataCollatorForSeq2Seq


logger = logging.getLogger(__name__)


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
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # task: str = field(
    #     default="summarization",
    #     metadata={
    #         "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
    #         "pegasus) or translation (or translation_{xx}_to_{yy})."
    #     },
    # )
    # dataset_name: Optional[str] = field(
    #     default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    # text_column: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    # )
    # summary_column: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    # )
    # train_qa_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the training qa data."}
    # )
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
    # train_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the training qa data."}
    # )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation qa data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test qa data."}
    )
    # train_retrieval_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the training retrieval (dpr|tfidf) data."}
    # )
    # validation_retrieval_seen_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the validation (dpr|tfidf) ata."}
    # )
    # validation_retrieval_unseen_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the validation (dpr|tfidf) data."}
    # )
    # test_retrieval_seen_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the test (dpr|tfidf) data."}
    # )
    # test_retrieval_unseen_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the test (dpr|tfidf) data."}
    # )
    # snippet_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the snippet data."}
    # )
    # tokenized_file: Optional[str] = field(
    #     default=None, metadata={"help": "tokenized sharc data"}
    # )
    # tree_mapping_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "map tree_id to its own reasoning structure such as follow_up questions / clauses"}
    # )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    # max_train_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
    #         "value if set."
    #     },
    # )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    # eval_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

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
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # get retrieval results
    def _load_json(path):
        with open(path) as f:
            data = json.load(f)
        return data

    def _loads_json(loadpath):
        with open(loadpath, 'r', encoding='utf-8') as fh:
            dataset = []
            for line in fh:
                example = json.loads(line)
                dataset.append(example)
        return dataset

    def _load_dataset(data):
        data_dict = {k: [] for k in data[0].keys()}
        for ex in data:
            for k in data_dict:
                data_dict[k].append(ex[k])
        return Dataset.from_dict(data_dict)

    validation_span_data = _load_json(data_args.validation_file)
    test_span_data = _load_json(data_args.test_file)

    DECISION_CLASSES = ['more', 'no', 'yes', ]
    decision_logits_dev = np.load(data_args.validation_file.replace("_span.json", ".npy"))
    decision_logits_test = np.load(data_args.test_file.replace("_span.json", ".npy"))
    decision_predictions_dev = [DECISION_CLASSES[cls] for cls in np.argmax(decision_logits_dev, axis=1)]
    decision_predictions_test = [DECISION_CLASSES[cls] for cls in np.argmax(decision_logits_test, axis=1)]
    dev_qa = _loads_json(data_args.validation_qa_seen_file) + _loads_json(data_args.validation_qa_unseen_file)
    test_qa = _loads_json(data_args.test_qa_seen_file) + _loads_json(data_args.test_qa_unseen_file)

    dataset_dev = _load_dataset(validation_span_data)
    dataset_test = _load_dataset(test_span_data)

    # ### seen unseen
    # validation_file_seen = data_args.validation_file.replace('dev', 'dev_seen')
    # validation_file_unseen = data_args.validation_file.replace('dev', 'dev_unseen')
    # test_file_seen = data_args.test_file.replace('test', 'test_seen')
    # test_file_unseen = data_args.test_file.replace('test', 'test_unseen')
    # validation_span_data_seen = _load_json(validation_file_seen)
    # test_span_data_seen = _load_json(test_file_seen)
    # validation_span_data_unseen = _load_json(validation_file_unseen)
    # test_span_data_unseen = _load_json(test_file_unseen)
    #
    # # predictions_dev_span.json
    # DECISION_CLASSES = ['more', 'no', 'yes', ]
    # decision_logits_dev_seen = np.load(validation_file_seen.replace("_span.json", ".npy"))
    # decision_logits_test_seen = np.load(test_file_seen.replace("_span.json", ".npy"))
    # decision_logits_dev_unseen = np.load(validation_file_unseen.replace("_span.json", ".npy"))
    # decision_logits_test_unseen = np.load(test_file_unseen.replace("_span.json", ".npy"))
    # decision_predictions_dev_seen = [DECISION_CLASSES[cls] for cls in np.argmax(decision_logits_dev_seen, axis=1)]
    # decision_predictions_test_seen = [DECISION_CLASSES[cls] for cls in np.argmax(decision_logits_test_seen, axis=1)]
    # decision_predictions_dev_unseen = [DECISION_CLASSES[cls] for cls in np.argmax(decision_logits_dev_unseen, axis=1)]
    # decision_predictions_test_unseen = [DECISION_CLASSES[cls] for cls in np.argmax(decision_logits_test_unseen, axis=1)]
    # dev_qa_seen = _loads_json(data_args.validation_qa_seen_file)
    # test_qa_seen = _loads_json(data_args.test_qa_seen_file)
    # dev_qa_unseen = _loads_json(data_args.validation_qa_unseen_file)
    # test_qa_unseen = _loads_json(data_args.test_qa_unseen_file)
    #
    # dataset_dev_seen = _load_dataset(validation_span_data_seen)
    # dataset_test_seen = _load_dataset(test_span_data_seen)
    # dataset_dev_unseen = _load_dataset(validation_span_data_unseen)
    # dataset_test_unseen = _load_dataset(test_span_data_unseen)

    # data_files = {
    #     # "train": data_args.train_file,
    #     "validation": data_args.validation_file,
    #     "test": data_args.test_file,
    #               }
    # for key in data_files.keys():
    #     logger.info(f"load a local file for {key}: {data_files[key]}")
    #
    # if data_args.validation_file.endswith(".csv"):
    #     # Loading a dataset from local csv files
    #     datasets = load_dataset("csv", data_files=data_files)
    # else:
    #     # Loading a dataset from local json files
    #     datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
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
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        spans = examples['answer']
        snippets = examples['span_snippet']
        targets = examples['answer']
        inputs = [span_i + tokenizer.sep_token + snippet_i for span_i, snippet_i in zip(spans, snippets)]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    max_target_length = data_args.val_max_target_length
    if data_args.max_val_samples is not None:
        dataset_dev = dataset_dev.select(range(data_args.max_val_samples))
        dataset_test = dataset_test.select(range(data_args.max_val_samples))
    dataset_dev = dataset_dev.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    dataset_test = dataset_test.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    # dataset_dev_seen = dataset_dev_seen.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )
    # dataset_test_seen = dataset_test_seen.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )
    # dataset_dev_unseen = dataset_dev_unseen.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )
    # dataset_test_unseen = dataset_test_unseen.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    def compute_metrics(preds, golds):
        import evaluator
        with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
            json.dump(preds, fp)
            fp.flush()
            json.dump(golds, fg)
            fg.flush()
            results = evaluator.evaluate(fg.name, fp.name, mode='combined')
            # results['combined'] = results['macro_accuracy'] * results['bleu_4']
        return results

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Evaluation
    tasks = ['dev', 'test']
    eval_datasets = [dataset_dev, dataset_test]
    all_decision_predictions = [decision_predictions_dev, decision_predictions_test]
    qa_datasets = [dev_qa, test_qa]

    # tasks = ['dev_seen', 'dev_unseen', 'test_seen', 'test_unseen']
    # eval_datasets = [dataset_dev_seen, dataset_dev_unseen, dataset_test_seen, dataset_test_unseen]
    # all_decision_predictions = [decision_predictions_dev_seen, decision_predictions_dev_unseen, decision_predictions_test_seen, decision_predictions_test_unseen]
    # qa_datasets = [dev_qa_seen, dev_qa_unseen, test_qa_seen, test_qa_unseen]

    for eval_dataset, task, decision_predictions, qa_dataset in zip(eval_datasets, tasks, all_decision_predictions, qa_datasets):
        eval_output = trainer.predict(test_dataset=eval_dataset)
        eval_predictions = tokenizer.batch_decode(eval_output.predictions, skip_special_tokens=True)
        eval_dataset.reset_format()
        predictions = []
        ground_truths = []
        for qg_pred, decision_pred, example, qa in zip(eval_predictions, decision_predictions, eval_dataset, qa_dataset):
            assert qa['utterance_id'] == example['id']
            curr_pred = {'utterance_id': example['id'],}
            if decision_pred == 'more':
                curr_pred['answer'] = qg_pred
            else:
                curr_pred['answer'] = decision_pred
            predictions.append(curr_pred)
            ground_truths.append({'utterance_id': example['id'], 'answer': qa['answer']})
        eval_results = compute_metrics(predictions, ground_truths)
        saving_file = data_args.validation_file if task == 'dev' else data_args.test_file
        # if 'dev' in task:
        #     saving_file = data_args.validation_file.replace('dev', task)
        # else:
        #     saving_file = data_args.test_file.replace('test', task)
        # ./out/v9_roberta-base_Lentail8.0_Ltrans4_seed27/predictions_dev_span.json
        prediction_eval_file = os.path.join(saving_file.replace("_span.json", "_e2e.json"))
        result_eval_file = os.path.join(saving_file.replace(f"predictions_{task}_span.json", f"results_{task}_e2e.json"))
        if trainer.is_world_process_zero():
            logger.info(f"***** Eval results {task} *****")
            for key, value in sorted(eval_results.items()):
                logger.info(f"  {key} = {value}")
            logger.info(f"Saving {prediction_eval_file}")
            logger.info(f"Saving {result_eval_file}")
            with open(prediction_eval_file, "w") as f:
                json.dump(predictions, f)
            with open(result_eval_file, "w") as f:
                json.dump(eval_results, f)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()