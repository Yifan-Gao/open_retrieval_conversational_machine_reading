#!/usr/bin/env python
import os
import torch
import string
import json
from tqdm import tqdm
import editdistance
from transformers import RobertaTokenizer
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

MATCH_IGNORE = {'do', 'did', 'does',
                'is', 'are', 'was', 'were', 'have', 'will', 'would',
                '?', }
PUNCT_WORDS = set(string.punctuation)


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


def _decode(tokenizer, doc):
    decoded = tokenizer.decode(doc, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip('\n').strip()
    return decoded


def filter_token(tokenizer, text):
    filtered_text = []
    for token_id in text:
        if _decode(tokenizer, token_id).lower() not in MATCH_IGNORE and _decode(tokenizer, token_id).strip() != "":
            filtered_text.append(token_id)
    return _decode(tokenizer, filtered_text)


def get_span(tokenizer, context, answer):
    answer = filter_token(tokenizer, answer)
    best, best_score = None, float('inf')
    stop = False
    for i in range(len(context)):
        if stop:
            break
        for j in range(i, len(context)):
            chunk = filter_token(tokenizer, context[i:j+1])
            if '\n' in chunk or '*' in chunk:  # do not extract span across sentences/bullets
                continue
            score = editdistance.eval(answer, chunk)
            if score < best_score or (score == best_score and j-i < best[1]-best[0]):
                best, best_score = (i, j), score
            if chunk == answer:
                stop = True
                break
    if best:
        s, e = best
        while (not _decode(tokenizer, context[s]).strip() or _decode(tokenizer, context[s]) in PUNCT_WORDS) and s < e:
            s += 1
        while (not _decode(tokenizer, context[e]).strip() or _decode(tokenizer, context[e]) in PUNCT_WORDS) and s < e:
            e -= 1
        return s, e, _decode(tokenizer, context[s:e+1]), best_score
    else:
        return -1, -1, "", best_score

def merge_edus(id2snippet_parsed):
    for id, snippet_parsed in id2snippet_parsed.items():
        if snippet_parsed['has_edu']:
            id2snippet_parsed[id]['edus'] = [_merge_edus(_edu) for _edu in snippet_parsed['edus']]
    return id2snippet_parsed


def _merge_edus(edus):
    # v2. merge edu with its beforehand one except
    # 1) this edu is not starting with 'if', 'and', 'or', 'to', 'unless', or
    # 2) its beforehand edu is end with ',', '.', ':'
    special_toks = ['if ', 'and ', 'or ', 'to ', 'unless ', 'but ', 'as ', 'except ']
    special_puncts = ['.', ':', ',',]
    spt_idx = []
    for idx, edu in enumerate(edus):
        if idx == 0:
            continue
        is_endwith = False
        for special_punct in special_puncts:
            if edus[idx-1].strip().endswith(special_punct):
                is_endwith = True
        is_startwith = False
        for special_tok in special_toks:
            if edu.startswith(special_tok):
                is_startwith = True
        if (not is_endwith) and (not is_startwith):
            spt_idx.append(idx)
    edus_spt = []
    for idx, edu in enumerate(edus):
        if idx not in spt_idx or idx == 0:
            edus_spt.append(edu)
        else:
            edus_spt[-1] += ' ' + edu
    return edus_spt


def _extract_edus(snippet_parsed, title_tokenized, sentences_tokenized, tokenizer):
    # return a nested tokenized edus, with (start, end) index for each edu
    edus_span = []  # for all sentences
    edus_tokenized = []
    # add title
    if snippet_parsed['title'].strip('\n').strip() != '':
        edus_tokenized.append([title_tokenized])
        edus_span.append([(0,len(title_tokenized)-1)])

    if snippet_parsed['is_bullet']:
        for sentence_tokenized in sentences_tokenized:
            edus_tokenized.append([sentence_tokenized])
            edus_span.append([(0, len(sentence_tokenized) - 1)])
    else:
        for idx_sentence in range(len(sentences_tokenized)):
            edus_span_i = []  # for i-th sentence
            edus_tokenized_i = []
            current_edus = snippet_parsed['edus'][idx_sentence]
            current_sentence_tokenized = sentences_tokenized[idx_sentence]

            p_start, p_end = 0, 0
            for edu in current_edus:
                edu = edu.strip('\n').strip().replace(' ', '').lower()
                # handle exception case train 261
                if ('``' in edu) and ('\'\'' in edu):
                    edu = edu.replace('``', '"').replace('\'\'', '"')
                for p_sent in range(p_start, len(current_sentence_tokenized)):
                    sent_span = _decode(tokenizer, current_sentence_tokenized[p_start:p_sent+1]).replace(' ', '').lower()
                    if edu == sent_span:
                        p_end = p_sent
                        edus_span_i.append((p_start, p_end))  # [span_s,span_e]
                        edus_tokenized_i.append(current_sentence_tokenized[p_start:p_end + 1])
                        p_start = p_end + 1
                        break
            assert len(current_edus) == len(edus_tokenized_i) == len(edus_span_i)
            assert p_end == len(current_sentence_tokenized) - 1
            edus_span.append(edus_span_i)  # [sent_idx, ]
            edus_tokenized.append(edus_tokenized_i)
    assert len(edus_span) == len(edus_tokenized) == len(sentences_tokenized) + int(title_tokenized != None)

    return edus_tokenized, edus_span


def extract_edus(fuqs, snippet_parsed, tokenized_data, tokenizer):
    # data_raw -> fuqs
    # all_edus -> snippet_parsed

    # assert data_raw['snippet'] == all_edus['snippet']

    output = {}

    # 1. tokenize all sentences
    if snippet_parsed['title'].strip('\n').strip() != '':
        title_tokenized = tokenized_data['titles'][snippet_parsed['title']]
    else:
        title_tokenized = None
    sentences_tokenized = [tokenized_data['clauses'][s] for s in snippet_parsed['clauses']]
    output['clause_t'] = [title_tokenized] + sentences_tokenized if title_tokenized else sentences_tokenized
    output['edu_t'], output['edu_span'] = _extract_edus(snippet_parsed, title_tokenized, sentences_tokenized, tokenizer)

    # 2. map question to edu
    # iterate all sentences, select the one with minimum edit distance
    output['q2clause'] = {}
    output['clause2q'] = [[] for _ in output['clause_t']]
    output['q2edu'] = {}
    output['edu2q'] = [[] for _ in output['edu_t']]
    for idx, sent in enumerate(output['edu_t']):
        output['edu2q'][idx].extend([[] for _ in sent])
    for question in fuqs:
        if "<fuq> " + question in tokenized_data['follow_up_questions']:
            question_tokenized = tokenized_data['follow_up_questions']["<fuq> " + question]
        else:
            question_tokenized = tokenized_data['inquire_answers'][question]
        all_editdist = []
        for idx, clause in enumerate(output['clause_t']):
            start, end, span_text, editdist = get_span(tokenizer, clause, question_tokenized)  # [s,e] both inclusive
            all_editdist.append((idx, start, end, span_text, editdist))

        # take the minimum one
        clause_id, clause_start, clause_end, clause_span_text, clause_dist = sorted(all_editdist, key=lambda x: x[-1])[0]
        output['q2clause'][question] = {
            'clause_id': clause_id,
            'clause_start': clause_start,  # [s,e] both inclusive
            'clause_end': clause_end,
            'clause_dist': clause_dist,
            'clause_span_text': clause_span_text
        }
        output['clause2q'][clause_id].append(question)

        # mapping to edus
        extract_span = set(range(output['q2clause'][question]['clause_start'],
                                 output['q2clause'][question]['clause_end'] + 1))
        output['q2edu'][question] = {
            'clause_id': output['q2clause'][question]['clause_id'],
            'edu_id': [],  # (id, overlap_toks)
        }

        for idx, span in enumerate(output['edu_span'][output['q2clause'][question]['clause_id']]):
            current_span = set(range(span[0], span[1] + 1))
            if extract_span.intersection(current_span):
                output['q2edu'][question]['edu_id'].append((idx, len(extract_span.intersection(current_span))))
                output['edu2q'][output['q2clause'][question]['clause_id']][idx].append(question)
        sorted_edu_id = sorted(output['q2edu'][question]['edu_id'], key=lambda x: x[-1], reverse=True)
        top_edu_id = sorted_edu_id[0][0]
        top_edu_span = output['edu_span'][output['q2clause'][question]['clause_id']][top_edu_id]
        top_edu_start = max(output['q2clause'][question]['clause_start'], top_edu_span[0])
        top_edu_end = min(output['q2clause'][question]['clause_end'], top_edu_span[1])
        output['q2edu'][question]['top_edu_id'] = top_edu_id
        output['q2edu'][question]['top_edu_start'] = top_edu_start
        output['q2edu'][question]['top_edu_end'] = top_edu_end  # [s,e] both inclusive
    assert len(output['q2edu']) == len(output['q2clause']) == len(fuqs)
    return output


# get tokenized passages
def _get_tokenized_kv(original, tokenized):
    out = {}
    for _q, _v in zip(original, tokenized['input_ids']):
        out[_q] = _v
    return out

# get retrieval results
def _load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# get sharc data
def _loads_json(path):
    with open(path, 'r', ) as f:
        dataset = []
        for idx, line in enumerate(f):
            example = json.loads(line)
            dataset.append(example)
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    id2snippet = _load_json(data_args.snippet_file)
    id2snippet_parsed = _load_json(data_args.snippet_file.replace(".json", "_parsed.json"))
    snippet2id = {v: k for k, v in id2snippet.items()}
    dataset_train = _loads_json(data_args.train_qa_file)
    dataset_validation_seen = _loads_json(data_args.validation_qa_seen_file)
    dataset_validation_unseen = _loads_json(data_args.validation_qa_unseen_file)
    dataset_test_seen = _loads_json(data_args.test_qa_seen_file)
    dataset_test_unseen = _loads_json(data_args.test_qa_unseen_file)
    dataset_all = sum([dataset_train, dataset_validation_seen, dataset_validation_unseen, dataset_test_seen, dataset_test_unseen], [])

    retrieval_snippet_id_train = _load_json(data_args.train_retrieval_file)
    retrieval_snippet_id_dev_seen = _load_json(data_args.validation_retrieval_seen_file)
    retrieval_snippet_id_dev_unseen = _load_json(data_args.validation_retrieval_unseen_file)
    retrieval_snippet_id_test_seen = _load_json(data_args.test_retrieval_seen_file)
    retrieval_snippet_id_test_unseen = _load_json(data_args.test_retrieval_unseen_file)


    # merge merge_edus()
    id2snippet_parsed = merge_edus(id2snippet_parsed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        additional_special_tokens=['<qu>', '<sc>', '<sn>', '<fuq>', '<fua>', '<fqa>', '<ssep>'],
    )

    tokenized_sharc_path = f'./data/{model_args.model_name_or_path}-tokenized.json'
    if os.path.exists(tokenized_sharc_path):
        with open(tokenized_sharc_path) as f:
            tokenized_sharc_data = json.load(f)
    else:
        os.makedirs('./data', exist_ok=True)
        # '<qu>', '<sc>', '<sn>', '<fuq>', '<fua>', '<ina>'
        questions = list(set(["<qu> " + _ex['question'] for _ex in dataset_all]))
        scenarios = list(set(["<sc> " + _ex['scenario'] for _ex in dataset_all]))
        # snippets = list(set(["<sn> " + _ex['snippet'] for _ex in dataset_all]))
        follow_up_questions = list(set(
            ["<fuq> " + fuqa['follow_up_question'] for _ex in dataset_all for fuqa in _ex['history']] +
            ["<fuq> " + fuqa['follow_up_question'] for _ex in dataset_all for fuqa in _ex['evidence']]))
        follow_up_answers = ["<fua> yes", "<fua> no", ]
        inquire_answers = list(set([_ex['answer'] for _ex in dataset_all]))
        clauses, titles = [], []
        for _ex in id2snippet_parsed.values():
            clauses.extend(_ex['clauses'])
            titles.append(_ex['title'])
        clauses = list(set(clauses))
        titles = list(set(titles))

        questions_tokenized = tokenizer(questions, add_special_tokens=False)
        scenarios_tokenized = tokenizer(scenarios, add_special_tokens=False)
        # snippets_tokenized = tokenizer(snippets, add_special_tokens=False)
        follow_up_questions_tokenized = tokenizer(follow_up_questions, add_special_tokens=False)
        follow_up_answers_tokenized = tokenizer(follow_up_answers, add_special_tokens=False)
        inquire_answers_tokenized = tokenizer(inquire_answers, add_special_tokens=False)
        # edus_tokenized = tokenizer(edus, add_special_tokens=False)
        clauses_tokenized = tokenizer(clauses, add_special_tokens = False)
        titles_tokenized = tokenizer(titles, add_special_tokens = False)
        tokenized_sharc_data = {
            'questions': _get_tokenized_kv(questions, questions_tokenized),
            'scenarios': _get_tokenized_kv(scenarios, scenarios_tokenized),
            # 'snippets': _get_tokenized_kv(snippets, snippets_tokenized),
            'follow_up_questions': _get_tokenized_kv(follow_up_questions, follow_up_questions_tokenized),
            'follow_up_answers': _get_tokenized_kv(follow_up_answers, follow_up_answers_tokenized),
            'inquire_answers': _get_tokenized_kv(inquire_answers, inquire_answers_tokenized),
            # 'edus': _get_tokenized_kv(edus, edus_tokenized),
            'clauses': _get_tokenized_kv(clauses, clauses_tokenized),
            'titles': _get_tokenized_kv(titles, titles_tokenized),
        }

        with open(tokenized_sharc_path, 'w') as f:
            json.dump(tokenized_sharc_data, f)
        print(f"Saving tokenized sharc data {tokenized_sharc_path}")

    # construct dialog trees
    tree2fuq = {}
    tree2snippet = {}
    for _ex in dataset_all:
        if _ex['tree_id'] not in tree2fuq:
            tree2fuq[_ex['tree_id']] = set()
        for h in _ex['history'] + _ex['evidence']:
            tree2fuq[_ex['tree_id']].add(h['follow_up_question'])
        if _ex['answer'].lower() not in ['yes', 'no', 'irrelevant']:
            tree2fuq[_ex['tree_id']].add(_ex['answer'])
        if 'tree_id' not in tree2snippet:
            tree2snippet[_ex['tree_id']] = _ex['snippet']
        else:
            assert tree2snippet[_ex['tree_id']] == _ex['snippet'], f"{tree2snippet[_ex['tree_id']]}\n{_ex['snippet']}"

    processed_tree_path = f'./data/{model_args.model_name_or_path}-tree-mapping.json'
    # for k in tqdm(tree2fuq.keys()):
    #     extract_edus(tree2fuq[k], id2snippet_parsed[snippet2id[tree2snippet[k]]], tokenized_sharc_data, tokenizer)
    # with open(processed_tree_path) as f:
    #     tree_mapping_data = json.load(f)
    # snippetid2snippetparsed = {}
    # for k, v in tree_mapping_data.items():
    #     if snippet2id[v['snippet']] not in snippetid2snippetparsed:
    #         snippetid2snippetparsed[snippet2id[v['snippet']]] = {k: v['processed_snippet']}
    #     else:
    #         if k not in snippetid2snippetparsed[snippet2id[v['snippet']]]:
    #             snippetid2snippetparsed[snippet2id[v['snippet']]][k] = v['processed_snippet']
    # from evaluator import MoreEvaluator, prepro
    # dev_preds, dev_golds = [], []
    # for ex, retrieval_ids in zip(dataset_validation_seen+dataset_validation_unseen, retrieval_snippet_id_dev_seen+retrieval_snippet_id_dev_unseen):
    #     if ex['answer'].lower() not in ['yes', 'no', 'irrelevant']:
    #         top1_snippet_id = retrieval_ids[0][0]
    #         if ex['tree_id'] in snippetid2snippetparsed[top1_snippet_id]:
    #             m = snippetid2snippetparsed[top1_snippet_id][ex['tree_id']]
    #             span = m['q2clause'][ex['answer']]['clause_span_text']
    #         else:
    #             if snippet2id[tree2snippet[ex['tree_id']]] in retrieval_ids[0][:3]:
    #                 m = snippetid2snippetparsed[snippet2id[tree2snippet[ex['tree_id']]]][ex['tree_id']]
    #                 span = m['q2clause'][ex['answer']]['clause_span_text']
    #             else:
    #                 span = ""
    #         dev_preds.append(prepro(span))
    #         dev_golds.append(prepro(ex['answer']))
    # evaluator = MoreEvaluator()
    # metrics = evaluator.evaluate(dev_golds, dev_preds)
    # print(metrics)
    #
    # discern_dev = torch.load("/research/dept7/ik_grp/yfgao/open_retrieval_sharc_private/Discern_private/data/proc_span_roberta_base_dev.pt")
    # discern_dev_preds, discern_dev_golds = [], []
    # for e in discern_dev:
    #     if e['answer'].lower() not in ['yes', 'no', 'irrelevant']:
    #         span = _decode(tokenizer, e['entail']['inp'][e['entail']['answer_span'][0]:e['entail']['answer_span'][1] + 1])
    #         discern_dev_preds.append(prepro(span))
    #         discern_dev_golds.append(prepro(e['answer']))
    # evaluator = MoreEvaluator()
    # metrics = evaluator.evaluate(discern_dev_golds, discern_dev_preds)
    # print(metrics)
    #
    # nested = []
    # dev_unmatched_pred, dev_unmatched_gold = [], []
    # dev_matched_pred, dev_matched_gold = [], []
    # for i in range(len(dev_golds)):
    #     if dev_preds[i] != discern_dev_preds[i]:
    #         nested.append({
    #             'i': i,
    #             'mp_pred': dev_preds[i],
    #             'mp_gold': dev_golds[i],
    #             'di_pred': discern_dev_preds[i],
    #             'di_gold': discern_dev_golds[i],
    #         })
    #         dev_unmatched_pred.append(discern_dev_preds[i])
    #         dev_unmatched_gold.append(discern_dev_golds[i])
    #     else:
    #         dev_matched_pred.append(discern_dev_preds[i])
    #         dev_matched_gold.append(discern_dev_golds[i])
    # evaluator = MoreEvaluator()
    # metrics = evaluator.evaluate(dev_unmatched_gold, dev_unmatched_pred)
    # print(metrics)
    # evaluator = MoreEvaluator()
    # metrics = evaluator.evaluate(dev_matched_gold, dev_matched_pred)
    # print(metrics)

    processed_tree = {
        k: {
            "processed_snippet": extract_edus(tree2fuq[k], id2snippet_parsed[snippet2id[tree2snippet[k]]], tokenized_sharc_data, tokenizer),
            "follow_up_questions": list(tree2fuq[k]),
            "snippet": tree2snippet[k],
        } for k in tqdm(tree2fuq.keys())
    }
    print(f'saving {processed_tree_path}')
    with open(processed_tree_path, 'w') as f:
        json.dump(processed_tree, f)


if __name__ == "__main__":
    main()