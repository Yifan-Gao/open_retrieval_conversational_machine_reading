#!/usr/bin/env python
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

MAX_LEN=512
MODEL_FILE = '/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_FILE, cache_dir=None)


def roberta_encode(doc):
    encoded = tokenizer.encode(doc.strip('\n').strip(), add_prefix_space=True, add_special_tokens=False)
    return encoded


def roberta_decode(doc):
    decoded = tokenizer.decode(doc, clean_up_tokenization_spaces=False).strip('\n').strip()
    return decoded


def merge_edus(edus):
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


def _extract_edus(all_edus, title_tokenized, sentences_tokenized):
    # return a nested tokenized edus, with (start, end) index for each edu
    edus_span = []  # for all sentences
    edus_tokenized = []
    # add title
    if all_edus['title'].strip('\n').strip() != '':
        edus_tokenized.append([title_tokenized])
        edus_span.append([(0,len(title_tokenized)-1)])

    if all_edus['is_bullet']:
        for sentence_tokenized in sentences_tokenized:
            edus_tokenized.append([sentence_tokenized])
            edus_span.append([(0, len(sentence_tokenized) - 1)])
        return edus_tokenized, edus_span
    else:
        is_extracted_success = True
        edus_filtered = []
        for edus in all_edus['edus']:
            merged_edus = merge_edus(edus)
            edus_filtered.append(merged_edus)

        for idx_sentence in range(len(sentences_tokenized)):
            edus_span_i = []  # for i-th sentence
            edus_tokenized_i = []
            current_edus = edus_filtered[idx_sentence]
            current_sentence_tokenized = sentences_tokenized[idx_sentence]

            p_start, p_end = 0, 0
            for edu in current_edus:
                edu = edu.strip('\n').strip().replace(' ', '').lower()
                # handle exception case train 261
                if ('``' in edu) and ('\'\'' in edu):
                    edu = edu.replace('``', '"').replace('\'\'', '"')
                for p_sent in range(p_start, len(current_sentence_tokenized)):
                    sent_span = roberta_decode(current_sentence_tokenized[p_start:p_sent+1]).replace(' ', '').lower()
                    if edu == sent_span:
                        p_end = p_sent
                        edus_span_i.append((p_start, p_end))  # [span_s,span_e]
                        edus_tokenized_i.append(current_sentence_tokenized[p_start:p_end + 1])
                        p_start = p_end + 1
                        break
            if len(current_edus) == len(edus_tokenized_i) == len(edus_span_i) and p_end == len(current_sentence_tokenized) - 1:
                # assert len(current_edus) == len(edus_tokenized_i) == len(edus_span_i)
                # assert p_end == len(current_sentence_tokenized) - 1
                edus_span.append(edus_span_i)  # [sent_idx, ]
                edus_tokenized.append(edus_tokenized_i)
            else:
                is_extracted_success = False
        if not is_extracted_success:
            edus_span = []  # for all sentences
            edus_tokenized = []
            # add title
            if all_edus['title'].strip('\n').strip() != '':
                edus_tokenized.append([title_tokenized])
                edus_span.append([(0, len(title_tokenized) - 1)])
            for sentence_tokenized in sentences_tokenized:
                edus_tokenized.append([sentence_tokenized])
                edus_span.append([(0, len(sentence_tokenized) - 1)])
        return edus_tokenized, edus_span


def preprocess_tokenize(data):
    for ex in tqdm(data):
        ex['tok'] = {}
        title_tokenized = roberta_encode(ex['parsed']['title']) if ex['parsed']['title'].strip('\n').strip() != '' else None
        sentences_tokenized = [roberta_encode(s) for s in ex['parsed']['clauses']]
        ex['tok']['initial_question_t'] = roberta_encode(ex['question'])
        ex['tok']['scenario_t'] = roberta_encode(ex['scenario']) if ex['scenario'] != '' else None
        ex['tok']['snippet_t'] = roberta_encode(ex['snippet'])
        ex['tok']['clause_t'] = [title_tokenized] + sentences_tokenized if ex['parsed']['title'].strip('\n').strip() != '' else sentences_tokenized
        ex['tok']['q_t'] = {fqa['follow_up_question']: roberta_encode(fqa['follow_up_question']) for fqa in ex['history']}
        ex['tok']['edu_t'], ex['tok']['edu_span'] = _extract_edus(ex['parsed'], title_tokenized, sentences_tokenized)
    return data


def preprocess_decision(data):
    for ex in tqdm(data):
        sep = tokenizer.sep_token_id
        cls = tokenizer.cls_token_id
        pad = tokenizer.pad_token_id

        # snippet
        inp = []
        rule_idx = []  # here we record all rule idx, and question relevant idx
        for clause_id, edus in enumerate(ex['tok']['edu_t']):
            for edu_id, edu in enumerate(edus):
                if len(inp) < MAX_LEN:
                    rule_idx.append(len(inp))
                inp += [cls] + edu
        inp += [sep]

        # user info (scenario, dialog history)
        user_idx = []
        question_tokenized = ex['tok']['initial_question_t']
        if len(inp) < MAX_LEN: user_idx.append(len(inp))
        question_idx = len(inp)
        inp += [cls] + question_tokenized + [sep]
        scenario_idx = -1
        if ex['scenario'] != '':
            scenario_tokenized = ex['tok']['scenario_t']
            if len(inp) < MAX_LEN: user_idx.append(len(inp))
            scenario_idx = len(inp)
            inp += [cls] + scenario_tokenized + [sep]
        for fqa in ex['history']:
            if len(inp) < MAX_LEN: user_idx.append(len(inp))
            fq, fa = fqa['follow_up_question'], fqa['follow_up_answer']
            fa = 'No' if 'no' in fa.lower() else 'Yes'  # fix possible typos like 'noe'
            inp += [cls] + roberta_encode('question') + ex['tok']['q_t'][fq] + roberta_encode('answer') + roberta_encode(fa) + [sep]

        # all
        input_mask = [1] * len(inp)
        if len(inp) > MAX_LEN:
            inp = inp[:MAX_LEN]
            input_mask = input_mask[:MAX_LEN]
        while len(inp) < MAX_LEN:
            inp.append(pad)
            input_mask.append(0)

        ex['entail'] = {
            'inp': inp,
            'input_ids': torch.LongTensor(inp),
            'input_mask': torch.LongTensor(input_mask),
            'rule_idx': torch.LongTensor(rule_idx),
            'user_idx': torch.LongTensor(user_idx),
            'question_idx': question_idx,
            'scenario_idx': scenario_idx,
        }

    return data


def preprocess_span(data):
    for ex in tqdm(data):
        sep = tokenizer.sep_token_id
        cls = tokenizer.cls_token_id
        pad = tokenizer.pad_token_id

        span_pointer_mask = []

        # snippet
        inp = []
        rule_idx = []
        for clause_id, clause in enumerate(ex['tok']['clause_t']):
            if len(inp) < MAX_LEN:
                rule_idx.append(len(inp))
                inp += [cls] + clause
                span_pointer_mask += ([0] + [1] * len(clause))  # [0] for [CLS]
        inp += [sep]

        # user info (scenario, dialog history)
        user_idx = []
        question_tokenized = ex['tok']['initial_question_t']
        if len(inp) < MAX_LEN: user_idx.append(len(inp))
        question_idx = len(inp)
        inp += [cls] + question_tokenized + [sep]
        scenario_idx = -1
        if ex['scenario'] != '':
            scenario_tokenized = ex['tok']['scenario_t']
            if len(inp) < MAX_LEN: user_idx.append(len(inp))
            scenario_idx = len(inp)
            inp += [cls] + scenario_tokenized + [sep]
        for fqa in ex['history']:
            if len(inp) < MAX_LEN: user_idx.append(len(inp))
            fq, fa = fqa['follow_up_question'], fqa['follow_up_answer']
            fa = 'No' if 'no' in fa.lower() else 'Yes'  # fix possible typos like 'noe'
            inp += [cls] + roberta_encode('question') + ex['tok']['q_t'][fq] + roberta_encode('answer') + roberta_encode(fa) + [sep]
        span_pointer_mask += [0] * (len(inp) - len(span_pointer_mask))

        # all
        input_mask = [1] * len(inp)
        if len(inp) > MAX_LEN:
            inp = inp[:MAX_LEN]
            input_mask = input_mask[:MAX_LEN]
            span_pointer_mask = span_pointer_mask[:MAX_LEN]
        while len(inp) < MAX_LEN:
            inp.append(pad)
            input_mask.append(0)
            span_pointer_mask.append(0)

        ex['entail_span'] = {
            'inp': inp,
            'input_ids': torch.LongTensor(inp),
            'input_mask': torch.LongTensor(input_mask),
            'rule_idx': torch.LongTensor(rule_idx),
            'user_idx': torch.LongTensor(user_idx),
            'question_idx': question_idx,
            'scenario_idx': scenario_idx,
            'span_pointer_mask': torch.LongTensor(span_pointer_mask),
        }

    return data


