import os
import re
import torch
import json
from pprint import pprint
from argparse import ArgumentParser
from model.decision import Module as Module_decision
from model.span import Module as Module_span
from e2e_preprocess import preprocess_tokenize, preprocess_decision, preprocess_span
from tqdm import tqdm
from unilmqg.biunilm.decode_seq2seq import main as qg_s2s
from tempfile import NamedTemporaryFile


def compute_metrics(preds, data):
    import evaluator
    with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
        json.dump(preds, fp)
        fp.flush()
        json.dump([{'utterance_id': e['utterance_id'], 'answer': e['answer']} for e in data], fg)
        fg.flush()
        results = evaluator.evaluate(fg.name, fp.name, mode='combined')
        # results['combined'] = results['macro_accuracy'] * results['bleu_4']
        return results


def preprocess_qg(preds, data):
    utterance2answer = {}
    for pred in preds:
        utterance2answer[pred['utterance_id']] = pred['answer']
    qg_data = []
    lines = []
    for ex in data:
        if utterance2answer[ex['utterance_id']].lower() not in ['yes', 'no', 'irrelevant']:
            src_pred_i = ' '.join([ex['snippet'], '[SEP]', utterance2answer[ex['utterance_id']].lower()]).replace('\n', ' ').strip()
            ex = {
                'utterance_id': ex['utterance_id'],
                'src': src_pred_i,
            }
            qg_data.append(ex)
            lines.append(src_pred_i)
    return qg_data, lines


def merge_edits(preds, qgpreds):
    # note: this happens in place
    qg = {p['utterance_id']: p for p in qgpreds}
    for p in preds:
        # p['orig_answer'] = p['answer']
        if p['utterance_id'] in qg:
            p['answer'] = qg[p['utterance_id']]['tgt']
    return preds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fin', default='/research/dept7/ik_grp/yfgao/sharc_ern/sharc_codalab/sharc_parsed.json', help='input data file')
    parser.add_argument('--dout', default='/research/dept7/ik_grp/yfgao/sharc_ern/sharc_codalab/sharc_output.json', help='directory to store output files')
    parser.add_argument('--model_decision', default='/research/dept7/ik_grp/yfgao/sharc_ern/sharc_codalab/model_decision.pt', help='decision model to use')
    parser.add_argument('--model_span', default='/research/dept7/ik_grp/yfgao/sharc_ern/sharc_codalab/model_span.pt', help='span model to use')
    parser.add_argument('--model_roberta_path', default='/opt/models/bert-base-uncased.tar.gz', help='bert model to use')
    parser.add_argument('--device', default='cuda', help='cpu not supported')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    parser.add_argument('--pretrained_lm_path', default='./pretrained_models/roberta_base/', help='path/to/pretrained/lm')

    # copy from unilm
    parser.add_argument("--bert_model", default='bert-large-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default='/research/king3/ik_grp/yfgao/pretrain_models/20200304_qg.bin',
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--cache_path", default='/opt/models', type=str,
                        help="Yifan added, bert vocab path")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', default=True,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    # parser.add_argument('--batch_size', type=int, default=2,
    #                     help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=48,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")

    args = parser.parse_args()

    print('loading raw file ...')
    with open(f"./data/{args.fin}_snippet_parsed.json") as f:
        sharc_data_parsed = json.load(f)
    with open(f"./data/sharc_raw/json/sharc_{args.fin}.json") as f:
        sharc_data = json.load(f)
    for ex in sharc_data:
        ex['parsed'] = sharc_data_parsed[ex['utterance_id']]
    print('Done')

    if args.debug:
        processed_data = torch.load('./out/e2e_processed_{}.pt'.format(args.fin))
    else:
        print('tokenization')
        processed_data = preprocess_tokenize(sharc_data)

        print('preprocessing data for decision')
        processed_data = preprocess_decision(processed_data)

        print('preprocessing data for span extraction')
        processed_data = preprocess_span(processed_data)

    if args.eval:
        torch.save(processed_data, './out/e2e_processed_{}.pt'.format(args.fin))

    utterance2answer = {}

    # decision prediction
    print('{} samples for decision prediction'.format(len(processed_data)))
    args_overwrite_decision = {
        'model_roberta_path': args.model_roberta_path,
        'model': 'decision',
        'dev_batch': args.batch_size,
        'device': args.device,
        'pretrained_lm_path': args.pretrained_lm_path,
    }
    model_decision = Module_decision.load(args.model_decision, override_args=args_overwrite_decision)
    model_decision.device = args.device
    model_decision.to(args.device)
    print('start decision prediction')
    model_decision_preds = model_decision.run_pred(processed_data)
    for pred in model_decision_preds:
        utterance2answer[pred['utterance_id']] = pred['pred_answer']
    del model_decision

    # span extraction
    data_span = []
    for ex in processed_data:
        if utterance2answer[ex['utterance_id']] == 'more':
            ex['entail'] = ex['entail_span']
            data_span.append(ex)
    print('{} samples for span extraction'.format(len(data_span)))
    args_overwrite_span = {
        'model_roberta_path': args.model_roberta_path,
        'model': 'span',
        'dev_batch': args.batch_size,
        'device': args.device,
        'pretrained_lm_path': args.pretrained_lm_path,
    }
    model_span = Module_span.load(args.model_span, override_args=args_overwrite_span)
    model_span.device = args.device
    model_span.to(args.device)
    print('start span extraction')
    model_span_preds = model_span.run_pred(data_span)
    for pred in model_span_preds:
        utterance2answer[pred['utterance_id']] = pred['answer']
    del model_span

    final_predictions = [{'utterance_id': k, 'answer': v} for k, v in utterance2answer.items()]

    span_results = compute_metrics(final_predictions, sharc_data)
    pprint(span_results)

    # QG
    sharc_pred = final_predictions
    qg_data, input_lines = preprocess_qg(sharc_pred, sharc_data)
    print('qg_data {}, input_lines {}'.format(len(qg_data), len(input_lines)))
    output_lines = qg_s2s(opt=args, inputs=input_lines)
    print("output_lines {}".format(len(output_lines)))
    qg_preds = []
    for ex, input_line, output_line in zip(qg_data, input_lines, output_lines):
        assert ex['src'] == input_line
        ex['tgt'] = output_line
        qg_preds.append(ex)
    e2e_preds = merge_edits(sharc_pred, qg_preds)

    e2e_results = compute_metrics(e2e_preds, sharc_data)
    pprint(e2e_results)
