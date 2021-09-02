#!/usr/bin/env bash

PYT=${OPENS}/bin/python

function train() {
  export OMP_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=$1
  S2S_MODEL=$2
  DECISION_MODEL=$3
  OUT=$DECISION_MODEL
  MODEL=$S2S_MODEL
  echo "=================================="
  echo "=================${OUT}================="
  echo "=================================="

  $PYT inference_seq2seq.py \
    --validation_qa_seen_file=../data/sharc_raw/json/sharc_open_dev_seen.jsonl \
    --validation_qa_unseen_file=../data/sharc_raw/json/sharc_open_dev_unseen.jsonl \
    --test_qa_seen_file=../data/sharc_raw/json/sharc_open_test_seen.jsonl \
    --test_qa_unseen_file=../data/sharc_raw/json/sharc_open_test_unseen.jsonl \
    --validation_file=${OUT}/predictions_dev_span.json \
    --test_file=${OUT}/predictions_test_span.json \
    --model_name_or_path=${MODEL} \
    --output_dir=${OUT} \
    --overwrite_output_dir \
    --do_eval \
    --per_device_eval_batch_size=32 \
    --predict_with_generate
}

CUDA_VISIBLE_DEVICES=$1
SEED=$2
#seed	BLEU	CKPT
#27	90.1	672
#95	90.07	864
#19	90.07	928
#87	90.09	672
#11	90.07	928
CKPT=$3

S2S_MODEL=./out/seq2seq_facebook/bart-base_BZ32_EP30_LR1e-4_seed${SEED}/checkpoint-${CKPT}
DECISION_MODEL=/research/dept7/ik_grp/yfgao/open_retrieval_sharc_private/mpqa/out/roberta-base-seed${SEED}

train $CUDA_VISIBLE_DEVICES $S2S_MODEL $DECISION_MODEL


