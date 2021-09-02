#!/usr/bin/env bash

PYT=${OPENS}/bin/python

function train() {
  export OMP_NUM_THREADS=1
#  MODEL=roberta-base
  MASPOT=$1
#  SEED=$2
  TOPK=100
  BZ=8
  SPAN_MODEL=$2
  DECISION_MODEL=$3
#  OUT="span_${MODEL}_BZ${BZ}_EP${EPOCH}_LR${LR}_MAXTOPK${TOPK}_seed${SEED}"
  OUT=$DECISION_MODEL
  MODEL=$SPAN_MODEL
  echo "=================================="
  echo "=================${OUT}================="
  echo "=================================="

  $PYT -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port=$MASPOT inference_span.py \
    --validation_qa_seen_file=../data/sharc_raw/json/sharc_open_dev_seen.jsonl \
    --validation_qa_unseen_file=../data/sharc_raw/json/sharc_open_dev_unseen.jsonl \
    --test_qa_seen_file=../data/sharc_raw/json/sharc_open_test_seen.jsonl \
    --test_qa_unseen_file=../data/sharc_raw/json/sharc_open_test_unseen.jsonl \
    --validation_retrieval_seen_file=../data/tfidf/dev_seen.json \
    --validation_retrieval_unseen_file=../data/tfidf/dev_unseen.json \
    --test_retrieval_seen_file=../data/tfidf/test_seen.json \
    --test_retrieval_unseen_file=../data/tfidf/test_unseen.json \
    --snippet_file=../data/sharc_raw/json/sharc_open_id2snippet.json \
    --tokenized_file=./data/roberta-base-tokenized.json \
    --tree_mapping_file=./data/roberta-base-tree-mapping.json \
    --top_k_snippets=$TOPK \
    --model_name_or_path=$MODEL \
    --output_dir=${OUT} \
    --overwrite_output_dir \
    --per_device_train_batch_size=$BZ \
    --per_device_eval_batch_size=$BZ \
    --logging_steps=100 \
    --fp16 \
    --overwrite_cache \
    --validation_qa_prediction_file=${OUT}/predictions_dev.npy \
    --test_qa_prediction_file=${OUT}/predictions_test.npy
}

MASPOT=5656
SEED=11
#SEED SPANCKPT
#27	1106
#95	1264
#19	948
#87	1580
#11	948
SPAN_MODEL=/research/dept7/ik_grp/lijj/open_retrieval_sharc_private/mpdiscern/out/span_roberta-base_BZ8_EP10_LR5e-5_MAXTOPK100_seed${SEED}/checkpoint-948
DECISION_MODEL=/research/dept7/ik_grp/yfgao/open_retrieval_sharc_private/mpqa/out/roberta-base-seed${SEED}

train $MASPOT $SPAN_MODEL $DECISION_MODEL


