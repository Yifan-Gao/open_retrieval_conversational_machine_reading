#!/usr/bin/env bash

PYT=/research/dept7/ik_grp/yfgao/miniconda3/envs/opensharc/bin/python

function train() {
  export OMP_NUM_THREADS=1
  MODEL=roberta-base
  TOPK=100
  BZ=8
  MASPOT=$1
  SEED=$2
  $PYT -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port=$MASPOT run_open_sharc_rag.py \
    --train_qa_file=../data/sharc_raw/json/sharc_open_train.jsonl \
    --validation_qa_seen_file=../data/sharc_raw/json/sharc_open_dev_seen.jsonl \
    --validation_qa_unseen_file=../data/sharc_raw/json/sharc_open_dev_unseen.jsonl \
    --test_qa_seen_file=../data/sharc_raw/json/sharc_open_test_seen.jsonl \
    --test_qa_unseen_file=../data/sharc_raw/json/sharc_open_test_unseen.jsonl \
    --train_retrieval_file=../data/tfidf/train.json \
    --validation_retrieval_seen_file=../data/tfidf/dev_seen.json \
    --validation_retrieval_unseen_file=../data/tfidf/dev_unseen.json \
    --test_retrieval_seen_file=../data/tfidf/test_seen.json \
    --test_retrieval_unseen_file=../data/tfidf/test_unseen.json \
    --snippet_file=../data/sharc_raw/json/sharc_open_id2snippet.json \
    --top_k_snippets=$TOPK \
    --model_name_or_path=$MODEL \
    --output_dir=./out/"${MODEL}-seed${SEED}" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluation_strategy=epoch \
    --per_device_train_batch_size=$BZ \
    --per_device_eval_batch_size=$BZ \
    --learning_rate=3e-5 \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --num_train_epochs=10 \
    --warmup_steps=100 \
    --logging_steps=100 \
    --seed=${SEED} \
    --fp16 \
    --metric_for_best_model=combined \
    --load_best_model_at_end \
    --overwrite_cache
}

MASPOT=$1
SEED=$2
train $MASPOT $SEED





