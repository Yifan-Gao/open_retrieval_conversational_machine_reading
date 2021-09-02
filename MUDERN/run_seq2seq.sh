#!/usr/bin/env bash

PYT=${OPENS}/bin/python

function train() {
  export OMP_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=$1
  MODEL=facebook/bart-base
  SEED=$2
  BZ=32
  EPOCH=$3
  LR=$4
  OUT="seq2seq_${MODEL}_BZ${BZ}_EP${EPOCH}_LR${LR}_seed${SEED}"
  echo "=================================="
  echo "=================${OUT}================="
  echo "=================================="

  $PYT run_seq2seq.py \
    --train_file=../data/sharc_raw/json/sharc_open_question_generation_train.jsonl \
    --validation_file=../data/sharc_raw/json/sharc_open_question_generation_dev.jsonl \
    --test_file=../data/sharc_raw/json/sharc_open_question_generation_test.jsonl \
    --model_name_or_path=${MODEL} \
    --output_dir=./out/${OUT} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluation_strategy=epoch \
    --per_device_train_batch_size=$BZ \
    --per_device_eval_batch_size=$BZ \
    --learning_rate=$LR \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --num_train_epochs=$EPOCH \
    --warmup_steps=16 \
    --logging_steps=16 \
    --seed=${SEED} \
    --fp16 \
    --metric_for_best_model=bleu \
    --load_best_model_at_end \
    --predict_with_generate \
    --overwrite_cache
}

CUDA_VISIBLE_DEVICES=$1
EPOCH=$2
LR=$3

SEED=$4

train $CUDA_VISIBLE_DEVICES $SEED $EPOCH $LR


