#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

PYT_SHARC=/research/king3/yfgao/miniconda3/envs/sharc_pyt1d1/bin/python

export CUDA_VISIBLE_DEVICES=$1
SEED=$2

${PYT_SHARC} -u train_sharc.py \
--train_batch=16 \
--gradient_accumulation_steps=2 \
--epoch=5 \
--seed=115 \
--learning_rate=5e-5 \
--dsave="out/{}_seed_${SEED}" \
--model=span \
--early_stop=dev_0_combined \
--pretrained_lm_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base \
--data=./data/ \
--data_type=span_roberta_base \
--prefix=train_span \
--eval_every_steps=100

