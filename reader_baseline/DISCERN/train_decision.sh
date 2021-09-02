#!/usr/bin/env bash

PYT=/research/king3/yfgao/miniconda3/envs/sharc_pyt1d1/bin/python

export CUDA_VISIBLE_DEVICES=$1
SEED=$2

${PYT} -u train_sharc.py \
--train_batch=16 \
--gradient_accumulation_steps=2 \
--epoch=5 \
--seed=${SEED} \
--learning_rate=5e-5 \
--loss_entail_weight=3.0 \
--dsave="out/{}_seed_${SEED}" \
--model=decision \
--early_stop=dev_0a_combined \
--pretrained_lm_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base \
--data=./data/ \
--data_type=decision_roberta_base \
--prefix=train_decision \
--trans_layer=2 \
--eval_every_steps=300  # 516



