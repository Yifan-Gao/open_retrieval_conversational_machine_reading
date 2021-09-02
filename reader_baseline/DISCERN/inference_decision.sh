#!/usr/bin/env bash


PYT=/research/king3/yfgao/miniconda3/envs/sharc_pyt1d1/bin/python

export CUDA_VISIBLE_DEVICES=$1

${PYT} -u train_sharc.py \
--dsave="./out/{}" \
--model=decision \
--data=./data/ \
--data_type=decision_roberta_base \
--prefix=inference_decision \
--resume=./out/train_decision/best.pt \
--pretrained_lm_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base \
--trans_layer=2 \
--test

