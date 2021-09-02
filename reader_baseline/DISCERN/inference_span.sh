#!/usr/bin/env bash

PYT_SHARC=/research/king3/yfgao/miniconda3/envs/sharc_pyt1d1/bin/python

export CUDA_VISIBLE_DEVICES=$1

${PYT_SHARC} -u train_sharc.py \
--dsave="out/{}" \
--model=span \
--data=./data/ \
--data_type=span_roberta_base \
--prefix=inference_span \
--resume=./out/train_span/best.pt \
--pretrained_lm_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base \
--test

