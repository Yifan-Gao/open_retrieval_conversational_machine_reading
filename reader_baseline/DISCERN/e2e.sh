#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

PYT_UNILM=/research/king3/yfgao/miniconda3/envs/unilm/bin/python

${PYT_UNILM} e2e_inference.py \
./out/e2e_dev_parsed.json \
./out/e2e_dev_prediction.json \
--model_decision=./pretrained_models/decision.pt \
--model_span=./pretrained_models/span.pt \
--model_roberta_path=./pretrained_models/roberta_base/ \
--model_recover_path=./pretrained_models/unilm/unilmqg.bin \
--cache_path=./pretrained_models/unilm/ \
--batch_size=5

PYT_SHARC=/research/king3/yfgao/miniconda3/envs/sharc/bin/python

${PYT_SHARC} evaluator.py ./out/e2e_dev_prediction.json ./data/sharc_raw/json/sharc_dev.json