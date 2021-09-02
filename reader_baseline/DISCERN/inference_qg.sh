#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

PYT_UNILM=/research/king3/yfgao/miniconda3/envs/unilm/bin/python
#
#HOME="/research/dept7/ik_grp/yfgao/sharc_ern/discern_data"
#
#DATA='sent_raw_final_iq_usersep_span_roberta_base'
#TIME=20201105
#TRAIN_BATCH=16
#MODEL='e2e_span_roberta_base'
#
#SEED=115
#SAVE_DIR="${HOME}/saved_models/${TIME}_${MODEL}_${DATA}"
#mkdir -p ${SAVE_DIR}
#PREFIX="bs_${TRAIN_BATCH}_seed_${SEED}"
#mkdir -p "${SAVE_DIR}/${PREFIX}"
#
#SHARC_PREDS=${SAVE_DIR}/${PREFIX}

${PYT_UNILM} -u qg.py \
--fin=./data/sharc_raw/json/sharc_dev.json \
--fpred=./out/inference_span \
--model_recover_path=/research/king3/yfgao/sharc/Discern_private/pretrained_models/unilm/unilmqg.bin \
--cache_path=/research/king3/yfgao/sharc/Discern_private/pretrained_models//unilm/

