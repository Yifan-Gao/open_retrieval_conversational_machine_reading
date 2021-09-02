#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
SEED=$2
SPLIT=$3
PYT_QG=/research/king3/yfgao/miniconda3/envs/emtqg/bin/python
#PYT_QG=/path/to/EMT_QG/bin/python
DM_MODEL=./saved_models/lew_10_lsw_0.6/seed_${SEED}/best.pt
QG_MODEL=./saved_models/unilm_16_0.00002_20/model.20.bin
BERT_PATH=/research/king3/ik_grp/yfgao/pretrain_models/bert-base-uncased.tar.gz

echo "=====$SEED====$SPLIT====="

${PYT_QG} inference_e2e.py \
--fin=data/sharc/json/sharc_${SPLIT}.json \
--dm=${DM_MODEL} \
--model_bert_base_path=${BERT_PATH} \
--model_recover_path=${QG_MODEL} \
--cache_path=/research/king3/ik_grp/yfgao/pretrain_models/ \
--batch_size=4


#--fin=data/sharc/json/sharc_dev.json
#--dm=out/train_decision_seed_11/best.pt
#--model_bert_base_path=/research/king3/ik_grp/yfgao/pretrain_models/bert-base-uncased.tar.gz
#--model_recover_path=/research/dept7/ik_grp/yfgao/open_retrieval_sharc_private/explicit_memory_tracker/saved_models/unilm_16_0.00002_20/model.20.bin
#--cache_path=/research/king3/ik_grp/yfgao/pretrain_models/
#--batch_size=4