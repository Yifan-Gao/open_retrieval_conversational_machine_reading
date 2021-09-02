#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
SEED=$2
SPLIT=$3
PYT_QG=/research/king3/yfgao/miniconda3/envs/emtqg/bin/python
#PYT_QG=/path/to/EMT_QG/bin/python
DM_MODEL=./out/train_decision_seed_${SEED}/best.pt
SPAN_MODEL=./out/train_span_seed_${SEED}/best.pt
QG_MODEL=/research/dept7/ik_grp/yfgao/open_retrieval_sharc_private/explicit_memory_tracker/saved_models/unilm_16_0.00002_20/model.20.bin
#BERT_PATH=/research/king3/ik_grp/yfgao/pretrain_models/bert-base-uncased.tar.gz

echo "=====$SEED====$SPLIT====="

${PYT_QG} e2e_inference.py \
--debug \
--fin=${SPLIT} \
--dout=out/e2e_seed${SEED}.json \
--model_decision=${DM_MODEL} \
--model_span=${SPAN_MODEL} \
--model_roberta_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base \
--model_recover_path=${QG_MODEL} \
--cache_path=/research/king3/ik_grp/yfgao/pretrain_models/ \
--batch_size=4 \
--pretrained_lm_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base

#--fin=dev
#--dout=out/debug.json
#--model_decision=out/train_decision_seed_11/best.pt
#--model_span=out/train_span_seed_11/best.pt
#--model_roberta_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base
#--model_recover_path=/research/dept7/ik_grp/yfgao/open_retrieval_sharc_private/explicit_memory_tracker/saved_models/unilm_16_0.00002_20/model.20.bin
#--cache_path=/research/king3/ik_grp/yfgao/pretrain_models/
#--batch_size=4
#--pretrained_lm_path=/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base
#--device=cpu