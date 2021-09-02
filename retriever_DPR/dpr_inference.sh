#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

PYT=/home/ubuntu/anaconda3/envs/sharc_dpr/bin/python

for split in train dev_seen dev_unseen test_seen test_unseen
do
  $PYT dense_retriever.py \
    --model_file=/home/ubuntu/data/MyFusionInDecoderOut/open_retrieval_sharc_private/ckpt/dpr_hf_bert_base.cp \
    --ctx_file=/home/ubuntu/data/MyFusionInDecoderOut/open_retrieval_sharc_private/data/sharc_raw/json/sharc_open_id2snippet.json \
    --qa_file=/home/ubuntu/data/MyFusionInDecoderOut/open_retrieval_sharc_private/data/sharc_raw/json/sharc_open_${split}.json \
    --encoded_ctx_file=/home/ubuntu/data/MyFusionInDecoderOut/open_retrieval_sharc_private/data/dpr_fb_ckpt/snippet_*.pkl \
    --out_file=/home/ubuntu/data/MyFusionInDecoderOut/open_retrieval_sharc_private/data/dpr_fb_ckpt/${split}.json \
    --n-docs=100
done