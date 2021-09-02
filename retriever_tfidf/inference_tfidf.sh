#!/usr/bin/env bash

PYT=/research/dept7/ik_grp/yfgao/miniconda3/envs/tfidf/bin/python

for split in train dev_seen dev_unseen test_seen test_unseen dev test
do
  echo "=-=-=-=-=-=-=${split}-=-=-=-=-=-=-"
  $PYT inference_tfidf.py \
    --qa_file=../data/sharc_raw/json/sharc_open_${split}.json \
    --db_path=../data/sharc_raw/json/sharc_open_id2snippet.json \
    --out_file=../data/tfidf/${split}.json \
    --tfidf_path=../data/tfidf/snippet-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
done