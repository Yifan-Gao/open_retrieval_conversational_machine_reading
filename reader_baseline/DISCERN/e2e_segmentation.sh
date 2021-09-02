#!/usr/bin/env bash

python segedu/e2e_segmentation.py ./data/sharc_raw/json/sharc_dev.json ./out/e2e_dev_parsed.json --vocab=./segedu/all_vocabulary.pickle --model=./segedu/trained_model.torchsave