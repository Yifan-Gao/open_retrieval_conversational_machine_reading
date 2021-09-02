PYT=${OPENS}/bin/python

function preprocess() {
  MODEL=$1
  $PYT preprocess.py \
  --train_qa_file=../data/sharc_raw/json/sharc_open_train.jsonl \
  --validation_qa_seen_file=../data/sharc_raw/json/sharc_open_dev_seen.jsonl \
  --validation_qa_unseen_file=../data/sharc_raw/json/sharc_open_dev_unseen.jsonl \
  --test_qa_seen_file=../data/sharc_raw/json/sharc_open_test_seen.jsonl \
  --test_qa_unseen_file=../data/sharc_raw/json/sharc_open_test_unseen.jsonl \
  --train_retrieval_file=../data/tfidf/train.json \
  --validation_retrieval_seen_file=../data/tfidf/dev_seen.json \
  --validation_retrieval_unseen_file=../data/tfidf/dev_unseen.json \
  --test_retrieval_seen_file=../data/tfidf/test_seen.json \
  --test_retrieval_unseen_file=../data/tfidf/test_unseen.json \
  --snippet_file=../data/sharc_raw/json/sharc_open_id2snippet.json \
  --model_name_or_path=${MODEL} \
  --overwrite_cache
}

preprocess roberta-base