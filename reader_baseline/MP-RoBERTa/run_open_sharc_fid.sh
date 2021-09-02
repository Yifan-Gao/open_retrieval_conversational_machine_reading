PYT=/home/ubuntu/anaconda3/envs/opensharc/bin/python

function train() {
  export OMP_NUM_THREADS=1
  MODEL=$1
  TOPK=$2
  BZ=8
  $PYT -m torch.distributed.launch \
    --nproc_per_node 8 run_open_sharc_fid.py \
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
    --top_k_snippets=$TOPK \
    --model_name_or_path=$MODEL \
    --output_dir=./out/"${MODEL}-top${TOPK}" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluation_strategy=epoch \
    --per_device_train_batch_size=$BZ \
    --per_device_eval_batch_size=$BZ \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --num_train_epochs=5 \
    --warmup_steps=100 \
    --logging_steps=100 \
    --seed=27 \
    --fp16 \
    --metric_for_best_model=combined
}

MODEL=roberta-base
for TOPK in 1 2 4 5
do
  echo "==========${MODEL}==${TOPK}============"
  train $MODEL $TOPK
done



#--train_qa_file=../data/sharc_raw/json/sharc_open_train.jsonl
#--validation_qa_seen_file=../data/sharc_raw/json/sharc_open_dev_seen.jsonl
#--validation_qa_unseen_file=../data/sharc_raw/json/sharc_open_dev_unseen.jsonl
#--test_qa_seen_file=../data/sharc_raw/json/sharc_open_test_seen.jsonl
#--test_qa_unseen_file=../data/sharc_raw/json/sharc_open_test_unseen.jsonl
#--train_retrieval_file=../data/tfidf/train.json
#--validation_retrieval_seen_file=../data/tfidf/dev_seen.json
#--validation_retrieval_unseen_file=../data/tfidf/dev_unseen.json
#--test_retrieval_seen_file=../data/tfidf/test_seen.json
#--test_retrieval_unseen_file=../data/tfidf/test_unseen.json
#--snippet_file=../data/sharc_raw/json/sharc_open_id2snippet.json
#--top_k_snippets=2
#--model_name_or_path=facebook/bart-base
#--output_dir=./out/"facebook/bart-base-top1"
#--overwrite_output_dir
#--do_train
#--do_eval
#--evaluation_strategy=epoch
#--per_device_train_batch_size=8
#--per_device_eval_batch_size=8
#--learning_rate=2e-5
#--weight_decay=0.01
#--max_grad_norm=1.0
#--num_train_epochs=5
#--warmup_steps=100
#--logging_steps=100
#--seed=27
#--fp16
#--metric_for_best_model=combined

