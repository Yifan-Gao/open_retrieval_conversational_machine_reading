#!/usr/bin/env bash

PYT=${OPENS}/bin/python


function train() {
  export OMP_NUM_THREADS=1
  MODEL=roberta-base
  TOPK=100
  BZ=8
  MASPOT=$1
  L_ENTAIL=$2
  EPOCH=10
  LR=3e-5
  LTRANS=$3
  SEED=$4
  OUT="ablation_discern_like_${MODEL}_Lentail${L_ENTAIL}_Ltrans${LTRANS}_seed${SEED}"

  echo '=============================='
  echo "=====$OUT====="
  echo '=============================='

  $PYT -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port=$MASPOT run_ablation_discern_like.py \
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
  --tokenized_file=./data/roberta-base-tokenized.json \
  --tree_mapping_file=./data/roberta-base-tree-mapping.json \
  --top_k_snippets=$TOPK \
  --model_name_or_path=$MODEL \
  --output_dir=./out/${OUT} \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --evaluation_strategy=epoch \
  --per_device_train_batch_size=$BZ \
  --per_device_eval_batch_size=$BZ \
  --learning_rate=${LR} \
  --weight_decay=0.01 \
  --max_grad_norm=1.0 \
  --num_train_epochs=${EPOCH} \
  --warmup_steps=100 \
  --logging_steps=100 \
  --seed=${SEED} \
  --fp16 \
  --metric_for_best_model=combined \
  --lambda_entailment=${L_ENTAIL} \
  --load_best_model_at_end \
  --sequence_transformer_layer=${LTRANS}
}

masport=$1
ltrans=4


#for lent in 0.0 1.0 2.0 4.0 6.0 8.0 10.0
#do
#  for seed in 27 95 19 87 11
#  do
#    train $masport $lent $ltrans $seed
#  done
#done

lent=8.0
seed=$2
#for seed in 27 95 19 87 11
#do
train $masport $lent $ltrans $seed
#done

