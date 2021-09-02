export CUDA_VISIBLE_DEVICES=$1

SEED=$2
SPLIT=$3

echo "=====seed_$SEED=====split_$SPLIT====="

python inference.py --fin "sharc/json/sharc_${SPLIT}.json" \
--dout "out/seed_${SEED}_${SPLIT}" \
--force \
--retrieval "save/sharc_seed_${SEED}-entail/best.pt" \
--editor "editor_save/editor_seed_${SEED}-double/best.pt" --verify



