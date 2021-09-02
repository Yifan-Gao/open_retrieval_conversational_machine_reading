export CUDA_VISIBLE_DEVICES=$1

SEED=$2
SPLIT=$3

python train_sharc.py --test --seed=${SEED} --prefix="sharc_seed_${SEED}" --resume="./save/sharc_seed_${SEED}-entail/best.pt" --test_split=$SPLIT