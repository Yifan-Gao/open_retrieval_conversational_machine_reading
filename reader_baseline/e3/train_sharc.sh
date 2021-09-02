export CUDA_VISIBLE_DEVICES=$1

SEED=$2
python train_sharc.py --seed=${SEED} --prefix="sharc_seed_${SEED}"