export CUDA_VISIBLE_DEVICES=$1

SEED=$2
python train_editor.py --seed=${SEED} --prefix="editor_seed_${SEED}"