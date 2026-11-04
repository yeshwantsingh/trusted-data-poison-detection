#!/bin/bash
#PBS -N advrob_train_backdoor
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -j oe
#PBS -o train_backdoor.log
#PBS -V

cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu

# NGPUS should match the ngpus in #PBS -l select (or a -l override passed to
# qsub) -- it only controls how many of the assigned GPUs get picked below.
# train_backdoor.py uses all visible GPUs via nn.DataParallel.
NGPUS=${NGPUS:-2}
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2,2n -k3,3rn | head -"$NGPUS" | cut -d',' -f1 | tr -d ' ' | paste -sd,)
echo "Picked GPUs: $CUDA_VISIBLE_DEVICES on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Override via: qsub -l select=1:ncpus=8:ngpus=2 \
#   -v ATTACK=wanet,ARCH=resnet18,EPOCHS=60,TARGET=0,POISON_RATE=0.1,NGPUS=2 \
#   scripts/submit_train_backdoor.sh
ATTACK=${ATTACK:-badnets}
ARCH=${ARCH:-resnet18}
EPOCHS=${EPOCHS:-60}
TARGET=${TARGET:-0}
POISON_RATE=${POISON_RATE:-0.1}

python train_backdoor.py --attack "$ATTACK" --arch "$ARCH" --epochs "$EPOCHS" \
  --target-label "$TARGET" --poison-rate "$POISON_RATE" \
  --data-dir ./data --out-dir ./checkpoints
