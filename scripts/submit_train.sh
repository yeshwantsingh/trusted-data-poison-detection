#!/bin/bash
#PBS -N advrob_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -j oe
#PBS -o train.log
#PBS -V

cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu

# NGPUS should match the ngpus in #PBS -l select (or a -l override passed to
# qsub). train.py uses all visible GPUs via nn.DataParallel.
NGPUS=${NGPUS:-2}
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2,2n -k3,3rn | head -"$NGPUS" | cut -d',' -f1 | tr -d ' ' | paste -sd,)
echo "Picked GPUs: $CUDA_VISIBLE_DEVICES on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Override via: qsub -l select=1:ncpus=8:ngpus=2 -v METHOD=trades,ARCH=resnet18,EPOCHS=60,SIGMA=0.25,NGPUS=2 scripts/submit_train.sh
METHOD=${METHOD:-pgd}
ARCH=${ARCH:-resnet18}
EPOCHS=${EPOCHS:-60}
SIGMA_ARG=""
if [ -n "$SIGMA" ]; then
  SIGMA_ARG="--sigma $SIGMA"
fi

python train.py --method "$METHOD" --arch "$ARCH" --epochs "$EPOCHS" \
  --data-dir ./data --out-dir ./checkpoints $SIGMA_ARG
