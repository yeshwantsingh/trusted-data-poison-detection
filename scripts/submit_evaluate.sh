#!/bin/bash
#PBS -N advrob_evaluate
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -j oe
#PBS -o evaluate.log
#PBS -V

cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2,2n -k3,3rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Picked GPU: $CUDA_VISIBLE_DEVICES on $(hostname)"

# Override via: qsub -v CHECKPOINT=./checkpoints/resnet18_pgd.pt scripts/submit_evaluate.sh
CHECKPOINT=${CHECKPOINT:-./checkpoints/resnet18_pgd.pt}

python evaluate.py --checkpoint "$CHECKPOINT"
