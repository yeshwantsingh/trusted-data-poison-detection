#!/bin/bash
#PBS -N advrob_certify
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -j oe
#PBS -o certify.log
#PBS -V

cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2,2n -k3,3rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Picked GPU: $CUDA_VISIBLE_DEVICES on $(hostname)"

# Override via: qsub -v CHECKPOINT=...,SIGMA=0.25 scripts/submit_certify.sh
CHECKPOINT=${CHECKPOINT:-./checkpoints/resnet18_standard_sigma0.25.pt}
SIGMA=${SIGMA:-0.25}

python certify.py --checkpoint "$CHECKPOINT" --sigma "$SIGMA" --num-images 500 --n 10000 \
  --out-csv ./results/certify_sigma${SIGMA}.csv
python plot_certified_accuracy.py --csv ./results/certify_sigma${SIGMA}.csv \
  --out ./results/certified_accuracy_sigma${SIGMA}.png
