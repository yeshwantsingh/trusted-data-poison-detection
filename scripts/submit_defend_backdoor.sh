#!/bin/bash
#PBS -N advrob_defend_backdoor
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -j oe
#PBS -o defend_backdoor.log
#PBS -V

cd $PBS_O_WORKDIR
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu

# NGPUS should match the ngpus in #PBS -l select (or a -l override passed to
# qsub). Note MCR never uses more than 1 GPU (its weight-space curve fit
# needs the plain, unwrapped model -- see src/models.py::to_device), so the
# second GPU sits idle for METHOD=mcr specifically; not worth a special case
# for this experiment's scale.
NGPUS=${NGPUS:-2}
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2,2n -k3,3rn | head -"$NGPUS" | cut -d',' -f1 | tr -d ' ' | paste -sd,)
echo "Picked GPUs: $CUDA_VISIBLE_DEVICES on $(hostname)"

# Override via: qsub -l select=1:ncpus=8:ngpus=2 -v CHECKPOINT=...,METHOD=nad,NGPUS=2 scripts/submit_defend_backdoor.sh
# GAMMA/OVERSAMPLE/OUT_DIR are only used by METHOD=hybrid; left unset, they
# fall back to defend_backdoor.py's own argparse defaults.
CHECKPOINT=${CHECKPOINT:-./checkpoints/resnet18_backdoor_badnets.pt}
METHOD=${METHOD:-abl}
OUT_DIR=${OUT_DIR:-./checkpoints}
EXTRA_ARGS=""
if [ -n "${GAMMA:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --gamma $GAMMA"; fi
if [ -n "${OVERSAMPLE:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --oversample-fraction $OVERSAMPLE"; fi
if [ -n "${ISOLATION:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --isolation-method $ISOLATION"; fi
if [ -n "${FINAL_Q:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --final-q $FINAL_Q"; fi
if [ -n "${CAFD_PERCENTILE:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --cafd-threshold-percentile $CAFD_PERCENTILE"; fi
if [ -n "${USE_NC:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --use-neural-cleanse"; fi
if [ -n "${DELTA:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --delta $DELTA"; fi
TAG=$(basename "$CHECKPOINT" .pt)
DEFENDED_CHECKPOINT="$OUT_DIR/${TAG}_defended_${METHOD}.pt"

python defend_backdoor.py --checkpoint "$CHECKPOINT" --method "$METHOD" \
  --data-dir ./data --out-dir "$OUT_DIR" $EXTRA_ARGS

echo "--- before defense ---"
python evaluate_backdoor.py --checkpoint "$CHECKPOINT"
echo "--- after defense ($METHOD) ---"
python evaluate_backdoor.py --checkpoint "$DEFENDED_CHECKPOINT"
