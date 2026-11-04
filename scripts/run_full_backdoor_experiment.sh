#!/bin/bash
# Orchestrates the full 6-attack x 6-defense backdoor experiment on the PBS
# gpu queue: submits all 6 train_backdoor.py jobs, then all 36
# defend_backdoor.py jobs (each depending on its attack job via
# afterok), waits for the whole queue to drain (max_user_run=2, so this
# can take a while), then runs evaluate_backdoor.py over every resulting
# checkpoint and writes results/backdoor_experiment_summary.csv.
#
# Not submitted as a PBS job itself -- run directly (e.g. in the
# background) from the repo root:
#   nohup bash scripts/run_full_backdoor_experiment.sh > run_full_experiment.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

ATTACKS="badnets blend sig wanet dynamic trojan"
DEFENSES="abl aibd cbd dbd mcr nad"
ARCH=resnet18
EPOCHS=60
NGPUS=${NGPUS:-2}

mkdir -p results logs
declare -A ATTACK_JOBID
ALL_JOBS=()

echo "=== submitting $ARCH attack jobs ($EPOCHS epochs each, $NGPUS GPUs/job) ==="
for attack in $ATTACKS; do
  jobid=$(qsub -N "advrob_atk_${attack}" -j oe -o "logs/train_backdoor_${attack}.log" \
    -l "select=1:ncpus=8:ngpus=${NGPUS}" \
    -v "ATTACK=${attack},ARCH=${ARCH},EPOCHS=${EPOCHS},TARGET=0,POISON_RATE=0.1,NGPUS=${NGPUS}" \
    scripts/submit_train_backdoor.sh)
  jobid=$(echo "$jobid" | xargs)
  ATTACK_JOBID[$attack]=$jobid
  ALL_JOBS+=("$jobid")
  echo "  $attack -> $jobid"
done

echo "=== submitting defense jobs (each depends on its attack job, $NGPUS GPUs/job) ==="
for attack in $ATTACKS; do
  atk_job=${ATTACK_JOBID[$attack]}
  checkpoint="./checkpoints/${ARCH}_backdoor_${attack}.pt"
  for defense in $DEFENSES; do
    jobid=$(qsub -N "advrob_def_${attack}_${defense}" -j oe -o "logs/defend_backdoor_${attack}_${defense}.log" \
      -l "select=1:ncpus=8:ngpus=${NGPUS}" \
      -W "depend=afterok:${atk_job}" \
      -v "CHECKPOINT=${checkpoint},METHOD=${defense},NGPUS=${NGPUS}" \
      scripts/submit_defend_backdoor.sh)
    jobid=$(echo "$jobid" | xargs)
    ALL_JOBS+=("$jobid")
    echo "  ${attack}/${defense} -> $jobid (depends on $atk_job)"
  done
done

echo "=== ${#ALL_JOBS[@]} jobs submitted; waiting for the queue to drain (max_user_run limits concurrency) ==="

is_pending() {
  local jobid=$1
  qstat -u "$USER" 2>/dev/null | awk -v id="${jobid%%.*}" '$1 ~ ("^" id "\\.") {print $10}' | grep -qE '^(R|Q|H)$'
}

while true; do
  remaining=0
  for j in "${ALL_JOBS[@]}"; do
    if is_pending "$j"; then
      remaining=$((remaining + 1))
    fi
  done
  echo "$(date '+%Y-%m-%d %H:%M:%S'): $remaining/${#ALL_JOBS[@]} jobs still queued/running"
  if [ "$remaining" -eq 0 ]; then
    break
  fi
  sleep 120
done

echo "=== all jobs finished; running evaluation sweep ==="

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate torch-gpu

summary="results/backdoor_experiment_summary.csv"
echo "checkpoint,attack,defense,clean_acc,asr" > "$summary"

parse_and_append() {
  local ckpt=$1 attack=$2 defense=$3
  if [ ! -f "$ckpt" ]; then
    echo "$ckpt,$attack,$defense,MISSING,MISSING" >> "$summary"
    return
  fi
  local out
  out=$(python evaluate_backdoor.py --checkpoint "$ckpt" 2>/dev/null)
  local clean asr
  clean=$(echo "$out" | grep "clean accuracy" | sed -E 's/.*: *([0-9.]+).*/\1/')
  asr=$(echo "$out" | grep "attack success rate" | sed -E 's/.*: *([0-9.]+).*/\1/')
  echo "$ckpt,$attack,$defense,$clean,$asr" >> "$summary"
}

for attack in $ATTACKS; do
  parse_and_append "./checkpoints/${ARCH}_backdoor_${attack}.pt" "$attack" "none"
  for defense in $DEFENSES; do
    parse_and_append "./checkpoints/${ARCH}_backdoor_${attack}_defended_${defense}.pt" "$attack" "$defense"
  done
done

echo "=== DONE. Summary: ==="
cat "$summary"
