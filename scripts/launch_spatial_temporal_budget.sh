#!/usr/bin/env bash
# One resumable worker per video. Re-runs the worker until no pending condition is left (checkpoints skip finished ones).
cd "$(dirname "$0")/.."
PY=/home/vailab02/anaconda3/envs/vlm_experiment_env/bin/python
LOG=experiments/spatial_temporal_budget_40960/logs
i=0
for v in A C D E; do
  gpu=$((3+i)); i=$((i+1))
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_spatial_temporal_budget.py infer --videos $v --retry-failed >> $LOG/infer_$v.log 2>&1 && break
    done ) &
done
wait
