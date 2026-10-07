#!/bin/bash
# The five ARC-ablation suites of plan_step44 as a PARALLEL job array (one suite per task); each writes
# results_planning_v3/arc_ablations_<suite>.json.
#   0 attn (20 trainings) | 1 repfamily (no training) | 2 resid (50 trainings) | 3 loo (35 trainings) | 4 within (5 trainings)
# Rough guess (NOT measured): a few minutes per CPU training, so the longest suite should take hours, not days. If a task
# hits the limit, resubmit just it with fewer seeds:  sbatch --array=2 --export=ALL,SEEDS="0 1 2" run_arc_ablations.sbatch
# SUBMIT:  cd /scratch/mroycho1/GURU && sbatch run_arc_ablations.sbatch
#SBATCH --job-name=arc_ablate
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/mroycho1/GURU/ablate_%A_%a.out
#SBATCH --error=/scratch/mroycho1/GURU/ablate_%A_%a.err
set -eo pipefail
# ---- environment: call the conda env python by ABSOLUTE path (activation proved unreliable inside batch jobs:
# the Table-1 job ran the system python and died with "No module named torch") ----
export HOME=/home/mroycho1
PY="${ARC_PY:-/home/mroycho1/.conda/envs/mdebnath/bin/python}"
if [ ! -x "$PY" ]; then echo "ERROR: env python not found at $PY (set ARC_PY=... to override)"; exit 2; fi
export PATH="$(dirname "$PY"):$PATH"; export CONDA_PREFIX="$(dirname "$(dirname "$PY")")"
"$PY" -c "import sys, torch, numpy, scipy, sklearn; print(\"python\", sys.version.split()[0], \"| torch\", torch.__version__, \"| cuda\", torch.cuda.is_available())" \
  || { echo "ERROR: required packages (torch/numpy/scipy/sklearn) missing for $PY"; exit 2; }
cd /scratch/mroycho1/GURU
SEEDS="${SEEDS:-0 1 2 42 123}"
SUITES=(attn repfamily resid loo within)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
echo "Task ${SLURM_ARRAY_TASK_ID}: suite=${SUITE} seeds=${SEEDS}  started $(date) on $(hostname)"
"$PY" plan_step44_arc_ablations.py --suite ${SUITE} --seeds ${SEEDS} --n_episodes 5000
echo "Task ${SLURM_ARRAY_TASK_ID} finished $(date)"
