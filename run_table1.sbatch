#!/bin/bash
# Table 1, three PRE-DECLARED ARC variants as a job array:  0 main | 1 gate_blend | 2 original (old per-episode pipeline)
# Writes results_planning_v3/table1_final_<tag>.json.   SUBMIT:  cd /scratch/mroycho1/GURU && sbatch run_table1.sbatch
#SBATCH --job-name=arc_table1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/mroycho1/GURU/table1_%A_%a.out
#SBATCH --error=/scratch/mroycho1/GURU/table1_%A_%a.err
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
case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=main;       FLAGS="--norm pooled  --y_norm domain  --gate none"  ;;
  1) TAG=gate_blend; FLAGS="--norm pooled  --y_norm domain  --gate blend" ;;
  2) TAG=original;   FLAGS="--norm episode --y_norm episode --gate none"  ;;
  *) echo "unexpected array index ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} tag=${TAG} flags=${FLAGS}  started $(date) on $(hostname)"
"$PY" plan_table1.py --n_episodes 5000 --seeds 0 1 2 42 123 --val_every 200 --patience 10 ${FLAGS} --tag ${TAG}
echo "Finished: $(date)"
