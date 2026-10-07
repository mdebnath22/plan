#!/bin/bash
# Trains ARC per seed and EXPORTS per-instance predictions (routing's ARC input, ARC on the custom domains, the
# prediction-level seed-ensemble statistic).  0 main | 1 gate_blend.   SUBMIT:  cd /scratch/mroycho1/GURU && sbatch run_arc_export.sbatch
#SBATCH --job-name=arc_export
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/mroycho1/GURU/export_%A_%a.out
#SBATCH --error=/scratch/mroycho1/GURU/export_%A_%a.err
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
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  "$PY" plan_step51_arc_scores_export.py --seeds 0 1 2 42 123 --check_against results_planning_v3/table1_final_main.json
else
  "$PY" plan_step51_arc_scores_export.py --seeds 0 1 2 42 123 --gate blend --tag gate_blend --check_against results_planning_v3/table1_final_gate_blend.json
fi
echo "Finished: $(date)"
