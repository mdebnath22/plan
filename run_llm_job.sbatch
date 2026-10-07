#!/bin/bash
# ONE LLM experiment per job, on a FULL GPU (the 72B model needs ~46 GiB; a MIG 2g.20gb slice fails with
# "unable to allocate CUDA0 buffer"). Choose the experiment at submission time:
#   sbatch -J scramble   --export=ALL,EXPERIMENT=scramble   run_llm_job.sbatch
#   sbatch -J cap72      --export=ALL,EXPERIMENT=capability run_llm_job.sbatch      # 72B row of the capability table
#   sbatch -J baseline   --export=ALL,EXPERIMENT=baseline   run_llm_job.sbatch
#   sbatch -J routing    --export=ALL,EXPERIMENT=routing    run_llm_job.sbatch      # resumable: reruns reuse cached LLM outcomes
#   sbatch -J capsmall --gres=gpu:1 --export=ALL,EXPERIMENT=small run_llm_job.sbatch   # 14b/9b/7b rows: fit a 20 GiB slice, so any GPU is fine
# htc is limited to 240 min per job (seen earlier: "does not permit jobs longer than 240 minutes"). Scripts that are not
# resumable (scramble, baseline) must finish inside one job; routing and the capability rows are resumable.
# There is no --constraint line on purpose: --constraint=a100_80gb gave "Invalid feature specification" on this cluster.
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/mroycho1/GURU/llm_%x_%j.out
#SBATCH --error=/scratch/mroycho1/GURU/llm_%x_%j.err
set -o pipefail
# ---- environment: call the conda env python by ABSOLUTE path (activation proved unreliable inside batch jobs:
# the Table-1 job ran the system python and died with "No module named torch") ----
export HOME=/home/mroycho1
PY="${ARC_PY:-/home/mroycho1/.conda/envs/mdebnath/bin/python}"
if [ ! -x "$PY" ]; then echo "ERROR: env python not found at $PY (set ARC_PY=... to override)"; exit 2; fi
export PATH="$(dirname "$PY"):$PATH"; export CONDA_PREFIX="$(dirname "$(dirname "$PY")")"
"$PY" -c "import sys, torch, numpy, scipy, sklearn; print(\"python\", sys.version.split()[0], \"| torch\", torch.__version__, \"| cuda\", torch.cuda.is_available())" \
  || { echo "ERROR: required packages (torch/numpy/scipy/sklearn) missing for $PY"; exit 2; }
cd /scratch/mroycho1/GURU
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=scramble|capability|baseline|routing|small via --export}"
case "$EXPERIMENT" in small) MODEL=qwen2.5:14b ;; *) MODEL=qwen2.5:72b ;; esac
echo "=== $EXPERIMENT  model=$MODEL  node=$(hostname)  started $(date) ==="
nvidia-smi -L || true
export START_OLLAMA_NO_KILL=1
source /scratch/mroycho1/GURU/start_ollama.sh "$MODEL" preload
RC=$?
if [ $RC -ne 0 ]; then echo "ERROR: Ollama did not become ready (rc=$RC); see messages above."; exit 3; fi
case "$EXPERIMENT" in
  scramble)   "$PY" plan_step41_scramble_real_validator.py --host "$HOST" --model qwen2.5:72b --n 40 ;;
  capability) "$PY" plan_step47_llm_capability.py --host "$HOST" --model qwen2.5:72b --n 40 ;;
  baseline)   "$PY" plan_step48_llm_difficulty_baseline.py --host "$HOST" --model qwen2.5:72b --ctx 24576 ;;
  routing)    if [ -f results_planning_v3/arc_scores_for_routing.json ]; then ARC="--arc_scores results_planning_v3/arc_scores_for_routing.json"; else ARC=""; echo "NOTE: no ARC scores file yet; ARC rows skipped"; fi
              "$PY" plan_step46_routing.py --host "$HOST" --model qwen2.5:72b --n 100 $ARC ;;
  small)      for m in qwen2.5:14b gemma2:9b mistral:7b; do "$PY" plan_step47_llm_capability.py --host "$HOST" --model $m --n 40; done ;;
  *) echo "unknown EXPERIMENT=$EXPERIMENT"; exit 1 ;;
esac
RC=$?
pkill -x ollama 2>/dev/null
echo "=== $EXPERIMENT finished rc=$RC at $(date) ==="
exit $RC
