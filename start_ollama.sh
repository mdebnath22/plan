# start_ollama.sh -- start an Ollama server on a GPU node and print the exact --host to use.
#
# USAGE (must be SOURCED, not executed, so `module` and the exports persist in your shell):
#     source start_ollama.sh                    # checks qwen2.5:72b, preloads it
#     source start_ollama.sh qwen2.5:14b        # check/preload a different model
#     source start_ollama.sh qwen2.5:72b nopreload
#
# What each step fixes (none of it survives into a new shell):
#   1. `module load ollama/0.12.10`   without it `ollama` is not on PATH -> "Exit 127"
#   2. OLLAMA_MODELS=/scratch/mroycho1/.ollama/models   where the models were actually pulled
#   3. OLLAMA_FLASH_ATTENTION / OLLAMA_CONTEXT_LENGTH     a 72B model is far slower without them
#   4. OLLAMA_LOAD_TIMEOUT=30m        default 5m; a cold 44 GiB read from /scratch took 404 s (117 MB/s)
#   5. the host:port the module chose (not always 11434; 11435 has appeared)
#   6. GPU-SIZE GUARD: a 72B Q4 model needs ~46 GiB of GPU memory. If your allocation is a MIG slice
#      (you were given `MIG 2g.20gb` = ~20 GiB), loading fails after ~15 minutes with
#      "unable to allocate CUDA0 buffer". This script checks what CUDA actually gives THIS job and stops early.
# Deliberately no `set -e` (it is sourced into your interactive shell).
# Env knobs: START_OLLAMA_GPU_GIB=<n> overrides the detected GPU size (testing);
#            START_OLLAMA_NO_KILL=1 skips killing other `ollama serve` processes (use inside batch jobs).

_MODEL="${1:-qwen2.5:72b}"
_PRELOAD="${2:-preload}"

module load ollama/0.12.10 || { echo "[start_ollama] 'module load ollama/0.12.10' failed. Try: module avail ollama"; return 1; }
if ! command -v ollama >/dev/null 2>&1; then
  echo "[start_ollama] 'ollama' is still not on PATH after loading the module. Are you on a GPU node?"
  return 1
fi

# ---- GPU-size guard: ask CUDA what THIS job can use (nvidia-smi lists every GPU on the node, not just yours) ----
_need=0
case "$_MODEL" in
  *72b*|*70b*) _need=46 ;;
  *8x22b*)     _need=80 ;;
  *32b*|*34b*) _need=20 ;;
  *14b*|*16b*) _need=10 ;;
esac
_gpu=""
if [ -n "${START_OLLAMA_GPU_GIB:-}" ]; then
  _gpu="${START_OLLAMA_GPU_GIB}|override"
elif command -v python3 >/dev/null 2>&1; then
  _gpu=$(python3 - 2>/dev/null <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        print("NOCUDA")
    else:
        d = torch.cuda.get_device_properties(0)
        print("%.1f|%s" % (d.total_memory / 2**30, d.name))
except Exception:
    print("UNKNOWN")
PY
)
fi
case "$_gpu" in
  NOCUDA)  echo "[start_ollama] WARNING: this Python's CUDA sees no GPU (could be a CPU-only torch build, or a non-GPU allocation). Cannot verify GPU size." ;;
  UNKNOWN|"") echo "[start_ollama] WARNING: could not determine GPU memory (no working torch). Check manually: the allocation must hold ~${_need} GiB for ${_MODEL}." ;;
  *)
    _gib="${_gpu%%|*}"; _gname="${_gpu#*|}"
    echo "[start_ollama] GPU this job can use: ${_gname}, ${_gib} GiB   (model ${_MODEL} needs ~${_need} GiB)"
    if python3 -c "import sys; sys.exit(0 if float('${_gib}') >= ${_need} else 1)" 2>/dev/null; then :; else
      echo "[start_ollama] STOP: this allocation is too small for ${_MODEL}. A ~20 GiB device is a MIG slice, not a full A100."
      echo "    The 'sg*' nodes are MIG-partitioned; full GPUs were on 'scg*' nodes (a100 x2/x4, h100 x4/x8 in earlier sinfo output)."
      echo "    Ask for a specific full-GPU type, e.g.:   interactive -G a100:1 -t 0-4:00      (htc caps jobs at 240 min)"
      echo "    or in a batch script:                     #SBATCH --gres=gpu:a100:1          (do NOT use --constraint=a100_80gb: 'Invalid feature')"
      echo "    then re-run this script; it re-checks. Small models (qwen2.5:14b, gemma2:9b, mistral:7b) DO fit this slice."
      return 1
    fi ;;
esac

export HOME=/home/mroycho1
export OLLAMA_MODELS=/scratch/mroycho1/.ollama/models
export OLLAMA_FLASH_ATTENTION=true
# 24576, not 16384: plan_step48_llm_difficulty_baseline.py requests num_ctx=24576 per call (its 20-example
# support block makes prompts far longer than the other scripts' single-instance prompts). If the server's
# default context differs from what a later call requests, Ollama RELOADS the model with the new KV-cache
# size before answering -- a second, hidden cold load that can itself exceed a calling script's request
# timeout (confirmed: this caused "ERROR:timed out" on plan_step48 after a successful preload at the old
# default). Keeping this value >= every script's requested num_ctx means the preload below is the only load.
# HARD-SET, not a soft "${VAR:-default}": a long-lived interactive shell that previously sourced an older
# version of this script already has OLLAMA_CONTEXT_LENGTH exported at the OLD value, and a soft default
# only applies when the variable is unset -- it would silently keep using the stale value forever in that
# shell (confirmed: this is exactly what happened, re-triggering the same reload-timeout bug after the fix).
export OLLAMA_CONTEXT_LENGTH=24576
export OLLAMA_LOAD_TIMEOUT=30m

if [ -z "${START_OLLAMA_NO_KILL:-}" ]; then pkill -f "ollama serve" 2>/dev/null; sleep 3; fi
nohup ollama serve > /tmp/ollama_new.log 2>&1 &
_PID=$!
echo "[start_ollama] started 'ollama serve' (pid $_PID); module-chosen OLLAMA_HOST='${OLLAMA_HOST:-<unset>}'"

_cands=()
if [ -n "${OLLAMA_HOST:-}" ]; then
  _h="$OLLAMA_HOST"
  case "$_h" in http://*|https://*) ;; *) _h="http://$_h" ;; esac
  _h="${_h/0.0.0.0/$(hostname)}"
  _cands+=("$_h")
fi
_cands+=("http://$(hostname):11434" "http://localhost:11434")

HOST=""
for _i in $(seq 1 90); do
  for _c in "${_cands[@]}"; do
    if curl -sf --max-time 3 "$_c/api/tags" >/dev/null 2>&1; then HOST="$_c"; break 2; fi
  done
  if ! kill -0 "$_PID" 2>/dev/null; then
    echo "[start_ollama] the ollama process exited early. Last log lines:"; tail -n 20 /tmp/ollama_new.log; return 1
  fi
  sleep 2
done
if [ -z "$HOST" ]; then
  echo "[start_ollama] server did not answer on any of: ${_cands[*]}"; tail -n 20 /tmp/ollama_new.log; return 1
fi
export HOST
echo "[start_ollama] server is UP at $HOST"

python3 - "$HOST" "$_MODEL" <<'PY'
import json, sys, urllib.request
host, want = sys.argv[1], sys.argv[2]
names = [m["name"] for m in json.load(urllib.request.urlopen(host + "/api/tags", timeout=10))["models"]]
print("[start_ollama] models available:", ", ".join(names))
print("[start_ollama] requested model", repr(want), "->", "FOUND" if want in names else "NOT FOUND (check OLLAMA_MODELS path)")
PY

if [ "$_PRELOAD" = "preload" ]; then
  echo "[start_ollama] preloading $_MODEL at num_ctx=${OLLAMA_CONTEXT_LENGTH} (blocks until loaded; a cold 72B"
  echo "  load has taken anywhere from ~7 to ~22 minutes across nodes seen so far -- this is the ONLY load,"
  echo "  not a quick check, because every later call below is also pinned to this same context size)..."
  _t0=$(date +%s)
  _resp=$(curl -s --max-time 2400 "$HOST/api/generate" \
    -d "{\"model\":\"$_MODEL\",\"prompt\":\"Say OK.\",\"stream\":false,\"keep_alive\":\"60m\",\"options\":{\"num_predict\":4,\"num_ctx\":${OLLAMA_CONTEXT_LENGTH}}}")
  _dt=$(( $(date +%s) - _t0 ))
  echo "$_resp" | python3 -c "
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print('[start_ollama] PRELOAD FAILED: non-JSON reply:', repr(raw[:200])); sys.exit(3)
if 'error' in d:
    print('[start_ollama] PRELOAD FAILED:', d['error']); sys.exit(3)
print('[start_ollama] preload OK, model replied:', repr(d.get('response')))
"
  _rc=$?
  if [ "$_rc" -ne 0 ]; then
    echo "[start_ollama] preload took ${_dt}s and FAILED. Diagnose with:"
    echo "    tail -n 80 /tmp/ollama_new.log     # look for 'offloaded', 'unable to allocate', runner errors"
    echo "    nvidia-smi -L                       # note: lists the node's GPUs/MIG slices, not just yours"
    echo "    ps -u \"\$(id -un)\" -o pid,etime,cmd | grep -i '[o]llama'    # leftover runners (\$USER is not a real account here)"
    unset _MODEL _PRELOAD _PID _cands _h _c _i _t0 _dt _resp _rc _need _gpu _gib _gname
    return 1
  fi
  echo "[start_ollama] model loaded in ${_dt}s"
fi
echo "[start_ollama] READY. Use:  --host $HOST   (also exported as \$HOST)"
unset _MODEL _PRELOAD _PID _cands _h _c _i _t0 _dt _resp _rc _need _gpu _gib _gname
