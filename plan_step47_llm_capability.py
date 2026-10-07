"""
plan_step47_llm_capability.py
================================
Fills Appendix A.3 (app:llm_capability_extended): BW/LOG/MBW valid-plan success for several models,
using the real state-simulating validator, on the CORRECTED (v3) instances.

Each model must be served by Ollama at --host (one at a time -- load one, run it, unload, load the next).
Use `unload_others()`-style practice from earlier sessions: stop any other loaded model before starting
a new one so you don't run out of GPU memory.

USAGE (repeat --model for each entry in the table):
  python plan_step47_llm_capability.py --host http://localhost:11434 --model qwen2.5:72b --n 40
  python plan_step47_llm_capability.py --host http://localhost:11434 --model qwen2.5:14b --n 40
  python plan_step47_llm_capability.py --host http://localhost:11434 --model gemma2:9b    --n 40
  python plan_step47_llm_capability.py --host http://localhost:11434 --model mistral:7b   --n 40
Each run appends its row to results_planning_v3/llm_capability_table.json (resumable; reruns of the
same model overwrite only that model's row).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_validator import is_valid

TE = ["blocksworld", "logistics", "mystery_blocksworld"]
RESULTS = Path("results_planning_v3"); RESULTS.mkdir(exist_ok=True)
OUT_PATH = RESULTS / "llm_capability_table.json"

PROMPT_TMPL = (
    "You are an expert AI planning assistant. Your task is to solve a classical planning problem "
    "specified in PDDL (Planning Domain Definition Language).\n\n"
    "I will provide you with:\n1. A PDDL domain file describing the planning domain\n"
    "2. A PDDL problem file describing the specific problem to solve\n\n"
    "Your goal is to find a valid plan -- a sequence of actions that transforms the initial state "
    "into a state satisfying the goal.\n\nIMPORTANT INSTRUCTIONS:\n"
    "- Think step by step about the current state and what actions are applicable.\n"
    "- Each action in your plan must be applicable in the state resulting from all previous actions.\n"
    "- Output ONLY the final plan, one action per line, in the format:\n  (action-name param1 param2 ...)\n"
    "- Do NOT include any explanation after the plan.\n- If the problem is unsolvable, output exactly: NO_PLAN\n\n"
    "=== DOMAIN ===\n{domain}\n\n=== PROBLEM ===\n{problem}\n\n"
    "Now solve the problem. Think through the state changes carefully, then output the plan:"
)


def call_llm(host, model, prompt, num_predict=2048, timeout=300):
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
                          "options": {"num_predict": num_predict, "temperature": 0.0}}).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "")
    except Exception as e:
        return f"ERROR:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--data_dir", default="data/planning_v3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warm = call_llm(args.host, args.model, "Say OK.", num_predict=4)
    if warm.startswith("ERROR:"):
        print(f"ABORT: no usable LLM server at {args.host} for '{args.model}' ({warm}).")
        sys.exit(2)

    D = Path(args.data_dir)
    eps_all = json.loads((D / "episodes.json").read_text())
    valid = np.load(D / "y_valid.npy")
    tt = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])
    rng = np.random.default_rng(args.seed)

    row = {}
    n_err_total = 0
    for dom in TE:
        idx_all = np.where((tt == dom) & valid)[0]
        idx = rng.choice(idx_all, min(args.n, len(idx_all)), replace=False)
        succ = 0
        for j, i in enumerate(idx):
            t_call = time.time()
            ep = eps_all[i]
            resp = call_llm(args.host, args.model, PROMPT_TMPL.format(domain=ep["domain_pddl"], problem=ep["problem_pddl"]))
            ok, reason = is_valid(resp, ep)
            succ += int(ok)
            n_err_total += int(reason == "error")
            print(f"    [{j+1}/{len(idx)}] {dom} valid={ok} reason={reason} ({time.time()-t_call:.1f}s)", flush=True)  # slow vs stuck must be visible
        row[dom] = dict(n=len(idx), success_rate=succ / len(idx))
        print(f"  {args.model}  {dom:22s} N={len(idx)}  success={succ/len(idx):.1%}")

    if n_err_total / (len(TE) * args.n) > 0.2:
        print(f"ABORT: {n_err_total}/{len(TE)*args.n} calls errored -- server likely down partway through. "
              "Not saving this model's row. Fix the server and rerun.")
        sys.exit(3)

    out = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    out[args.model] = row
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT_PATH}  (row for '{args.model}')")
    print("\n=== Current table (all models run so far) ===")
    print(f"{'Model':<16}{'BW':>8}{'LOG':>8}{'MBW':>8}")
    for m, r in out.items():
        print(f"{m:<16}" + "".join(f"{r[d]['success_rate']*100:>7.1f}%" for d in TE))


if __name__ == "__main__":
    main()
