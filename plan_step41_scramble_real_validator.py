"""
plan_step41_scramble_real_validator.py
=======================================
Reruns the progressive predicate-scrambling experiment (old plan_step31 EXP C) with the REAL validator.

WHAT WAS WRONG BEFORE
  plan_step31's is_valid() only checked that the response contained lines starting with "(" and that every action
  NAME existed in the scrambled domain. It never checked preconditions or the goal. The old 87.5% -> 52.5% curve
  therefore measured "did the model use real action names", not "is the plan correct". It also took the first N
  Blocksworld instances whether or not BFS had solved them.

WHAT THIS DOES
  * instances: the first N Blocksworld instances with valid==True (BFS-solved, so a plan exists) from data/planning_v3.
  * prompt wording, temperature 0 and the 512-token cap are IDENTICAL to the old run, so only the validator changes.
  * scrambling: same per-symbol substitution as before (level = probability each of the 9 BW symbols is renamed),
    seeded per (level, instance) so results do not depend on iteration order.
  * validation: a scrambled plan is mapped back to the original action names and run through plan_validator.is_valid()
    (parse -> arity -> preconditions by state simulation -> goal check). Renaming preserves the transition system,
    so original semantics are the right ground truth.
  * STRICT rule: an action name that is not in the (scrambled) domain the model was shown counts as invalid
    ("name_not_in_domain"), even if it is semantically fine -- the prompt says "Use ONLY these actions".
  * reports Wilson 95% CIs and a failure-reason breakdown, and saves every response for audit.

USAGE
  python plan_step41_scramble_real_validator.py --selftest          # offline, no LLM needed (run this first)
  python plan_step41_scramble_real_validator.py --host http://localhost:11434 --model qwen2.5:72b --n 40
"""
import argparse, json, re, sys, time, urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_validator import is_valid  # real state-simulating validator

ROOT = Path(".")
DATA = ROOT / "data" / "planning_v3"
RESULTS = ROOT / "results_planning_v3"; RESULTS.mkdir(exist_ok=True)

MBW_MAP = {"on": "pred7", "ontable": "pred3", "clear": "pred1", "handempty": "pred5", "holding": "pred2",
           "pick-up": "act3", "put-down": "act1", "stack": "act4", "unstack": "act2"}
ACTION_SYMS = ["pick-up", "put-down", "stack", "unstack"]
LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]


def scramble(domain_pddl, problem_pddl, level, instance_id):
    """Rename each of the 9 BW symbols with probability `level`. Returns (domain, problem, sub)."""
    rng = np.random.default_rng([int(round(level * 100)), int(instance_id)])
    sub = {old: new for old, new in MBW_MAP.items() if rng.random() < level}
    def apply(text):
        for old, new in sub.items():
            text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
        return text
    return apply(domain_pddl), apply(problem_pddl), sub


def build_prompt(c_dom, c_prob):
    acts = re.findall(r":action\s+(\S+)", c_dom)
    return (f"You are a PDDL expert. Solve this problem.\n\nDOMAIN:\n{c_dom}\n\nPROBLEM:\n{c_prob}\n\n"
            f"Use ONLY these actions: {', '.join(acts)}\n"
            "Output one action per line: (action args)\nIf unsolvable: NO_PLAN\n"), acts


def call_llm(host, model, prompt, num_predict=2048, timeout=300, retries=2):
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
                          "options": {"num_predict": num_predict, "temperature": 0.0}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read()).get("response", "")
        except Exception as e:
            err = f"ERROR:{e}"
            time.sleep(2)
    return err


def judge(response, ep, c_dom, sub):
    """(valid, reason) under the STRICT + REAL-validator rule described in the module docstring."""
    if not response or response.startswith("ERROR:"):
        return False, "error"
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    shown = {a.lower() for a in re.findall(r":action\s+(\S+)", c_dom)}
    # strict: any plan-looking line whose action name is not in the domain shown to the model
    for line in (l.strip() for l in txt.split("\n")):
        if line.startswith("("):
            toks = line.strip("()").split()
            if toks and toks[0].lower() not in shown and not re.search(r"\bNO[_\s-]PLAN\b", txt, re.I):
                return False, "name_not_in_domain"
    # map scrambled action names back to the original ones, then validate by state simulation
    back = {sub[a]: a for a in ACTION_SYMS if a in sub}
    for new, old in back.items():
        txt = re.sub(r"\b" + re.escape(new) + r"\b", old, txt)
    ok, reason = is_valid(txt, ep)
    return ok, reason


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def load_instances(n):
    eps = json.loads((DATA / "episodes.json").read_text())
    bw = [e for e in eps if e.get("task_type") == "blocksworld" and e.get("valid") and e.get("n_steps", 0) > 0]
    assert len(bw) >= n, f"only {len(bw)} BFS-solved Blocksworld instances available"
    return bw[:n]


def selftest():
    """Offline checks of the scramble -> map-back -> validate pipeline using BFS-optimal plans as an oracle."""
    eps = load_instances(40)
    bad = 0
    for level in LEVELS:
        ok_oracle = ok_orig_names = ok_reversed = 0; n_renamed_any = 0
        for ep in eps:
            c_dom, c_prob, sub = scramble(ep["domain_pddl"], ep["problem_pddl"], level, ep["instance_id"])
            plan = ep["plan"]
            scr = lambda a: re.sub(r"^\((\S+)", lambda m: "(" + sub.get(m.group(1), m.group(1)), a)
            oracle = "\n".join(scr(a) for a in plan)                    # correct plan, written in the scrambled vocabulary
            ok_oracle += judge(oracle, ep, c_dom, sub)[0]
            orig = "\n".join(plan)                                       # correct plan, but with the ORIGINAL action names
            used = {re.match(r"\((\S+)", a).group(1) for a in plan}
            uses_renamed = bool(used & set(sub))     # the strict rule only flags action names the plan actually USES
            n_renamed_any += uses_renamed
            v, why = judge(orig, ep, c_dom, sub)
            # expected: rejected as name_not_in_domain iff the plan uses a renamed action; otherwise accepted
            ok_orig_names += (v == (not uses_renamed)) and ((why == "name_not_in_domain") == uses_renamed)
            rev = "\n".join(scr(a) for a in reversed(plan)) if len(plan) > 1 else "(unstack b1 b2)"   # wrong order
            ok_reversed += (not judge(rev, ep, c_dom, sub)[0])
        n = len(eps)
        print(f"level {level:>4.0%}: oracle accepted {ok_oracle}/{n} | original-names handled correctly {ok_orig_names}/{n} "
              f"| wrong-order rejected {ok_reversed}/{n}")
        bad += (ok_oracle != n) + (ok_orig_names != n)
    print("\nSELFTEST", "PASSED" if bad == 0 else "FAILED")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--num_predict", type=int, default=2048,
                    help="BUG FIXED: this previously defaulted to 512, far too small for a 72B model's "
                         "chain-of-thought reasoning before it reaches the actual plan lines, causing "
                         "generation to be cut off before any plan was emitted -- confirmed as the cause "
                         "of a uniform 0% success rate across every corruption level in an earlier run.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    eps = load_instances(args.n)
    print(f"model={args.model}  N={len(eps)} BFS-solved Blocksworld instances per level  real validator + strict names\n")
    warm = call_llm(args.host, args.model, "Say OK.", num_predict=4)   # warm-up / load the model
    if warm.startswith("ERROR:"):
        print(f"ABORT: no usable LLM server at {args.host} for model '{args.model}' ({warm}).\n"
              "  * Run this on a GPU node (e.g. `interactive -G a100:1 -t 0-4:00`), not a CPU login/compute node.\n"
              "  * Start the SAME Ollama server you used for the earlier Qwen runs on that node, and check `curl "
              f"{args.host}/api/tags` lists the model.\n"
              "  No results file was written.")
        sys.exit(2)
    out = {"model": args.model, "n": len(eps), "num_predict": args.num_predict, "levels": {}}
    print(f"{'level':>7} {'valid-plan':>11} {'95% Wilson CI':>16}   failure reasons")
    for level in LEVELS:
        succ, reasons, recs = 0, Counter(), []
        for j, ep in enumerate(eps):
            t_call = time.time()
            c_dom, c_prob, sub = scramble(ep["domain_pddl"], ep["problem_pddl"], level, ep["instance_id"])
            prompt, _ = build_prompt(c_dom, c_prob)
            resp = call_llm(args.host, args.model, prompt, num_predict=args.num_predict)
            ok, why = judge(resp, ep, c_dom, sub)
            succ += int(ok); reasons[why] += 1
            recs.append({"instance_id": ep["instance_id"], "renamed": sorted(sub), "valid": bool(ok), "reason": why, "response": resp[:400]})
            # Per-instance progress: without this, a slow-but-working run and a hung run are
            # indistinguishable for the ~40 calls it takes to finish one level. If this print is
            # itself not advancing every few seconds to low minutes, the server is genuinely stuck,
            # not just slow -- check `curl $HOST/api/ps` and `nvidia-smi` in another shell.
            print(f"    [{j+1}/{len(eps)}] instance {ep['instance_id']}  valid={ok}  reason={why}  "
                  f"({time.time()-t_call:.1f}s)", flush=True)
        n_err = reasons.get("error", 0)
        if n_err / len(eps) > 0.2:
            print(f"ABORT at level {level:.0%}: {n_err}/{len(eps)} LLM calls errored (server down, timeout, or model unloaded). "
                  "Not recording this level; any levels already saved are valid. Fix the server and rerun.")
            sys.exit(3)
        lo, hi = wilson(succ, len(eps))
        print(f"{level:>7.0%} {succ/len(eps):>11.1%} {f'[{lo:.1%}, {hi:.1%}]':>16}   {dict(reasons)}", flush=True)
        out["levels"][str(level)] = {"valid_plan": succ / len(eps), "wilson95": [lo, hi], "reasons": dict(reasons), "records": recs}
        (RESULTS / f"scramble_real_validator_{args.model.replace(':','_')}.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {RESULTS}/scramble_real_validator_{args.model.replace(':','_')}.json")


if __name__ == "__main__":
    main()
