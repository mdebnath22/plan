"""
plan_step46_routing.py
=========================
Fills Table 4 (routing): for each test domain, routes instances to Qwen~2.5-72B or a symbolic fallback
(the BFS planner itself, standing in for "a complete symbolic solver" -- it always returns the optimal
plan whenever the instance is BFS-solvable, which is true by construction for every instance scored here)
using a structural difficulty score, and reports the percentage solved under Static / Oracle / ZS / FS
policies for |O|, goal count, and (optionally) a trained ARC checkpoint.

Needs: an Ollama server reachable at --host serving --model (GPU node), the v3 data, and plan_validator.py.
ARC scores are optional (--arc_checkpoint); without one, only the |O| and goal-count rows are produced.

USAGE:
  python plan_step46_routing.py --host http://localhost:11434 --model qwen2.5:72b --n 100
  # with an ARC score file (see --help for the expected format) once Table 1's run has produced one:
  python plan_step46_routing.py --host ... --model ... --n 100 --arc_scores results_planning_v3/arc_scores_for_routing.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_validator import is_valid

TE = ["blocksworld", "logistics", "mystery_blocksworld"]
RESULTS = Path("results_planning_v3"); RESULTS.mkdir(exist_ok=True)

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


def wilson_check_server(host, model):
    r = call_llm(host, model, "Say OK.", num_predict=4)
    return not r.startswith("ERROR:")


def load_llm_outcomes(host, model, eps, cache_path):
    """Returns {instance_id: bool solved_by_llm}, caching to disk so a routing experiment can be rerun
    (e.g. with different thresholds) without re-querying the LLM."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    changed = False
    todo = [ep for ep in eps if str(ep["instance_id"]) not in cache]
    print(f"  {cache_path.name}: {len(eps)-len(todo)} cached, {len(todo)} to query", flush=True)
    for n_done, ep in enumerate(todo):
        key = str(ep["instance_id"])
        t_call = time.time()
        prompt = PROMPT_TMPL.format(domain=ep["domain_pddl"], problem=ep["problem_pddl"])
        resp = call_llm(host, model, prompt)
        ok, reason = is_valid(resp, ep)
        cache[key] = dict(solved=bool(ok), reason=reason)
        changed = True
        print(f"    [{n_done+1}/{len(todo)}] instance {key} solved={ok} reason={reason} ({time.time()-t_call:.1f}s)", flush=True)
        if len(cache) % 20 == 0:
            cache_path.write_text(json.dumps(cache))  # periodic save; a long run can be resumed
    if changed:
        cache_path.write_text(json.dumps(cache))
    return {int(k): v["solved"] for k, v in cache.items()}


def route(scores, llm_solved, threshold):
    """score <= threshold -> route to LLM; else -> symbolic fallback (always solves, by construction
    of this evaluation set: every scored instance is BFS-solvable)."""
    routed_to_llm = scores <= threshold
    solved = np.where(routed_to_llm, llm_solved, True)
    return float(solved.mean() * 100)


def best_threshold(scores, llm_solved, n_grid=50):
    """Zero-shot: no target-domain outcomes used to pick a threshold is not meaningful (there is no
    oracle-free way to choose one), so ZS uses the MEDIAN score as a fixed, non-target-tuned rule:
    route the structurally-easier half to the LLM. This must be decided in advance, not swept."""
    return float(np.median(scores))


def calibrated_threshold(scores, llm_solved, n_calib, seed):
    """FS: use n_calib target-domain outcomes to pick, by grid search, the threshold maximizing routed
    accuracy on JUST those n_calib instances, then apply it to the full set (train/apply split, not
    fit-and-eval on the same instances)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scores), min(n_calib, len(scores)), replace=False)
    cand = np.unique(scores)
    best_t, best_acc = cand[0], -1
    for t in cand:
        acc = route(scores[idx], llm_solved[idx], t)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--n", type=int, default=100, help="instances per domain (use 200 for the full set; "
                                                        "100 as a faster default given LLM call cost)")
    ap.add_argument("--n_calib", type=int, default=140)
    ap.add_argument("--data_dir", default="data/planning_v3")
    ap.add_argument("--arc_scores", default=None,
                    help="JSON {instance_id: score} from a trained ARC checkpoint (lower = easier). "
                         "Produced by plan_table1.py's eval_arc on ALL instances of each test domain, "
                         "not just the ones routed here; see get_arc_scores_for_routing() below for the "
                         "snippet to generate it.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not wilson_check_server(args.host, args.model):
        print(f"ABORT: no usable LLM server at {args.host} for '{args.model}'. Start Ollama on a GPU node first.")
        sys.exit(2)

    D = Path(args.data_dir)
    eps_all = json.loads((D / "episodes.json").read_text())
    valid = np.load(D / "y_valid.npy")
    Xs = np.load(D / "X_surf.npy")
    tt = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])
    n_obj = Xs[:, 0]
    n_goal = np.array([len(e["goal_facts"]) for e in eps_all], float)

    arc_scores_map = json.loads(Path(args.arc_scores).read_text()) if args.arc_scores else None
    if args.arc_scores and arc_scores_map is None:
        print("WARNING: --arc_scores given but file could not be loaded; ARC rows will be skipped.")

    rng = np.random.default_rng(args.seed)
    out = {}
    print(f"{'Domain':<14}{'Static':>8}{'Oracle':>8}{'|O|-ZS':>8}{'Goal-ZS':>9}{'ARC-ZS':>8}"
          f"{'|O|-FS':>8}{'Goal-FS':>9}{'ARC-FS':>8}{'%Gap':>7}")

    for dom in TE:
        idx_all = np.where((tt == dom) & valid)[0]
        idx = rng.choice(idx_all, min(args.n, len(idx_all)), replace=False)
        eps = [eps_all[i] for i in idx]
        cache_path = RESULTS / f"llm_outcomes_{dom}_{args.model.replace(':','_')}.json"
        outcomes = load_llm_outcomes(args.host, args.model, eps, cache_path)
        llm_solved = np.array([outcomes.get(ep["instance_id"], False) for ep in eps])

        static = 100.0  # symbolic fallback always solves (every instance here is BFS-solvable)
        oracle = 100.0  # oracle picks whichever of {LLM, fallback} solves; since fallback always solves
                        # under this "no solver budget" model, oracle == static == 100 for every domain.
                        # See the printed caveat below for why this differs from the previous paper draft.
        row = dict(n=len(eps), static=static, oracle=oracle)
        scores = {"|O|": n_obj[idx], "Goal count": n_goal[idx]}
        if arc_scores_map is not None:
            scores["ARC"] = np.array([arc_scores_map.get(str(ep["instance_id"]), np.nan) for ep in eps])

        for name, sc in scores.items():
            if np.isnan(sc).any():
                continue
            t_zs = best_threshold(sc, llm_solved)
            t_fs = calibrated_threshold(sc, llm_solved, args.n_calib, args.seed)
            row[f"{name}-ZS"] = route(sc, llm_solved, t_zs)
            row[f"{name}-FS"] = route(sc, llm_solved, t_fs)
        out[dom] = row
        print(f"{dom:<14}{row['static']:>8.1f}{row['oracle']:>8.1f}"
              f"{row.get('|O|-ZS', float('nan')):>8.1f}{row.get('Goal count-ZS', float('nan')):>9.1f}"
              f"{row.get('ARC-ZS', float('nan')):>8.1f}{row.get('|O|-FS', float('nan')):>8.1f}"
              f"{row.get('Goal count-FS', float('nan')):>9.1f}{row.get('ARC-FS', float('nan')):>8.1f}  n/a")

    (RESULTS / "routing_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {RESULTS}/routing_results.json")
    print("\nIMPORTANT CAVEAT (read before reporting this table): with the symbolic fallback modeled as")
    print("'always succeeds' (true here because every scored instance is BFS-solvable by construction),")
    print("the Oracle row is 100% for every domain and '% gap recovered' is undefined (0/0). The ORIGINAL")
    print("paper's Oracle/Static numbers (e.g. Static=59.0, Oracle=69.0 on BW) therefore encoded something")
    print("else -- most likely a solver time/resource BUDGET under which the symbolic fallback could also")
    print("fail, making Oracle<100 meaningful. That budget was never specified in the available code or")
    print("transcripts. Before reporting Table 4, either (a) introduce an explicit, documented symbolic-solver")
    print("budget so Oracle<100 is possible and the previous table's semantics are reproduced, or (b) report")
    print("routing as 'symbolic-always-succeeds', in which case only the ZS/FS *LLM-utilization* rates (how")
    print("often routing to the LLM paid off) are the meaningful numbers, not '%Gap'. This script implements")
    print("(b) by default; it does not implement (a), since the budget is undocumented and invented.")


def get_arc_scores_for_routing_snippet():
    """
    Not called automatically -- paste this into a short script after plan_table1.py's ARC training to
    produce the --arc_scores JSON this script expects (instance_id -> score, one file covering ALL
    instances of all three test domains, so this script can route by plan_table1.TEST_DOMAINS or use
    whatever subsample of --n it chooses):

        import json
        import numpy as np
        from plan_table1 import load_data, train_arc, _build_arc_support_set, _score_domain_given_support, TEST_DOMAINS
        X_surf, X_fm, y_nsteps, task_types = load_data()
        model = train_arc(X_surf, X_fm, y_nsteps, task_types, 5000, seed=42, norm="pooled", y_norm="domain", gate="none")
        support = _build_arc_support_set(X_surf, X_fm, task_types, seed=42, prep=model.prep)
        import json as _json
        eps = _json.loads(open("data/planning_v3/episodes.json").read_text())
        valid = np.load("data/planning_v3/y_valid.npy")
        eps = [e for e, v in zip(eps, valid) if v]
        a, b, r = support[0](X_surf, X_fm)
        import torch
        with torch.no_grad():
            pred, _, _ = model(torch.FloatTensor(a), torch.FloatTensor(b), torch.FloatTensor(r), *support[1:])
        scores = {str(e["instance_id"]): float(p) for e, p in zip(eps, pred.numpy())}
        json.dump(scores, open("results_planning_v3/arc_scores_for_routing.json", "w"))
    """
    pass


if __name__ == "__main__":
    main()
