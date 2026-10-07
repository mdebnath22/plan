"""
plan_step48_llm_difficulty_baseline.py
=========================================
Fills Appendix B.7 (app:llm_baseline): asks the LLM to predict difficulty directly, conditioned on the
same 20-example support block ARC conditions on, via two elicitation strategies (absolute scoring,
pairwise comparison). Compares against the already-recomputed structural baselines.

USAGE:
  python plan_step48_llm_difficulty_baseline.py --host http://localhost:11434 --model qwen2.5:72b
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

TE = ["blocksworld", "logistics", "mystery_blocksworld"]
TR = ["depot", "rovers", "satellite"]
RESULTS = Path("results_planning_v3"); RESULTS.mkdir(exist_ok=True)

ABS_TMPL = (
    "You are an expert at analyzing PDDL planning problems. Below are labeled examples from RELATED "
    "planning domains, each showing a problem and its optimal (shortest) plan length. The query problem "
    "is from a DIFFERENT domain with different predicates and actions, but similar structural principles "
    "(object counts, goal complexity, initial state structure) may transfer.\n\n"
    "=== LABELED EXAMPLES (from other domains) ===\n{support}\n\n=== QUERY DOMAIN ===\n{qdomain}\n\n"
    "=== QUERY PROBLEM (predict difficulty for this one) ===\n{qproblem}\n\n"
    "Based on the pattern in the labeled examples (how problem structure relates to plan length), "
    "estimate a DIFFICULTY SCORE for the query problem, from 0 (trivially easy, very short plan) to "
    "100 (very hard, very long plan).\nOutput ONLY a single integer between 0 and 100. No explanation.\nScore:"
)
PAIR_TMPL = (
    "You are an expert at analyzing PDDL planning problems. Below are labeled examples from RELATED "
    "planning domains, each showing a problem and its optimal (shortest) plan length.\n\n"
    "=== LABELED EXAMPLES (from other domains) ===\n{support}\n\n=== QUERY DOMAIN ===\n{qdomain}\n\n"
    "=== PROBLEM A ===\n{a}\n\n=== PROBLEM B ===\n{b}\n\n"
    "Which problem requires a LONGER optimal plan (is harder)? Output ONLY 'A' or 'B'. No explanation.\nAnswer:"
)


CTX = None   # set from --ctx in main(); sent as num_ctx on EVERY request so the server never falls back to a smaller default

def call_llm(host, model, prompt, num_predict=8, timeout=900):
    import urllib.request
    opts = {"num_predict": num_predict, "temperature": 0.0}
    if CTX:
        opts["num_ctx"] = CTX
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m", "options": opts}).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "")
    except Exception as e:
        return f"ERROR:{e}"


def build_support_block(eps_by_domain, rng, k=20):
    pool = []
    for d in TR:
        pool += eps_by_domain[d]
    chosen = rng.choice(len(pool), min(k, len(pool)), replace=False)
    lines = []
    for i in chosen:
        ep = pool[i]
        lines.append(f"Problem:\n{ep['problem_pddl']}\nOptimal plan length: {ep['n_steps']}\n")
    return "\n".join(lines)


def parse_score(resp):
    if resp.startswith("ERROR:"):
        return None
    m = re.search(r"-?\d+", resp)
    if not m:
        return None
    v = int(m.group())
    # guard against the earlier bug where an HTTP error code (e.g. 500) was parsed as a score
    if v < 0 or v > 100:
        return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--n_query", type=int, default=60, help="query instances per domain (absolute scoring)")
    ap.add_argument("--n_pairs", type=int, default=150, help="pairwise comparisons per domain")
    ap.add_argument("--data_dir", default="data/planning_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=24576,
                    help="context window requested per call. The 20-example support block makes prompts ~9.5k-16.7k tokens, "
                         "far above Ollama's 8192/4096 defaults, which would SILENTLY truncate the prompt.")
    args = ap.parse_args()
    global CTX
    CTX = args.ctx

    warm = call_llm(args.host, args.model, "Say OK.", num_predict=4)
    if warm.startswith("ERROR:"):
        print(f"ABORT: no usable LLM server at {args.host} for '{args.model}' ({warm})."); sys.exit(2)

    D = Path(args.data_dir)
    eps_all = json.loads((D / "episodes.json").read_text())
    valid = np.load(D / "y_valid.npy")
    tt = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])
    eps_by_domain = {d: [e for e, v, t in zip(eps_all, valid, tt) if v and t == d] for d in set(tt)}

    rng = np.random.default_rng(args.seed)
    support = build_support_block(eps_by_domain, rng)
    longest_dom = max((e for d in TE for e in eps_by_domain[d]), key=lambda e: len(e["domain_pddl"]) + len(e["problem_pddl"]))
    worst = max(len(ABS_TMPL.format(support=support, qdomain=longest_dom["domain_pddl"], qproblem=longest_dom["problem_pddl"])),
                len(PAIR_TMPL.format(support=support, qdomain=longest_dom["domain_pddl"], a=longest_dom["problem_pddl"], b=longest_dom["problem_pddl"])))
    est_tokens = int(worst / 2.8)
    print(f"support block: 20 examples, {len(support)} chars; worst-case full prompt ~{worst} chars ~ {est_tokens} tokens "
          f"(conservative 2.8 chars/token); requested context = {CTX}")
    if est_tokens > 0.95 * CTX:
        print(f"ABORT: worst-case prompt (~{est_tokens} tokens) would not fit in --ctx {CTX}; the server would silently truncate it, "
              "which would invalidate the baseline. Raise --ctx (and make sure the GPU has the memory for the KV cache) or use fewer support examples.")
        sys.exit(4)

    out = {"absolute": {}, "pairwise": {}, "protocol": {"n_support_examples": 20, "num_ctx": CTX, "est_worst_prompt_tokens": est_tokens}}

    print("=== Absolute scoring ===")
    for dom in TE:
        pool = eps_by_domain[dom]
        idx = rng.choice(len(pool), min(args.n_query, len(pool)), replace=False)
        scores, targets, n_unparsed = [], [], 0
        t_dom = time.time()
        for k_, i in enumerate(idx):
            if k_ % 10 == 0:
                print(f"    abs {dom}: {k_}/{len(idx)} done, {time.time()-t_dom:.0f}s elapsed", flush=True)
            ep = pool[i]
            resp = call_llm(args.host, args.model,
                            ABS_TMPL.format(support=support, qdomain=ep["domain_pddl"], qproblem=ep["problem_pddl"]))
            s = parse_score(resp)
            if s is None:
                n_unparsed += 1; continue
            scores.append(s); targets.append(ep["n_steps"])
        if len(scores) < 10:
            print(f"  {dom}: only {len(scores)} parsed scores, skipping"); continue
        rho = abs(stats.spearmanr(scores, targets)[0])
        out["absolute"][dom] = dict(n=len(scores), n_unparsed=n_unparsed, rho=float(rho))
        print(f"  {dom:22s} N={len(scores)} (unparsed={n_unparsed}) |rho|={rho:.3f}")

    print("\n=== Pairwise comparison ===")
    for dom in TE:
        pool = eps_by_domain[dom]
        n_inst = min(args.n_query * 2, len(pool))  # win-rate needs more instances than absolute scoring
        inst_idx = rng.choice(len(pool), n_inst, replace=False)
        wins = {i: 0 for i in inst_idx}; comps = {i: 0 for i in inst_idx}
        n_pairs = min(args.n_pairs, n_inst * (n_inst - 1) // 2)
        pairs_done = 0
        seen_pairs = set()
        t_dom = time.time()
        while pairs_done < n_pairs:
            if pairs_done % 10 == 0:
                print(f"    pair {dom}: {pairs_done}/{n_pairs} done, {time.time()-t_dom:.0f}s elapsed", flush=True)
            a, b = rng.choice(inst_idx, 2, replace=False)
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key); pairs_done += 1
            ea, eb = pool[a], pool[b]
            resp = call_llm(args.host, args.model,
                            PAIR_TMPL.format(support=support, qdomain=ea["domain_pddl"],
                                             a=ea["problem_pddl"], b=eb["problem_pddl"]), num_predict=4)
            ans = resp.strip().upper()
            if "A" in ans and "B" not in ans:
                wins[a] += 1
            elif "B" in ans and "A" not in ans:
                wins[b] += 1
            comps[a] += 1; comps[b] += 1
        win_rate = [wins[i] / comps[i] if comps[i] > 0 else np.nan for i in inst_idx]
        targets = [pool[i]["n_steps"] for i in inst_idx]
        mask = ~np.isnan(win_rate)
        rho = abs(stats.spearmanr(np.array(win_rate)[mask], np.array(targets)[mask])[0])
        out["pairwise"][dom] = dict(n_instances=int(n_inst), n_pairs=pairs_done,
                                    avg_comparisons_per_instance=float(np.mean(list(comps.values()))), rho=float(rho))
        print(f"  {dom:22s} N_instances={n_inst} N_pairs={pairs_done} "
              f"avg_comparisons/instance={np.mean(list(comps.values())):.1f}  |rho|={rho:.3f}")

    (RESULTS / f"llm_difficulty_baseline_{args.model.replace(':','_')}.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> results_planning_v3/llm_difficulty_baseline_{args.model.replace(':','_')}.json")

    def mean_of(d):
        return float(np.mean([d[k]["rho"] for k in TE if k in d])) if d else float("nan")
    print(f"\nMean |rho|: absolute={mean_of(out['absolute']):.3f}  pairwise={mean_of(out['pairwise']):.3f}")
    print("Compare against the already-recomputed structural baselines (|O|=0.657, goal count=0.802, "
          "Ridge on x_s=0.707) to see whether in-context LLM reasoning matches them.")


if __name__ == "__main__":
    main()
