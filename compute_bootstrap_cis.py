"""
compute_bootstrap_cis.py
===========================
Recomputes the paired-bootstrap confidence intervals reported in the paper:
  - Delta|rho| = Ridge(x_s) - |O|, on each test domain
  - Delta|rho| = h_add - goal_count, on each test domain
These two deltas are not saved by any other script in this bundle; this script derives both directly
from data/planning_v3 by resampling instances 2,000 times and recomputing both correlations fresh on
each resample, rather than resampling a pre-saved point estimate.

Requires plan_step50_relaxed_heuristic_features.py to be present in the same directory, since this script
imports h_add() and ground_actions_v3() directly from it rather than reimplementing them.

USAGE:
  python compute_bootstrap_cis.py --data data/planning_v3
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_step50_relaxed_heuristic_features import h_add, ground_actions_v3

TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]
NAME = {"blocksworld": "BW", "logistics": "LOG", "mystery_blocksworld": "MBW"}


def rho(a, b):
    r = stats.spearmanr(a, b)[0]
    return float(abs(r)) if not np.isnan(r) else float("nan")


def paired_bootstrap(score_a, score_b, target, n_boot=2000, seed=0):
    """Resample instances (not predictions) and recompute both |rho| fresh each time."""
    rng = np.random.default_rng(seed)
    n = len(target)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(rho(score_a[idx], target[idx]) - rho(score_b[idx], target[idx]))
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/planning_v3")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    D = Path(args.data)
    eps = json.loads((D / "episodes.json").read_text())
    Xs = np.load(D / "X_surf.npy")
    y = np.load(D / "y_nsteps.npy").astype(float)
    valid = np.load(D / "y_valid.npy")
    tt = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])

    n_obj = np.array([e["n_objects"] for e in eps], float)
    n_goal = np.array([len(e["goal_facts"]) for e in eps], float)

    m = valid
    eps = [e for e, v in zip(eps, valid) if v]
    Xs, y, tt, n_obj, n_goal = Xs[m], y[m], tt[m], n_obj[m], n_goal[m]

    # Ridge(x_s), fit on source domains exactly as in compute_paper_numbers.py
    tr = np.isin(tt, TRAIN_DOMAINS)
    sc = StandardScaler().fit(Xs[tr])
    ridge_pred_full = Ridge(alpha=1.0).fit(sc.transform(Xs[tr]), y[tr]).predict(sc.transform(Xs))

    print(f"=== Ridge(x_s) - |O|  (n_boot={args.n_boot}) ===")
    for d in TEST_DOMAINS:
        k = tt == d
        mean, lo, hi = paired_bootstrap(ridge_pred_full[k], n_obj[k], y[k], n_boot=args.n_boot)
        print(f"  {NAME[d]:4s}  Delta={mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print(f"\n=== h_add - goal_count  (n_boot={args.n_boot}) ===")
    print("    (recomputing h_add per instance; this re-grounds every test-domain action set)")
    for d in TEST_DOMAINS:
        k = tt == d
        sub_eps = [e for e, kk in zip(eps, k) if kk]
        h_scores, goal_scores, targets = [], [], []
        for e in sub_eps:
            if e.get("n_steps", 0) <= 0:
                continue
            actions = ground_actions_v3(e)
            h_scores.append(h_add(set(e["init_facts"]), set(e["goal_facts"]), actions))
            goal_scores.append(len(e["goal_facts"]))
            targets.append(e["n_steps"])
        h_scores, goal_scores, targets = map(np.array, (h_scores, goal_scores, targets))
        mean, lo, hi = paired_bootstrap(h_scores, goal_scores, targets, n_boot=args.n_boot)
        print(f"  {NAME[d]:4s}  N={len(targets)}  Delta={mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
