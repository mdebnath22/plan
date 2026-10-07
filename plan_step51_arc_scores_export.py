"""
plan_step51_arc_scores_export.py
===================================
plan_table1.py trains and scores ARC but never SAVES per-instance predictions, so three pending items had
no input: (1) the ARC column of the routing table, (2) ARC on the three custom non-IPC domains, and
(3) prediction-level seed ensembling. This script trains the main ARC configuration for each seed and writes
all three.

For each seed it trains ARC exactly as plan_table1.py does (train_arc, same arguments), then scores
  * every BFS-solved instance of the main dataset   -> routing input + test-domain correlations
  * every BFS-solved instance of the custom domains -> custom-domain ARC column (if data/custom_domains_v3 exists
                                                       and its X_fm.npy holds REAL embeddings)
and finally averages the PREDICTIONS across seeds (an ensemble) before correlating once.

Outputs (results_planning_v3/):
  arc_scores_for_routing{_tag}.json     {instance_id: ensemble score}      (plan_step46_routing.py --arc_scores)
  arc_custom_domain_scores{_tag}.json   per-domain |rho| of ARC / ensemble on the custom domains
  arc_seed_ensemble{_tag}.json          per-seed vs ensemble |rho| on the test domains

Conventions (identical to the rest of the paper): test-domain correlations use ALL BFS-solved instances
(N=126/164/133, as in Table 1); custom-domain correlations exclude n*=0 (as plan_step49 --score does).
Lower score = structurally easier, which is the direction the routing script assumes.

USAGE:
  python plan_step51_arc_scores_export.py --seeds 0 1 2 42 123
  python plan_step51_arc_scores_export.py --gate blend --tag gate_blend
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_table1 import (load_data, train_arc, _build_arc_support_set, TEST_DOMAINS, RESULTS_DIR, DATA_DIR)

CUSTOM_DIR = Path("data/custom_domains_v3")


def rho_abs(a, b):
    r = stats.spearmanr(a, b)[0]
    return float(abs(r)) if not np.isnan(r) else float("nan")


def predict(model, support, X_surf_new, X_fm_new):
    """Score arbitrary instances with a trained model and its (fixed) support set."""
    transform, S_surf, S_fm, S_V = support
    a, b, r = transform(X_surf_new, X_fm_new)
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(torch.FloatTensor(a), torch.FloatTensor(b), torch.FloatTensor(r), S_surf, S_fm, S_V)
    return np.asarray(pred.numpy(), dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42, 123])
    ap.add_argument("--n_episodes", type=int, default=5000)
    ap.add_argument("--gate", default="none", choices=["none", "blend", "zeff"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--check_against", default=None,
                    help="path to a plan_table1.py results JSON (e.g. results_planning_v3/table1_final_main.json): "
                         "prints how closely this script's per-seed correlations reproduce its ARC per-seed results")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    X_surf, X_fm, y, tt = load_data()
    eps = json.loads((DATA_DIR / "episodes.json").read_text())
    valid = np.load(DATA_DIR / "y_valid.npy")
    ids = [e["instance_id"] for e, v in zip(eps, valid) if v]      # same filter/order as load_data()
    assert len(ids) == len(tt), (len(ids), len(tt))

    have_custom = (CUSTOM_DIR / "X_fm.npy").exists()
    if have_custom:
        c_eps = json.loads((CUSTOM_DIR / "episodes.json").read_text())
        c_Xs = np.load(CUSTOM_DIR / "X_surf.npy"); c_Xf = np.load(CUSTOM_DIR / "X_fm.npy")
        c_y = np.load(CUSTOM_DIR / "y_nsteps.npy"); c_v = np.load(CUSTOM_DIR / "y_valid.npy")
        c_tt = np.array([str(t) for t in np.load(CUSTOM_DIR / "task_types.npy", allow_pickle=True)])
        c_ok = c_v & (c_y > 0)                                       # n*>0, as in plan_step49 --score
        c_ids = [e["instance_id"] for e, ok in zip(c_eps, c_ok) if ok]
        c_Xs, c_Xf, c_y, c_tt = c_Xs[c_ok], c_Xf[c_ok], c_y[c_ok], c_tt[c_ok]
    else:
        print(f"NOTE: {CUSTOM_DIR}/X_fm.npy not found -> custom-domain scoring skipped "
              "(generate it with plan_step49_custom_domains_v3.py on a node with sentence-transformers).")

    per_seed_main, per_seed_custom = [], []
    for seed in args.seeds:
        print(f"\n--- training ARC (gate={args.gate}) seed {seed} ---", flush=True)
        model = train_arc(X_surf, X_fm, y, tt, args.n_episodes, seed, verbose=False,
                          norm="pooled", y_norm="domain", gate=args.gate)
        support = _build_arc_support_set(X_surf, X_fm, tt, seed=seed, prep=model.prep)
        per_seed_main.append(predict(model, support, X_surf, X_fm))
        rhos = {d: rho_abs(per_seed_main[-1][tt == d], y[tt == d]) for d in TEST_DOMAINS}
        print(f"    seed {seed} test-domain |rho|: " + "  ".join(f"{d}={v:.3f}" for d, v in rhos.items()), flush=True)
        if have_custom:
            per_seed_custom.append(predict(model, support, c_Xs, c_Xf))

    P = np.stack(per_seed_main, axis=0)               # [S, N]
    ens = P.mean(axis=0)
    summary = {"gate": args.gate, "seeds": args.seeds, "test_domains": {}}
    print("\n=== test domains: per-seed vs prediction-level ensemble (|rho|) ===")
    for d in TEST_DOMAINS:
        k = tt == d
        per = [rho_abs(P[s][k], y[k]) for s in range(len(args.seeds))]
        e = rho_abs(ens[k], y[k])
        summary["test_domains"][d] = dict(per_seed=per, mean=float(np.mean(per)), std=float(np.std(per)), ensemble=e, n=int(k.sum()))
        print(f"  {d:22s} N={k.sum():3d}  per-seed mean={np.mean(per):.3f} (+/- {np.std(per):.3f})   ENSEMBLE={e:.3f}")
    means = [summary["test_domains"][d]["mean"] for d in TEST_DOMAINS]; ens_means = [summary["test_domains"][d]["ensemble"] for d in TEST_DOMAINS]
    summary["mean_over_domains"] = dict(per_seed_mean=float(np.mean(means)), ensemble=float(np.mean(ens_means)))
    print(f"  {'MEAN':22s}        per-seed mean={np.mean(means):.3f}                    ENSEMBLE={np.mean(ens_means):.3f}")

    if args.check_against and Path(args.check_against).exists():
        saved = json.loads(Path(args.check_against).read_text())
        sp = saved.get("arc_per_seed")
        if sp and len(sp) == len(args.seeds):
            diffs = [abs(summary["test_domains"][d]["per_seed"][i] - sp[i][d]) for i in range(len(sp)) for d in TEST_DOMAINS]
            print(f"\nConsistency vs {args.check_against}: max |difference| in per-seed |rho| = {max(diffs):.4f}"
                  " (expected ~0 on the same hardware/software; small differences can come from CPU vs GPU or library versions)")
        else:
            print(f"\n(--check_against: no comparable 'arc_per_seed' in {args.check_against} for these seeds)")

    (RESULTS_DIR / f"arc_scores_for_routing{suffix}.json").write_text(json.dumps({str(i): float(s) for i, s in zip(ids, ens)}))
    (RESULTS_DIR / f"arc_seed_ensemble{suffix}.json").write_text(json.dumps(summary, indent=2))

    if have_custom:
        PC = np.stack(per_seed_custom, axis=0); ensc = PC.mean(axis=0)
        cust = {}
        print("\n=== custom domains (n*>0): per-seed vs ensemble (|rho|) ===")
        for d in sorted(set(c_tt)):
            k = c_tt == d
            if k.sum() < 10:
                print(f"  {d}: only {k.sum()} instances, skipped"); continue
            per = [rho_abs(PC[s][k], c_y[k]) for s in range(len(args.seeds))]
            cust[d] = dict(n=int(k.sum()), per_seed=per, mean=float(np.mean(per)), std=float(np.std(per)),
                           ensemble=rho_abs(ensc[k], c_y[k]))
            print(f"  {d:10s} N={k.sum():3d}  per-seed mean={cust[d]['mean']:.3f} (+/- {cust[d]['std']:.3f})  ENSEMBLE={cust[d]['ensemble']:.3f}")
        (RESULTS_DIR / f"arc_custom_domain_scores{suffix}.json").write_text(json.dumps(cust, indent=2))

    print(f"\nSaved -> {RESULTS_DIR}/arc_scores_for_routing{suffix}.json, arc_seed_ensemble{suffix}.json"
          + (f", arc_custom_domain_scores{suffix}.json" if have_custom else ""))


if __name__ == "__main__":
    main()
