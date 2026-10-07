"""
plan_step53_rf_concat.py
===========================
Fills the one remaining row of the learned-baselines table: Random Forest trained on the concatenation
x_s || x_e, scored zero-shot on the test domains. Uses the SAME RandomForestRegressor configuration as the
existing Random Forest (x_s) row (n_estimators=300, random_state=0, n_jobs=1) and the same train/eval
protocol (fit on source domains, score |Spearman rho| on each test domain), so the new row is directly
comparable to the rest of Table "Learned Baselines" / Appendix B.3.

Needs real x_e (sentence-transformer embeddings), i.e. X_fm.npy with real embeddings, not the random
fallback -- same requirement as plan_step45_encoder_robustness.py.

USAGE:
  python plan_step53_rf_concat.py --data data/planning_v3
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/planning_v3")
    args = ap.parse_args()

    D = Path(args.data)
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("ABORT: sentence-transformers is not installed. Without it, X_fm.npy may hold the random "
              "fallback embedding, which would make this row meaningless. Install it first:\n"
              "  pip install sentence-transformers")
        return

    Xs = np.load(D / "X_surf.npy")
    Xf = np.load(D / "X_fm.npy")
    y = np.load(D / "y_nsteps.npy").astype(float)
    valid = np.load(D / "y_valid.npy")
    tt = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])

    m = valid
    Xs, Xf, y, tt = Xs[m], Xf[m], y[m], tt[m]
    X = np.hstack([Xs, Xf])

    tr = np.isin(tt, TRAIN_DOMAINS)
    sc = StandardScaler().fit(X[tr])
    model = RandomForestRegressor(300, random_state=0, n_jobs=1).fit(sc.transform(X[tr]), y[tr])
    pred = model.predict(sc.transform(X))

    results = {}
    for d in TEST_DOMAINS:
        k = tt == d
        rho, _ = stats.spearmanr(pred[k], y[k])
        results[d] = float(abs(rho)) if not np.isnan(rho) else float("nan")
    results["mean"] = float(np.mean([results[d] for d in TEST_DOMAINS]))

    print("Random Forest (x_s || x_e):")
    print(f"  BW={results['blocksworld']:.3f}  LOG={results['logistics']:.3f}  "
          f"MBW={results['mystery_blocksworld']:.3f}  mean={results['mean']:.3f}")
    print("\nFor comparison, the already-computed rows from Table \"Learned Baselines\":")
    print("  Random Forest (x_s)     BW=0.187  LOG=0.735  MBW=0.240  mean=0.388")
    print("  Ridge (x_s)             BW=0.644  LOG=0.889  MBW=0.587  mean=0.707")

    out_dir = Path("results_planning_v3"); out_dir.mkdir(exist_ok=True)
    (out_dir / "rf_concat_scores.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_dir}/rf_concat_scores.json")


if __name__ == "__main__":
    main()
