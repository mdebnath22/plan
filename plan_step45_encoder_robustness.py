"""
plan_step45_encoder_robustness.py
=====================================
Fills Appendix C.3 (app:encoder_robustness): within-domain cosine similarity and effective rank of the FM
embeddings, computed with three different sentence-transformer encoders on the CORRECTED (leak-free)
descriptions. Needs only sentence-transformers + the v3 episodes.json; no ARC, no torch training.

USAGE:
  python plan_step45_encoder_robustness.py --data data/planning_v3
"""
import argparse, json
from pathlib import Path

import numpy as np

ENCODERS = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"]
TE = ["blocksworld", "logistics", "mystery_blocksworld"]


def stats_for(emb):
    Mn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    S = Mn @ Mn.T
    cos = float((S.sum() - len(S)) / (len(S) * (len(S) - 1)))
    sv = np.linalg.svd(emb, compute_uv=False)
    p = sv / sv.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return cos, eff_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/planning_v3")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    eps = json.loads((Path(args.data) / "episodes.json").read_text())
    valid = np.load(Path(args.data) / "y_valid.npy")
    tt_full = np.array([str(t) for t in np.load(Path(args.data) / "task_types.npy", allow_pickle=True)])
    # BUG FIXED: tt must be filtered by the SAME valid mask as eps -- tt_full is the unfiltered
    # (length-1400) array; using it directly against the filtered (length-1061) eps list caused
    # assert len(eps) == len(tt) to fail every time.
    eps = [e for e, v in zip(eps, valid) if v]
    tt = tt_full[valid]
    assert len(eps) == len(tt), (len(eps), len(tt))

    # explicit check that no leak-like word is present -- if this trips, STOP, do not trust the numbers below
    leak_words = ["estimated", "easy", "medium", "hard", "difficulty", "steps."]
    leaked = [e for e in eps if any(w in e["description"].lower() for w in leak_words)]
    assert not leaked, f"{len(leaked)} descriptions contain leak-like words -- fix the generator before running this."
    print(f"Leak check passed: 0/{len(eps)} descriptions contain a leak-like word.\n")

    descs = [e["description"] for e in eps]
    out = {}
    for name in ENCODERS:
        print(f"Encoding with {name} ...")
        model = SentenceTransformer(name)
        emb = model.encode(descs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        row = {}
        for d in TE:
            cos, er = stats_for(emb[tt == d])
            row[d] = dict(cosine=cos, eff_rank=er)
        mean_cos = float(np.mean([row[d]["cosine"] for d in TE]))
        row["mean_cosine"] = mean_cos
        row["mean_eff_rank"] = float(np.mean([row[d]["eff_rank"] for d in TE]))
        out[name] = row
        print(f"  {name}: " + "  ".join(f"{d}={row[d]['cosine']:.4f}(rank {row[d]['eff_rank']:.2f})" for d in TE)
              + f"  mean cosine={mean_cos:.4f}")

    RES = Path("results_planning_v3"); RES.mkdir(exist_ok=True)
    (RES / "encoder_robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {RES}/encoder_robustness.json")
    print("\n=== LaTeX rows ===")
    for name, row in out.items():
        print(f"\\texttt{{{name}}} & {row['blocksworld']['cosine']:.4f} & {row['logistics']['cosine']:.4f} & "
              f"{row['mystery_blocksworld']['cosine']:.4f} & {row['mean_cosine']:.4f} & {row['mean_eff_rank']:.2f} \\\\")


if __name__ == "__main__":
    main()
