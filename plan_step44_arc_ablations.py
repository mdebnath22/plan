"""
plan_step44_arc_ablations.py
===============================
Fills the ARC-design appendix tables that need torch + real embeddings:
  --suite attn      Appendix C.2 (attention directionality): sym-semantic, reversed, normalized-key, ARC's own.
  --suite repfamily  Appendix C.1 (representation family): Ridge(x_e), Ridge(x_s||x_e) -- the x_s row already
                     exists in plan_table1.py's output and is not recomputed here.
  --suite resid      Appendix C.4 (residualisation sensitivity): PCA dim in {5,10,20,30,50}, Ridge alpha in
                     {0.01,0.1,1,10,100}, holding the other at its default.
  --suite loo        Appendix B.2 (leave-one-domain-out) WITH ARC, retrained from scratch per fold.
  --suite within     Table 2 / within-size discrimination WITH ARC, scored on the held-out size buckets.
  --suite all        run everything above in sequence.

Imports PlanningGURU/train_arc/eval_arc/etc. from plan_table1.py so this uses EXACTLY the same model and
training code as Table 1 -- these are ablations of that same model, not a reimplementation.

USAGE (run on the cluster, after plan_table1.py's main results):
  python plan_step44_arc_ablations.py --suite all --real_emb --seeds 0 1 2 42 123
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_table1 import (
    load_data, PooledPrep, PlanningGURU, MetaSampler, train_arc, eval_arc,
    _build_arc_support_set, _score_domain_given_support,
    TRAIN_DOMAINS, TEST_DOMAINS, VAL_DOMAIN,
)

RESULTS = Path("results_planning_v3"); RESULTS.mkdir(exist_ok=True)
SEVEN = ["blocksworld", "depot", "gripper", "logistics", "mystery_blocksworld", "rovers", "satellite"]


def mean_of(d):
    vals = [v for v in d.values() if isinstance(v, float) and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ============================================================================
# Attention directionality. Reuses PlanningGURU's forward EXCEPT for which
# tensor plays query and which plays key, controlled by a thin wrapper that
# swaps which encoder is applied to which input before calling the same
# attention/fusion code. This keeps the fusion, gate, and loss code IDENTICAL
# across variants; only the query/key assignment changes.
# ============================================================================

class AttnVariantGURU(PlanningGURU):
    """mode:
       'asym'     : semantic query / structural key      (ARC's own configuration, for reference)
       'sym_sem'  : semantic query / semantic key         (query_enc applied to BOTH q_fm and S_fm)
       'reversed' : structural query / semantic key       (roles swapped)
       'norm_key' : semantic query / L2-normalized structural key
    """
    def __init__(self, surf_dim, fm_dim, d_model=128, mode="asym"):
        super().__init__(surf_dim, fm_dim, d_model=d_model, gate_mode="none")
        assert mode in ("asym", "sym_sem", "reversed", "norm_key")
        self.mode = mode
        # BUG FIXED: 'sym_sem' (both query and key consume the 768-dim FM embedding) was missing
        # from this dimension-rebuild block entirely. key_enc was left at the parent class's default
        # (built for the 28-dim structural input), so forward()'s sym_sem branch -- which feeds it
        # S_fm (768-dim) -- crashed with a shape mismatch (confirmed: "60x768 and 28x512").
        if mode == "reversed":
            # query must now consume x_s (surf_dim) and key must consume x_e (fm_dim):
            # rebuild the two encoders with swapped input dims, same architecture otherwise.
            self.query_enc = nn.Sequential(nn.Linear(surf_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                           nn.Dropout(0.1), nn.Linear(512, d_model), nn.LayerNorm(d_model))
            self.key_enc = nn.Sequential(nn.Linear(fm_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                         nn.Dropout(0.1), nn.Linear(512, d_model), nn.LayerNorm(d_model))
        elif mode == "sym_sem":
            # query stays as the parent's default (consumes x_e, fm_dim -- already correct); only
            # key_enc needs rebuilding, since both query AND key now consume the fm_dim embedding.
            self.key_enc = nn.Sequential(nn.Linear(fm_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                         nn.Dropout(0.1), nn.Linear(512, d_model), nn.LayerNorm(d_model))
        # mode == "norm_key": key_enc stays at the parent's default (surf_dim input) -- correct,
        # since this variant only L2-normalizes S_surf before feeding the SAME surf_dim encoder.
        # mode == "asym": no changes needed; the parent's default (query=fm_dim, key=surf_dim) is
        # already ARC's own configuration.

    def forward(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V):
        if self.mode == "asym":
            q_in, k_in = q_fm, S_surf
        elif self.mode == "sym_sem":
            q_in, k_in = q_fm, S_fm
        elif self.mode == "reversed":
            q_in, k_in = q_surf, S_fm
        elif self.mode == "norm_key":
            q_in = q_fm
            k_in = S_surf / (S_surf.norm(dim=-1, keepdim=True) + 1e-8)
        q = self.query_enc(q_in.unsqueeze(0) if q_in.dim() == 1 else q_in)
        k = self.key_enc(k_in)
        v = self.value_proj(S_V)
        alpha = torch.softmax(torch.matmul(q, k.T) * self.scale, dim=-1)
        z = torch.matmul(alpha, v)
        if q_in.dim() == 1: z = z.squeeze(0); alpha = alpha.squeeze(0)
        fused = torch.cat([q_surf, z, q_resid], dim=-1)
        feats = self.fusion(fused)
        y = self.head_reg(feats).squeeze(-1)
        return y, feats, alpha


def run_attn_variant(X_surf, X_fm, y_nsteps, task_types, mode, n_episodes, seed, val_every=100, patience=10):
    """Matched-budget training loop, structurally identical to train_arc() but using AttnVariantGURU."""
    torch.manual_seed(seed)
    tr_mask = np.isin(task_types, TRAIN_DOMAINS)
    prep = PooledPrep(X_surf[tr_mask], X_fm[tr_mask])
    sampler = MetaSampler(TRAIN_DOMAINS, X_surf, X_fm, y_nsteps, task_types, seed=seed, prep=prep, y_norm="domain")
    model = AttnVariantGURU(X_surf.shape[1], X_fm.shape[1], mode=mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
    val_support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=prep)

    import torch.nn.functional as F
    best, best_state, since = -1.0, None, 0
    for ep in range(n_episodes):
        d = sampler.sample_episode()
        pred, feats, alpha = model(d["Q_surf"], d["Q_fm"], d["Q_resid"], d["S_surf"], d["S_fm"], d["S_V"])
        y_target = d["Y_norm"] if d["Y_norm"] is not None else (d["Y"] - d["Y"].mean()) / (d["Y"].std() + 1e-6)
        loss = F.mse_loss(pred, y_target)
        ent = -(alpha * (alpha + 1e-8).log()).sum(-1).mean()
        loss = loss - 0.05 * ent
        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (ep + 1) % val_every == 0:
            s = _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, VAL_DOMAIN, *val_support)
            if s is not None:
                if s > best:
                    best, best_state, since = s, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    since += 1
                if since >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.prep = prep
    results = {}
    support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=prep)
    for d_ in TEST_DOMAINS:
        results[d_] = _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, d_, *support) or float("nan")
    return results, best, feats


def suite_attn(X_surf, X_fm, y_nsteps, task_types, seeds, n_episodes):
    print("\n=== Attention directionality ablation (matched budget, 5 seeds) ===")
    out = {}
    for mode in ["sym_sem", "reversed", "norm_key", "asym"]:
        per_seed = []
        feats_all = []
        for seed in seeds:
            r, grip, feats = run_attn_variant(X_surf, X_fm, y_nsteps, task_types, mode, n_episodes, seed)
            per_seed.append(r); feats_all.append(feats.detach().numpy())
        means = {d: float(np.nanmean([r[d] for r in per_seed])) for d in TEST_DOMAINS}
        means["mean"] = mean_of(means)
        # geometry diagnostics on the learned fused representation h (pooled across seeds' query batches)
        H = np.concatenate(feats_all, axis=0)
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        cos = float((Hn @ Hn.T - np.eye(len(Hn))).sum() / (len(Hn) * (len(Hn) - 1)))
        sv = np.linalg.svd(H, compute_uv=False); p = sv / sv.sum()
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        out[mode] = dict(per_domain=means, cos_h=cos, eff_rank_h=eff_rank)
        print(f"  {mode:10s} cos(h)={cos:.3f}  eff_rank(h)={eff_rank:.2f}  mean|rho|={means['mean']:.3f}  {means}")
    return out


# ============================================================================
# Representation-family: non-episodic Ridge on x_e and on x_s||x_e. x_s-only
# row is already in plan_table1.py's table1_final_*.json; not recomputed here.
# ============================================================================

def suite_repfamily(X_surf, X_fm, y_nsteps, task_types):
    print("\n=== Representation-family ablation (non-episodic Ridge) ===")
    tr = np.isin(task_types, TRAIN_DOMAINS)
    out = {}
    for name, X in [("Ridge (x_e)", X_fm), ("Ridge (x_s || x_e)", np.hstack([X_surf, X_fm]))]:
        sc = StandardScaler().fit(X[tr]); r = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y_nsteps[tr])
        pred = r.predict(sc.transform(X))
        row = {d: float(abs(stats.spearmanr(pred[task_types == d], y_nsteps[task_types == d])[0])) for d in TEST_DOMAINS}
        row["mean"] = mean_of(row); out[name] = row
        print(f"  {name:20s} {row}")
    return out


# ============================================================================
# Residualisation sensitivity: sweep PCA dim and Ridge alpha used inside
# PooledPrep, retraining ARC (main config) at each setting, 5 seeds.
# ============================================================================

def suite_resid(X_surf, X_fm, y_nsteps, task_types, seeds, n_episodes):
    print("\n=== Residualisation sensitivity (5 seeds each) ===")
    out = {"pca_dim": {}, "ridge_alpha": {}}
    tr_mask = np.isin(task_types, TRAIN_DOMAINS)

    def run_once(n_comp, alpha, seed):
        class PatchedPrep(PooledPrep):
            def __init__(self, Xs, Xf, n_comp=n_comp):
                self.ss = StandardScaler().fit(Xs); self.se = StandardScaler().fit(Xf)
                a, b = self.ss.transform(Xs), self.se.transform(Xf)
                from sklearn.decomposition import PCA
                from sklearn.pipeline import Pipeline
                self.res = Pipeline([("pca", PCA(n_components=min(n_comp, a.shape[1]))),
                                     ("ridge", Ridge(alpha=alpha))]).fit(a, b)
        torch.manual_seed(seed)
        prep = PatchedPrep(X_surf[tr_mask], X_fm[tr_mask])
        sampler = MetaSampler(TRAIN_DOMAINS, X_surf, X_fm, y_nsteps, task_types, seed=seed, prep=prep, y_norm="domain")
        model = PlanningGURU(X_surf.shape[1], X_fm.shape[1], gate_mode="none")
        model.prep = prep
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
        val_support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=prep)
        import torch.nn.functional as F
        best, best_state, since = -1.0, None, 0
        for ep in range(n_episodes):
            d = sampler.sample_episode()
            pred, _, alpha_w = model(d["Q_surf"], d["Q_fm"], d["Q_resid"], d["S_surf"], d["S_fm"], d["S_V"])
            y_target = d["Y_norm"]
            loss = F.mse_loss(pred, y_target) - 0.05 * (-(alpha_w * (alpha_w + 1e-8).log()).sum(-1).mean())
            if torch.isnan(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if (ep + 1) % 100 == 0:
                s = _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, VAL_DOMAIN, *val_support)
                if s is not None:
                    if s > best: best, best_state, since = s, {k: v.clone() for k, v in model.state_dict().items()}, 0
                    else: since += 1
                    if since >= 10: break
        if best_state is not None: model.load_state_dict(best_state)
        support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=prep)
        r = {d_: _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, d_, *support) or float("nan") for d_ in TEST_DOMAINS}
        return mean_of(r)

    for n_comp in [5, 10, 20, 30, 50]:
        vals = [run_once(n_comp, 1.0, s) for s in seeds]
        out["pca_dim"][n_comp] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
        print(f"  PCA dim={n_comp:3d}  mean|rho|={np.mean(vals):.3f} (+/- {np.std(vals):.3f})")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        vals = [run_once(20, alpha, s) for s in seeds]
        out["ridge_alpha"][alpha] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
        print(f"  Ridge alpha={alpha:<6g}  mean|rho|={np.mean(vals):.3f} (+/- {np.std(vals):.3f})")
    return out


# ============================================================================
# Leave-one-domain-out WITH ARC (retrained per fold), 5 seeds each.
# ============================================================================

def suite_loo(X_surf, X_fm, y_nsteps, task_types, seeds, n_episodes):
    """
    IMPORTANT: train_arc(), MetaSampler, and _build_arc_support_set all read the training-domain list via
    the module-level globals plan_table1.TRAIN_DOMAINS (and train_arc separately reads plan_table1.VAL_DOMAIN
    for its early-stopping check) rather than taking them as a function argument. Python resolves a bare
    name used inside a function at CALL time via that function's __globals__ (here, plan_table1's own
    module namespace), so temporarily overwriting plan_table1.TRAIN_DOMAINS / plan_table1.VAL_DOMAIN before
    calling train_arc correctly makes every one of those internal helpers use the 6-domain fold instead of
    the fixed 3-domain primary split, with no need to duplicate train_arc's ~90-line training loop here.
    The constants are restored after every fold (and in a try/finally) so later suites in the same run are
    unaffected if this one is interrupted.

    One domain-specific wrinkle: Gripper is the primary split's validation domain, so when Gripper itself is
    the held-out (test) domain for a fold, it cannot also serve as that fold's validation domain. We fall
    back to Depot as the validation domain for that one fold only; this is a visible, documented choice, not
    a silent default, and it is the only fold where the validation domain differs from Gripper.
    """
    import plan_table1 as pt
    orig_train, orig_val = pt.TRAIN_DOMAINS, pt.VAL_DOMAIN
    print("\n=== Leave-one-domain-out WITH ARC (retrained per fold, 5 seeds) ===")
    out = {}
    try:
        for held in SEVEN:
            train_doms = [d for d in SEVEN if d != held]
            val_dom = "gripper" if held != "gripper" else "depot"
            pt.TRAIN_DOMAINS = train_doms
            pt.VAL_DOMAIN = val_dom
            per_seed = []
            for seed in seeds:
                model = train_arc(X_surf, X_fm, y_nsteps, task_types, n_episodes, seed,
                                  verbose=False, norm="pooled", y_norm="domain", gate="none")
                prep = model.prep
                support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=prep)
                s = _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, held, *support)
                per_seed.append(s if s is not None else float("nan"))
            out[held] = dict(mean=float(np.nanmean(per_seed)), std=float(np.nanstd(per_seed)),
                             per_seed=per_seed, train_domains=train_doms, val_domain=val_dom)
            print(f"  held-out {held:22s} (val={val_dom}) ARC mean={out[held]['mean']:.3f} "
                  f"(+/- {out[held]['std']:.3f})  per-seed={per_seed}")
    finally:
        pt.TRAIN_DOMAINS, pt.VAL_DOMAIN = orig_train, orig_val
    return out


# ============================================================================
# Within-size discrimination WITH ARC: score the already-trained main-config
# model (5 seeds) on each size bucket of each test domain.
# ============================================================================

def suite_within(X_surf, X_fm, y_nsteps, task_types, eps_records, seeds, n_episodes):
    print("\n=== Within-size discrimination WITH ARC (5 seeds) ===")
    complexity = np.array([e.get("complexity", -1) for e in eps_records])
    out = {}
    models = []
    for seed in seeds:
        m = train_arc(X_surf, X_fm, y_nsteps, task_types, n_episodes, seed, verbose=False,
                      norm="pooled", y_norm="domain", gate="none")
        models.append(m)
    for d in TEST_DOMAINS:
        out[d] = {}
        for lvl in sorted(set(complexity[(task_types == d)])):
            if lvl < 0: continue
            mask = (task_types == d) & (complexity == lvl)
            if mask.sum() < 5:
                continue
            per_seed = []
            for seed, model in zip(seeds, models):
                support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=model.prep)
                a, b, r = support[0](X_surf[mask], X_fm[mask])
                with torch.no_grad():
                    pred, _, _ = model(torch.FloatTensor(a), torch.FloatTensor(b), torch.FloatTensor(r), *support[1:])
                rho = stats.spearmanr(pred.numpy(), y_nsteps[mask])[0]
                per_seed.append(float(abs(rho)) if not np.isnan(rho) else float("nan"))
            out[d][f"B{lvl+1}"] = dict(n=int(mask.sum()), mean=float(np.nanmean(per_seed)), std=float(np.nanstd(per_seed)))
        means = [out[d][b]["mean"] for b in out[d]]
        out[d]["_mean_over_buckets"] = float(np.nanmean(means))
        print(f"  {d:22s} " + "  ".join(f"{b}: N={out[d][b]['n']} rho={out[d][b]['mean']:.2f}" for b in out[d] if b.startswith("B"))
              + f"   mean={out[d]['_mean_over_buckets']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["attn", "repfamily", "resid", "loo", "within", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42, 123])
    ap.add_argument("--n_episodes", type=int, default=5000)
    ap.add_argument("--data_dir", default="data/planning_v3")
    args = ap.parse_args()

    X_surf, X_fm, y_nsteps, task_types = load_data()
    eps_records = json.loads((Path(args.data_dir) / "episodes.json").read_text())
    valid = np.load(Path(args.data_dir) / "y_valid.npy")
    eps_records = [e for e, v in zip(eps_records, valid) if v]
    assert len(eps_records) == len(task_types), (len(eps_records), len(task_types))

    # One output file PER SUITE when a single suite is requested, so suites can run as parallel jobs
    # (run_arc_ablations.sbatch) without overwriting each other's results; `--suite all` keeps the old name.
    out_path = RESULTS / ("arc_ablations.json" if args.suite == "all" else f"arc_ablations_{args.suite}.json")
    out = {}
    if args.suite in ("attn", "all"):
        out["attention_direction"] = suite_attn(X_surf, X_fm, y_nsteps, task_types, args.seeds, args.n_episodes)
        json.dump(out, open(out_path, "w"), indent=2, default=str)
    if args.suite in ("repfamily", "all"):
        out["representation_family"] = suite_repfamily(X_surf, X_fm, y_nsteps, task_types)
        json.dump(out, open(out_path, "w"), indent=2, default=str)
    if args.suite in ("resid", "all"):
        out["residual_sensitivity"] = suite_resid(X_surf, X_fm, y_nsteps, task_types, args.seeds, args.n_episodes)
        json.dump(out, open(out_path, "w"), indent=2, default=str)
    if args.suite in ("loo", "all"):
        out["loo_arc"] = suite_loo(X_surf, X_fm, y_nsteps, task_types, args.seeds, args.n_episodes)
        json.dump(out, open(out_path, "w"), indent=2, default=str)
    if args.suite in ("within", "all"):
        out["within_size_arc"] = suite_within(X_surf, X_fm, y_nsteps, task_types, eps_records, args.seeds, args.n_episodes)
        json.dump(out, open(out_path, "w"), indent=2, default=str)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
