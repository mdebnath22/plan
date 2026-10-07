"""
plan_table1.py
================
THE single canonical script that produces Table 1. Every number in the
table is computed here, from the corrected data in data/planning_v3/,
with nothing hardcoded. If you need to regenerate Table 1, run this
script and this script only -- do not hand-copy numbers from any other
script, notebook, or previous run into a paper draft or into another
script's "known good" dict (this is exactly the bug found in
plan_step22_extended_baselines.py's ARC_KNOWN = {...} literal).

Every baseline (|O|, LogME, H-score, ERM, ARC) is trained/computed on
the SAME meta-train domains (Depot, Rovers, Satellite) and evaluated
zero-shot on the SAME meta-test domains (Blocksworld, Logistics,
Mystery-Blocksworld), using ONLY instances with valid==True.

ARC DESIGN SWITCHES (--norm / --y_norm / --gate)
  Defaults are the settings supported by the dev-set ablation (see the
  PROTOCOL note below); the ORIGINAL pipeline is still available for a
  like-for-like comparison with `--norm episode --y_norm episode`.

  --norm pooled|episode
      pooled : ONE StandardScaler + PCA/Ridge residualiser fit on the pooled
               TRAIN-domain rows, used identically for training episodes,
               Gripper validation and test-domain scoring (fit on source
               domains only, as the paper states).
      episode: original behaviour -- a fresh scaler/residualiser is fit on
               every episode's single-domain support set during training,
               but a DIFFERENT (pooled-mixture) scaler is used at test time.
               That train/test normalisation mismatch is the code-level
               asymmetry this switch exists to remove. ERM never had it.
  --y_norm domain|episode
      domain : regress targets z-scored with that source domain's own
               mean/std (computed once from its labelled instances).
      episode: original -- z-score using the 30 query instances' sample
               statistics (noisy).
  --gate none|blend|zeff
      none  : no gate (the code that produced all earlier cluster numbers).
      blend : y = c * (w^T x_s) + (1 - c) * y_attn, c = sigmoid(MLP(x_s)).
              One reading of the paper's "complexity-aware gate combines the
              linear predictor with the attention output".
      zeff  : z_eff = (1 - c) * z fed to fusion (the other reading; it was
              worse on the dev set and is kept only for completeness).

PROTOCOL (to keep the test domains honest)
  These defaults were chosen on DEV data only (leave-one-train-domain-out
  over Depot/Rovers/Satellite, plus Gripper), NOT on Blocksworld/Logistics/
  Mystery-Blocksworld. On the dev set `--gate none` and `--gate blend` were a
  wash (4-domain dev mean 0.478 vs 0.491; blend better on Depot/Satellite,
  worse on Gripper). So do NOT choose between them by their test scores:
  run BOTH, report BOTH, and use `--tag` so neither overwrites the other
  (run_table1.sbatch does exactly this as a job array).

NOTE: the dev ablation was run with proxy (TF-IDF/SVD) text embeddings in an
environment without torch, so it informs design choices only. This script,
run on the cluster with the real sentence-transformer embeddings, is the
source of any reported number.

USAGE:
  python plan_table1.py --n_episodes 5000 --seeds 0 1 2 42 123
  python plan_table1.py --norm episode --y_norm episode --tag original   # old pipeline
  python plan_table1.py --gate blend --tag gate_blend                    # pre-declared variant
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/planning_v3")
RESULTS_DIR = Path("results_planning_v3"); RESULTS_DIR.mkdir(exist_ok=True)

TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
VAL_DOMAIN = "gripper"  # matches the paper's stated protocol: "Gripper
                        # serves as a validation domain for early stopping."
                        # The original train_arc() never actually used it --
                        # it trained for a fixed episode count and evaluated
                        # whatever weights existed at the end, which on a
                        # noisy single-episode training signal (see the
                        # oscillating loss trace observed on the cluster) is
                        # close to sampling a random point on an unconverged
                        # trajectory. This is the fix for that.


# ============================================================================
# Data loading -- filters to valid==True EVERYWHERE. This is the fix for
# bug #4 (BFS-failure instances must never enter any correlation/regression).
# ============================================================================

def load_data():
    eps = json.loads((DATA_DIR / "episodes.json").read_text())
    X_surf = np.load(DATA_DIR / "X_surf.npy")
    X_fm = np.load(DATA_DIR / "X_fm.npy")
    y_nsteps = np.load(DATA_DIR / "y_nsteps.npy")
    y_valid = np.load(DATA_DIR / "y_valid.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)

    # HARD FILTER: only valid (BFS-solved) instances are usable labels.
    mask = y_valid
    print(f"Filtering to valid==True: {mask.sum()}/{len(mask)} instances "
          f"({100*mask.sum()/len(mask):.1f}%). Excluded instances had "
          f"n_steps=0 sentinel values and are NOT usable labels.")
    return X_surf[mask], X_fm[mask], y_nsteps[mask], task_types[mask]


# ============================================================================
# Baseline 1: Object-cardinality |O|
# ============================================================================

def eval_object_cardinality(X_surf, y_nsteps, task_types):
    results = {}
    for domain in TEST_DOMAINS:
        m = task_types == domain
        if m.sum() < 5:
            results[domain] = float("nan"); continue
        n_obj = X_surf[m, 0]  # x_s[0] is n_objects by construction
        rho, _ = stats.spearmanr(n_obj, y_nsteps[m])
        results[domain] = float(abs(rho))
    return results


# ============================================================================
# Baseline 2: LogME (You et al. 2021) on frozen FM embedding x_e
# ============================================================================

def logme_score(features, targets, max_iter=20):
    """Standard LogME: fits an empirical-Bayes linear-Gaussian model
    between frozen features and continuous targets, returns log evidence."""
    f = features.astype(np.float64)
    y = targets.astype(np.float64)
    n, d = f.shape
    if n <= d:
        u, s, vh = np.linalg.svd(f, full_matrices=False)
    else:
        u, s, vh = np.linalg.svd(f, full_matrices=False)
    s2 = s ** 2
    y_proj = u.T @ y
    alpha, beta = 1.0, 1.0
    for _ in range(max_iter):
        gamma = np.sum(s2 / (s2 * alpha / beta + alpha))
        m2 = np.sum((beta * s2 * y_proj ** 2) / (beta * s2 + alpha) ** 2)
        alpha_new = max(gamma / max(m2, 1e-12), 1e-12)
        res2 = np.sum((y - u @ ((beta * s2 * y_proj) / (beta * s2 + alpha))) ** 2)
        beta_new = max((n - gamma) / max(res2, 1e-12), 1e-12)
        if abs(alpha_new - alpha) < 1e-6 and abs(beta_new - beta) < 1e-6:
            alpha, beta = alpha_new, beta_new; break
        alpha, beta = alpha_new, beta_new
    evidence = (0.5 * d * np.log(alpha) + 0.5 * n * np.log(beta)
                - 0.5 * n * np.log(2 * np.pi)
                - 0.5 * np.sum(np.log(beta * s2 + alpha))
                - 0.5 * beta * np.sum((y - u @ ((beta * s2 * y_proj) / (beta * s2 + alpha))) ** 2)
                - 0.5 * alpha * np.sum(((beta * s2 * y_proj) / (beta * s2 + alpha)) ** 2))
    return float(evidence)


def eval_logme(X_fm, y_nsteps, task_types):
    """
    LogME as a difficulty-ranking baseline, using the SAME zero-shot
    protocol as ARC and ERM: fit on TRAIN domains, predict on TEST
    domains, correlate predictions against the test domain's own labels.

    BUG FIXED: this function previously fit Ridge(alpha=1.0) and called
    that "LogME" -- it never called logme_score() at all, so it was
    mathematically identical to plain ridge regression on x_e with a
    fixed, untuned regularization strength, not LogME. (Confirmed: this
    is why plan_step44_arc_ablations.py's independently-computed
    "Ridge (x_e)" row reproduced the old "LogME" row to four decimal
    places on a real cluster run -- they were the same computation.)
    Genuine LogME (You et al., 2021) is an empirical-Bayes procedure
    that selects (alpha, beta) by evidence maximization rather than
    fixing them; logme_score() above already implements that EM loop
    but its alpha/beta were previously discarded. This version reuses
    them: it refits the SAME ridge regression but with the evidence-
    optimal alpha/beta ratio in place of the arbitrary alpha=1.0.

    IMPORTANT: fitting and scoring on the SAME instances (as an earlier
    version of this function did) is a leakage bug -- with a 768-dim
    frozen embedding and only ~20-40 instances per test domain, a
    same-set ridge fit trivially overfits and produces a spuriously
    high in-sample correlation (observed: |rho| > 0.9) that has nothing
    to do with genuine transferability. Caught by direct testing before
    this script was used to produce any paper number.
    """
    train_mask = np.isin(task_types, TRAIN_DOMAINS)
    Xe_tr, y_tr = X_fm[train_mask], y_nsteps[train_mask]
    sc = StandardScaler().fit(Xe_tr)
    Xe_tr_n = sc.transform(Xe_tr)

    # Run the SAME EM loop logme_score() uses internally, but keep (alpha, beta) instead of
    # discarding them after computing the scalar evidence -- this is the actual LogME fit.
    f64, y64 = Xe_tr_n.astype(np.float64), y_tr.astype(np.float64)
    n, d = f64.shape
    u, s, vh = np.linalg.svd(f64, full_matrices=False)
    s2 = s ** 2
    y_proj = u.T @ y64
    alpha, beta = 1.0, 1.0
    for _ in range(20):
        gamma = np.sum(s2 / (s2 * alpha / beta + alpha))
        m2 = np.sum((beta * s2 * y_proj ** 2) / (beta * s2 + alpha) ** 2)
        alpha_new = max(gamma / max(m2, 1e-12), 1e-12)
        res2 = np.sum((y64 - u @ ((beta * s2 * y_proj) / (beta * s2 + alpha))) ** 2)
        beta_new = max((n - gamma) / max(res2, 1e-12), 1e-12)
        if abs(alpha_new - alpha) < 1e-6 and abs(beta_new - beta) < 1e-6:
            alpha, beta = alpha_new, beta_new; break
        alpha, beta = alpha_new, beta_new
    # LogME's posterior mean weight vector is equivalent to ridge regression with
    # regularization strength alpha/beta (same derivation as logme_score's own posterior).
    ridge = Ridge(alpha=alpha / beta).fit(Xe_tr_n, y_tr)

    results = {}
    for domain in TEST_DOMAINS:
        m = task_types == domain
        if m.sum() < 5:
            results[domain] = float("nan"); continue
        Xe_te_n = sc.transform(X_fm[m])
        pred = ridge.predict(Xe_te_n)
        rho, _ = stats.spearmanr(pred, y_nsteps[m])
        results[domain] = float(abs(rho))
    print(f"    [LogME] evidence-optimal alpha/beta = {alpha/beta:.4g}  (previously hardcoded to 1.0)")
    return results


def eval_hscore(X_fm, y_nsteps, task_types, n_bins=5):
    """
    H-score (Bao et al. 2019), using the SAME zero-shot protocol as
    ARC/ERM/LogME above: quantile-bin y_nsteps on TRAIN domains only,
    fit the discriminant direction on train, then project TEST-domain
    instances onto it and correlate with the test domain's own labels.
    Fitting and scoring on the same instances would have the identical
    leakage problem diagnosed in eval_logme above.
    """
    train_mask = np.isin(task_types, TRAIN_DOMAINS)
    Xe_tr, y_tr = X_fm[train_mask], y_nsteps[train_mask]
    sc = StandardScaler().fit(Xe_tr)
    Xe_tr_n = sc.transform(Xe_tr)

    bins = np.quantile(y_tr, np.linspace(0, 1, n_bins + 1))
    labels_tr = np.clip(np.digitize(y_tr, bins[1:-1]), 0, n_bins - 1)
    cov = np.cov(Xe_tr_n, rowvar=False) + 1e-6 * np.eye(Xe_tr_n.shape[1])
    cov_inv = np.linalg.pinv(cov)
    class_means = np.array([Xe_tr_n[labels_tr == c].mean(0) if (labels_tr == c).any()
                             else np.zeros(Xe_tr_n.shape[1]) for c in range(n_bins)])
    discriminant_direction = cov_inv @ class_means.mean(0)

    results = {}
    for domain in TEST_DOMAINS:
        m = task_types == domain
        if m.sum() < 5:
            results[domain] = float("nan"); continue
        Xe_te_n = sc.transform(X_fm[m])
        proj = Xe_te_n @ discriminant_direction
        rho, _ = stats.spearmanr(proj, y_nsteps[m])
        results[domain] = float(abs(rho))
    return results


# ============================================================================
# Baseline 4: ERM (non-episodic MLP), trained on train domains, zero-shot
# on test domains
# ============================================================================

class ERM_MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def eval_erm(X_surf, X_fm, y_nsteps, task_types, seed=42, n_epochs=300, y_norm="raw"):
    """y_norm='domain': regress each SOURCE domain's targets z-scored with that domain's own mean/std,
    i.e. the same target treatment ARC uses by default -- so ERM is not handicapped relative to ARC.
    Spearman is scale-free, so this only changes what the model is trained to fit."""
    torch.manual_seed(seed)
    train_mask = np.isin(task_types, TRAIN_DOMAINS)
    X_tr = np.hstack([X_surf[train_mask], X_fm[train_mask]])
    y_tr = y_nsteps[train_mask].astype(float).copy()
    if y_norm == "domain":
        tt_tr = task_types[train_mask]
        for d_ in TRAIN_DOMAINS:
            mm = tt_tr == d_
            y_tr[mm] = (y_tr[mm] - y_tr[mm].mean()) / (y_tr[mm].std() + 1e-6)
    sc = StandardScaler().fit(X_tr)
    X_tr_n = sc.transform(X_tr)

    model = ERM_MLP(X_tr_n.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.FloatTensor(X_tr_n); yt = torch.FloatTensor(y_tr)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(Xt), yt)
        loss.backward(); opt.step()

    results = {}
    model.eval()
    with torch.no_grad():
        for domain in TEST_DOMAINS:
            m = task_types == domain
            if m.sum() < 5:
                results[domain] = float("nan"); continue
            X_te = np.hstack([X_surf[m], X_fm[m]])
            X_te_n = sc.transform(X_te)
            pred = model(torch.FloatTensor(X_te_n)).numpy()
            rho, _ = stats.spearmanr(pred, y_nsteps[m])
            results[domain] = float(rho)  # SIGNED, matching the paper's ERM row
    return results


def eval_erm_multiseed(X_surf, X_fm, y_nsteps, task_types, seeds, n_epochs=300, y_norm="raw"):
    """ERM is also a trained model, not a closed-form statistic like
    Object-cardinality/LogME/H-score, so it is equally subject to the
    single-seed variance problem observed with ARC. Run across the same
    seed set and report per-domain mean +/- std, not one seed's draw."""
    per_seed = []
    for seed in seeds:
        r = eval_erm(X_surf, X_fm, y_nsteps, task_types, seed=seed, n_epochs=n_epochs, y_norm=y_norm)
        per_seed.append(r)
    domains = list(per_seed[0].keys())
    mean_r = {d: float(np.nanmean([r[d] for r in per_seed])) for d in domains}
    std_r = {d: float(np.nanstd([r[d] for r in per_seed])) for d in domains}
    return mean_r, std_r, per_seed


# ============================================================================
# ARC: the actual episodic model, trained fresh here, zero-shot evaluated.
# No hardcoded constants anywhere in this function.
# ============================================================================

class PooledPrep:
    """
    ONE feature normaliser + residualiser, fit once on pooled TRAIN-domain
    rows. Used identically for (a) training episodes, (b) Gripper validation
    and (c) test-domain scoring, so the model never sees features on a
    different scale at test time than it was trained on.

    Fit on source-domain data only (no target-domain features or labels),
    matching the paper's stated preprocessing rule.
    """

    def __init__(self, Xs, Xf, n_comp=20):
        self.ss = StandardScaler().fit(Xs)
        self.se = StandardScaler().fit(Xf)
        a, b = self.ss.transform(Xs), self.se.transform(Xf)
        self.res = Pipeline([("pca", PCA(n_components=min(n_comp, a.shape[1]))),
                             ("ridge", Ridge(alpha=1.0))]).fit(a, b)

    def __call__(self, Xs, Xf):
        a, b = self.ss.transform(Xs), self.se.transform(Xf)
        return (a.astype(np.float32), b.astype(np.float32),
                (b - self.res.predict(a)).astype(np.float32))


class PlanningGURU(nn.Module):
    def __init__(self, surf_dim, fm_dim, d_model=128, gate_mode="none"):
        super().__init__()
        self.query_enc = nn.Sequential(nn.Linear(fm_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                       nn.Dropout(0.1), nn.Linear(512, d_model), nn.LayerNorm(d_model))
        self.key_enc = nn.Sequential(nn.Linear(surf_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(512, d_model), nn.LayerNorm(d_model))
        self.value_proj = nn.Sequential(nn.Linear(surf_dim + fm_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        fused_in = surf_dim + d_model + fm_dim
        self.fusion = nn.Sequential(nn.Linear(fused_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05))
        self.head_reg = nn.Linear(128, 1)
        self.scale = d_model ** -0.5
        assert gate_mode in ("none", "blend", "zeff"), gate_mode
        self.gate_mode = gate_mode
        # Gate modules are created ONLY when requested, and AFTER every
        # pre-existing layer, so `--gate none` consumes exactly the same
        # random-number stream as the earlier script (earlier cluster runs
        # stay reproducible under `--norm episode --y_norm episode`).
        if gate_mode != "none":
            self.gate_net = nn.Sequential(nn.Linear(surf_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        if gate_mode == "blend":
            self.lin_simple = nn.Linear(surf_dim, 1)  # y_simple = w^T x_s^q

    def forward(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V):
        q = self.query_enc(q_fm.unsqueeze(0) if q_fm.dim() == 1 else q_fm)
        k = self.key_enc(S_surf)
        v = self.value_proj(S_V)
        alpha = torch.softmax(torch.matmul(q, k.T) * self.scale, dim=-1)
        z = torch.matmul(alpha, v)
        if q_fm.dim() == 1: z = z.squeeze(0); alpha = alpha.squeeze(0)
        c = None
        if self.gate_mode != "none":
            c = torch.sigmoid(self.gate_net(q_surf))          # (..., 1), from x_s^q only
            if self.gate_mode == "zeff":
                z = (1.0 - c) * z
        fused = torch.cat([q_surf, z, q_resid], dim=-1)
        feats = self.fusion(fused)
        y = self.head_reg(feats).squeeze(-1)
        if self.gate_mode == "blend":
            cs = c.squeeze(-1)
            y = cs * self.lin_simple(q_surf).squeeze(-1) + (1.0 - cs) * y
        return y, feats, alpha


class MetaSampler:
    def __init__(self, domain_list, X_surf, X_fm, y_nsteps, task_types, seed=42,
                 prep=None, y_norm="episode"):
        self.rng = np.random.default_rng(seed)
        self.prep, self.y_norm = prep, y_norm
        self.domain_data = {}
        for d in domain_list:
            m = task_types == d
            if m.sum() < 20:
                continue
            rec = {"X_surf": X_surf[m], "X_fm": X_fm[m], "y": y_nsteps[m],
                   "mu": float(y_nsteps[m].mean()), "sd": float(y_nsteps[m].std() + 1e-6)}
            if prep is not None:
                rec["A"], rec["B"], rec["R"] = prep(X_surf[m], X_fm[m])  # normalised ONCE, pooled stats
            self.domain_data[d] = rec
        self.valid_tasks = list(self.domain_data.keys())

    def sample_episode(self, max_support=60):
        domain = self.rng.choice(self.valid_tasks)
        d = self.domain_data[domain]
        N = len(d["y"])
        n_s = min(max_support, max(10, int(N * 0.7)))
        n_q = min(30, N - n_s)
        if n_q < 3:
            n_s = max(5, N - 3); n_q = N - n_s
        idx = self.rng.permutation(N)
        s_idx, q_idx = idx[:n_s], idx[n_s:n_s + n_q]
        y_q = d["y"][q_idx]

        if self.prep is not None:
            # pooled normalisation: same transform as validation/test time
            Xs_s_n, Xs_q_n = d["A"][s_idx], d["A"][q_idx]
            Xe_s_n, Xe_q_n = d["B"][s_idx], d["B"][q_idx]
            Xr_s, Xr_q = d["R"][s_idx], d["R"][q_idx]
        else:
            # original behaviour: scaler + residualiser fit on THIS episode's support set
            Xs_s, Xs_q = d["X_surf"][s_idx], d["X_surf"][q_idx]
            Xe_s, Xe_q = d["X_fm"][s_idx], d["X_fm"][q_idx]
            sc_s = StandardScaler().fit(Xs_s); sc_e = StandardScaler().fit(Xe_s)
            Xs_s_n, Xs_q_n = sc_s.transform(Xs_s), sc_s.transform(Xs_q)
            Xe_s_n, Xe_q_n = sc_e.transform(Xe_s), sc_e.transform(Xe_q)
            n_comp = max(2, min(20, n_s // 5, Xs_s_n.shape[1]))
            pred = Pipeline([("pca", PCA(n_components=n_comp)), ("ridge", Ridge(alpha=1.0))])
            pred.fit(Xs_s_n, Xe_s_n)
            Xr_s = Xe_s_n - pred.predict(Xs_s_n)
            Xr_q = Xe_q_n - pred.predict(Xs_q_n)

        S_V = np.hstack([Xs_s_n, Xr_s])
        t = lambda x: torch.FloatTensor(x)
        ep = {"S_surf": t(Xs_s_n), "S_fm": t(Xe_s_n), "S_V": t(S_V),
              "Q_surf": t(Xs_q_n), "Q_fm": t(Xe_q_n), "Q_resid": t(Xr_q), "Y": t(y_q)}
        # Domain-level target z-score (mean/std from ALL labelled instances of
        # this source domain). None => train_arc falls back to the original
        # per-episode z-score.
        ep["Y_norm"] = t((y_q - d["mu"]) / d["sd"]) if self.y_norm == "domain" else None
        return ep


def train_arc(X_surf, X_fm, y_nsteps, task_types, n_episodes, seed, use_contrastive=False,
              lambda_contrast=0.1, verbose=True, val_every=200, patience=10,
              norm="pooled", y_norm="domain", gate="none"):
    """
    Trains ARC with Gripper-based validation, best-checkpoint selection,
    and patience-based early stopping -- matching the paper's own stated
    protocol ("Gripper serves as a validation domain for early stopping"),
    which the original version of this function did not actually
    implement: it trained for a fixed n_episodes and returned whatever
    weights existed after the LAST episode.

    That mattered in practice: with each training step being a single,
    small, highly variable episode (one domain, <=30 query instances,
    per-episode target normalization), the loss trace oscillates rather
    than converging smoothly, so the final episode's weights are close to
    an arbitrary sample from a non-converged trajectory. Two identically
    configured runs (seed=42 vs seed=0) previously produced ARC means of
    0.468 and 0.061 on the same data -- consistent with exactly this
    problem, not with two genuine independent estimates of ARC's ability.

    norm / y_norm / gate: design switches, documented in the module
    docstring. Defaults are the dev-set-supported settings; the original
    pipeline is `norm="episode", y_norm="episode", gate="none"`.

    val_every: run a Gripper check every this many episodes.
    patience: stop early if `patience` consecutive validation checks pass
              with no improvement over the best score seen so far.
    """
    torch.manual_seed(seed)
    assert norm in ("pooled", "episode") and y_norm in ("domain", "episode"), (norm, y_norm)
    prep = None
    if norm == "pooled":
        tr_mask = np.isin(task_types, TRAIN_DOMAINS)
        prep = PooledPrep(X_surf[tr_mask], X_fm[tr_mask])   # source domains only
    sampler = MetaSampler(TRAIN_DOMAINS, X_surf, X_fm, y_nsteps, task_types, seed=seed,
                          prep=prep, y_norm=y_norm)
    model = PlanningGURU(X_surf.shape[1], X_fm.shape[1], gate_mode=gate)
    model.prep = prep   # plain attribute: eval_arc() must use the SAME normaliser as training
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)

    loss_history = []
    window = max(10, n_episodes // 100)

    best_val_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_episode = 0
    checks_since_improvement = 0
    stopped_early_at = None

    # Build the Gripper-scoring support set ONCE, outside the training
    # loop. It's deterministic given `seed`, so rebuilding it on every
    # validation check (as an earlier version of this function did, via
    # a function that reseeded identically each time) was ~25 redundant
    # sklearn fits per seed for zero benefit.
    val_support = _build_arc_support_set(X_surf, X_fm, task_types, seed=seed, prep=prep)

    for ep in range(n_episodes):
        d = sampler.sample_episode()
        pred, feats, alpha = model(d["Q_surf"], d["Q_fm"], d["Q_resid"], d["S_surf"], d["S_fm"], d["S_V"])
        if d["Y_norm"] is not None:
            y_target = d["Y_norm"]                      # domain-level z-score
        else:
            y_target = (d["Y"] - d["Y"].mean()) / (d["Y"].std() + 1e-6)   # original per-episode z-score
        loss = F.mse_loss(pred, y_target)
        if use_contrastive:
            q_proj = model.query_enc(d["Q_fm"])
            q_norm = F.normalize(q_proj, dim=-1)
            sim = q_norm @ q_norm.T / 0.1
            y_diff = (d["Y"].unsqueeze(0) - d["Y"].unsqueeze(1)).abs()
            pos_mask = (y_diff < y_diff.median()).float()
            log_prob = F.log_softmax(sim, dim=-1)
            contrast_loss = -(log_prob * pos_mask).sum(-1).mean()
            loss = loss + lambda_contrast * contrast_loss
        ent = -(alpha * (alpha + 1e-8).log()).sum(-1).mean()
        loss = loss - 0.05 * ent
        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        loss_history.append(float(loss.item()))
        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            recent = loss_history[-window:]
            print(f"    ep {ep+1}/{n_episodes}  loss(avg over last {len(recent)})="
                  f"{np.mean(recent):.4f}  (min={np.min(recent):.4f} max={np.max(recent):.4f})")

        if (ep + 1) % val_every == 0:
            val_score = _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types,
                                                     VAL_DOMAIN, *val_support)
            if val_score is None:
                continue  # can't validate this check; don't count against patience
            improved = val_score > best_val_score
            if improved:
                best_val_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_episode = ep + 1
                checks_since_improvement = 0
            else:
                checks_since_improvement += 1
            if verbose:
                flag = " (new best)" if improved else f" ({checks_since_improvement}/{patience} since best)"
                print(f"    [val] ep {ep+1}/{n_episodes}  Gripper |rho|={val_score:.4f}{flag}")
            if checks_since_improvement >= patience:
                stopped_early_at = ep + 1
                if verbose:
                    print(f"    Early stopping: no Gripper improvement in {patience} checks "
                          f"(best={best_val_score:.4f} at episode {best_episode})")
                break

    model.load_state_dict(best_state)
    if verbose:
        status = f"stopped early at episode {stopped_early_at}" if stopped_early_at else f"ran all {n_episodes} episodes"
        print(f"    Training done ({status}). Restored best checkpoint: "
              f"episode {best_episode}, Gripper |rho|={best_val_score:.4f}")
    return model


def _build_arc_support_set(X_surf, X_fm, task_types, n_support=60, seed=42, prep=None):
    """
    Builds the support-set tensors ONCE for a given seed and returns
    (transform, S_surf, S_fm, S_V), where transform(Xs, Xf) -> (a, b, r) maps
    raw features of ANY domain into the model's input space.

      prep given   -> transform is the pooled train-domain normaliser (the
                      same one used for training episodes).
      prep is None -> original behaviour: scaler/residualiser fit on this
                      60-instance support sample (differs from training-time
                      per-episode normalisation -- the mismatch discussed in
                      the module docstring).

    Call once per seed and reuse across every validation check and the final
    test-domain evaluation.
    """
    train_mask = np.isin(task_types, TRAIN_DOMAINS)
    rng = np.random.default_rng(seed)
    idx_tr = np.where(train_mask)[0]
    sup_idx = rng.choice(idx_tr, min(n_support, len(idx_tr)), replace=False)
    Xs_sup, Xe_sup = X_surf[sup_idx], X_fm[sup_idx]

    if prep is not None:
        transform = prep
    else:
        sc_s = StandardScaler().fit(Xs_sup); sc_e = StandardScaler().fit(Xe_sup)
        Xs_sup_n = sc_s.transform(Xs_sup)
        n_comp = max(2, min(20, len(sup_idx) // 5, Xs_sup_n.shape[1]))
        pred_r = Pipeline([("pca", PCA(n_components=n_comp)), ("ridge", Ridge(alpha=1.0))])
        pred_r.fit(Xs_sup_n, sc_e.transform(Xe_sup))

        def transform(Xs, Xf):
            a, b = sc_s.transform(Xs), sc_e.transform(Xf)
            return a, b, b - pred_r.predict(a)

    a, b, r = transform(Xs_sup, Xe_sup)
    S_surf = torch.FloatTensor(a); S_fm = torch.FloatTensor(b)
    S_V = torch.FloatTensor(np.hstack([a, r]))
    return transform, S_surf, S_fm, S_V


def _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, domain,
                                 transform, S_surf, S_fm, S_V):
    """Score one domain zero-shot using an ALREADY-BUILT support set."""
    m = task_types == domain
    if m.sum() < 5:
        return None
    a, b, r = transform(X_surf[m], X_fm[m])
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(torch.FloatTensor(a), torch.FloatTensor(b),
                           torch.FloatTensor(r), S_surf, S_fm, S_V)
        scores = pred.numpy()
    model.train()
    rho, _ = stats.spearmanr(scores, y_nsteps[m])
    if np.isnan(rho):
        return None
    return float(abs(rho))


def _score_domain_with_arc(model, X_surf, X_fm, y_nsteps, task_types, domain,
                            n_support=60, seed=42, prep=None):
    """One-shot convenience wrapper (build support set + score)."""
    support = _build_arc_support_set(X_surf, X_fm, task_types, n_support=n_support,
                                     seed=seed, prep=prep)
    return _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types, domain, *support)


def eval_arc(model, X_surf, X_fm, y_nsteps, task_types, n_support=60, seed=42):
    # Use the SAME normaliser the model was trained with (attached by train_arc).
    prep = getattr(model, "prep", None)
    support = _build_arc_support_set(X_surf, X_fm, task_types, n_support=n_support,
                                     seed=seed, prep=prep)
    results = {}
    for domain in TEST_DOMAINS:
        score = _score_domain_given_support(model, X_surf, X_fm, y_nsteps, task_types,
                                            domain, *support)
        results[domain] = score if score is not None else float("nan")
    return results


def train_and_eval_arc_multiseed(X_surf, X_fm, y_nsteps, task_types, n_episodes, seeds,
                                  use_contrastive=False, verbose=True, val_every=200, patience=10,
                                  norm="pooled", y_norm="domain", gate="none"):
    """
    Runs ARC training+eval independently across multiple seeds and reports
    mean +/- std per domain, instead of a single seed's number.

    This is required, not optional: an earlier single-seed run (seed=42)
    gave ARC mean=0.468 on this exact corrected dataset, while a
    same-architecture, same-hyperparameter run at seed=0 (from the
    InfoNCE ablation, without-contrastive condition) gave mean=0.061 --
    a swing far too large to be ordinary domain-sampling noise. Reporting
    either number alone as "ARC's score" would be exactly the kind of
    unverifiable, seed-cherry-pickable claim this whole revision effort
    exists to eliminate. Gripper-based early stopping (see train_arc)
    addresses the likely ROOT CAUSE of that swing; multi-seed averaging
    remains in place as a second, independent safeguard on top of it.
    """
    per_seed_results = []
    for seed in seeds:
        print(f"  --- ARC seed {seed} ---")
        model = train_arc(X_surf, X_fm, y_nsteps, task_types, n_episodes, seed,
                          use_contrastive=use_contrastive, verbose=verbose,
                          val_every=val_every, patience=patience,
                          norm=norm, y_norm=y_norm, gate=gate)
        r = eval_arc(model, X_surf, X_fm, y_nsteps, task_types, seed=seed)
        per_seed_results.append(r)
        print(f"    seed {seed} result: {r}  mean={np.nanmean(list(r.values())):.4f}")

    domains = list(per_seed_results[0].keys())
    mean_r = {d: float(np.nanmean([r[d] for r in per_seed_results])) for d in domains}
    std_r = {d: float(np.nanstd([r[d] for r in per_seed_results])) for d in domains}
    return mean_r, std_r, per_seed_results


def eval_ridge_xs(X_surf, y_nsteps, task_types, ytype="raw"):
    """
    Ridge regression on the structural features x_s ONLY, trained on the
    source domains and scored zero-shot on the test domains (|Spearman rho|).
    Deterministic. Included because the dev-set ablation showed that this
    trivial learner matches the best ARC variant on held-out source domains
    (LOTO 0.529 vs 0.523): any claim that ARC's retrieval/attention adds
    value has to beat this row, not just |O|.
    """
    tr = np.isin(task_types, TRAIN_DOMAINS)
    sc = StandardScaler().fit(X_surf[tr])
    yt = y_nsteps.astype(float).copy()
    if ytype == "domain":   # per-source-domain z-score, same target treatment as ARC's default
        for d_ in TRAIN_DOMAINS:
            mm = task_types == d_
            yt[mm] = (yt[mm] - yt[mm].mean()) / (yt[mm].std() + 1e-6)
    model = Ridge(alpha=1.0).fit(sc.transform(X_surf[tr]), yt[tr])
    results = {}
    for domain in TEST_DOMAINS:
        m = task_types == domain
        if m.sum() < 5:
            results[domain] = float("nan"); continue
        rho, _ = stats.spearmanr(model.predict(sc.transform(X_surf[m])), y_nsteps[m])
        results[domain] = float(abs(rho))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42, 123],
                    help="ARC and ERM are trained models and must be run "
                         "across multiple seeds, not one. A single-seed "
                         "run previously gave ARC mean=0.468 at seed=42 "
                         "and mean=0.061 at seed=0 on the same corrected "
                         "data -- a swing too large to report as 'the' "
                         "ARC score. Default is 5 seeds; use fewer only "
                         "for a quick smoke test, and say so when you do.")
    p.add_argument("--use_contrastive", action="store_true",
                    help="Only for the InfoNCE ablation script; Table 1's "
                         "canonical ARC row uses whichever setting the "
                         "ablation determines should be kept.")
    p.add_argument("--val_every", type=int, default=200,
                    help="Check Gripper validation score every this many "
                         "training episodes; keep the best-scoring "
                         "checkpoint instead of the final episode's weights.")
    p.add_argument("--patience", type=int, default=10,
                    help="Stop training early if this many consecutive "
                         "Gripper validation checks pass with no "
                         "improvement over the best score seen so far.")
    p.add_argument("--norm", choices=["pooled", "episode"], default="pooled",
                    help="Feature normalisation: one pooled train-domain scaler for "
                         "training AND test (pooled), or the original per-episode "
                         "scaler at training with a different one at test (episode).")
    p.add_argument("--y_norm", choices=["domain", "episode"], default="domain",
                    help="Target z-scoring: per source domain (domain) or per "
                         "30-instance query set (episode, original).")
    p.add_argument("--gate", choices=["none", "blend", "zeff"], default="none",
                    help="Complexity-aware gate variant. Dev set: none vs blend was a "
                         "wash, so report BOTH; do not pick by test score.")
    p.add_argument("--tag", default="",
                    help="Suffix for the results file so variants never overwrite "
                         "each other (table1_final_<tag>.json).")
    args = p.parse_args()

    if len(args.seeds) < 3:
        print(f"WARNING: only {len(args.seeds)} seed(s) requested. Mean/std "
              f"from fewer than 3 seeds should be treated as a smoke test, "
              f"not a reportable Table 1 number.\n")

    X_surf, X_fm, y_nsteps, task_types = load_data()

    print("\n=== Baseline: Object-cardinality |O| (deterministic, no seed dependence) ===")
    obj_results = eval_object_cardinality(X_surf, y_nsteps, task_types)
    print(obj_results)

    print("\n=== Baseline: LogME (deterministic given data, no seed dependence) ===")
    logme_results = eval_logme(X_fm, y_nsteps, task_types)
    print(logme_results)

    print("\n=== Baseline: H-score (deterministic given data, no seed dependence) ===")
    hscore_results = eval_hscore(X_fm, y_nsteps, task_types)
    print(hscore_results)

    print(f"\n=== Baseline: ERM (non-episodic MLP), {len(args.seeds)} seeds ===")
    erm_mean, erm_std, erm_per_seed = eval_erm_multiseed(X_surf, X_fm, y_nsteps, task_types, args.seeds)
    print(f"  mean: {erm_mean}")
    print(f"  std:  {erm_std}")

    print(f"\n=== Baseline: ERM with domain-z targets (same target treatment as ARC), {len(args.seeds)} seeds ===")
    ermz_mean, ermz_std, ermz_per_seed = eval_erm_multiseed(X_surf, X_fm, y_nsteps, task_types, args.seeds, y_norm="domain")
    print(f"  mean: {ermz_mean}")
    print(f"  std:  {ermz_std}")

    print("\n=== Baseline: Ridge on structural features x_s only (deterministic) ===")
    ridge_results = eval_ridge_xs(X_surf, y_nsteps, task_types)
    print(ridge_results)
    ridgez_results = eval_ridge_xs(X_surf, y_nsteps, task_types, ytype="domain")
    print("Ridge(x_s) with domain-z targets:", ridgez_results)

    print(f"\n=== ARC (episodic, contrastive={args.use_contrastive}, norm={args.norm}, "
          f"y_norm={args.y_norm}, gate={args.gate}), {len(args.seeds)} seeds, "
          f"Gripper early stopping (val_every={args.val_every}, patience={args.patience}) ===")
    arc_mean, arc_std, arc_per_seed = train_and_eval_arc_multiseed(
        X_surf, X_fm, y_nsteps, task_types, args.n_episodes, args.seeds,
        use_contrastive=args.use_contrastive, val_every=args.val_every, patience=args.patience,
        norm=args.norm, y_norm=args.y_norm, gate=args.gate)

    def mean_of(d):
        vals = [v for v in d.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 86)
    print(f"{'Method':<20} {'BW':>14} {'LOG':>14} {'MBW':>14} {'Mean':>10}")
    print("-" * 86)
    # Deterministic baselines: no +/- to report
    for name, r in [("Object-card.", obj_results), ("Ridge(x_s)", ridge_results), ("Ridge(x_s,z)", ridgez_results),
                    ("LogME", logme_results), ("H-score", hscore_results)]:
        print(f"{name:<20} {r['blocksworld']:>14.3f} {r['logistics']:>14.3f} "
              f"{r['mystery_blocksworld']:>14.3f} {mean_of(r):>10.3f}")
    # Trained models: mean +/- std across seeds
    for name, mean_r, std_r in [("ERM", erm_mean, erm_std), ("ERM (domain-z)", ermz_mean, ermz_std), (f"ARC[{args.norm}/{args.y_norm}/{args.gate}]", arc_mean, arc_std)]:
        def cell(d):
            return f"{mean_r[d]:.3f}+/-{std_r[d]:.3f}"
        print(f"{name:<20} {cell('blocksworld'):>14} {cell('logistics'):>14} "
              f"{cell('mystery_blocksworld'):>14} {mean_of(mean_r):>10.3f}")

    output = {
        "object_cardinality": obj_results, "ridge_xs": ridge_results, "ridge_xs_domain_z": ridgez_results,
        "erm_domain_z_mean": ermz_mean, "erm_domain_z_std": ermz_std, "erm_domain_z_per_seed": ermz_per_seed,
        "logme": logme_results, "hscore": hscore_results,
        "erm_mean": erm_mean, "erm_std": erm_std, "erm_per_seed": erm_per_seed,
        "arc_mean": arc_mean, "arc_std": arc_std, "arc_per_seed": arc_per_seed,
        "arc_used_contrastive": args.use_contrastive,
        "arc_config": {"norm": args.norm, "y_norm": args.y_norm, "gate": args.gate,
                       "val_every": args.val_every, "patience": args.patience},
        "n_episodes": args.n_episodes, "seeds": args.seeds,
        "n_valid_instances": int(len(y_nsteps)),
    }
    out_path = RESULTS_DIR / (f"table1_final_{args.tag}.json" if args.tag else "table1_final.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved -> {out_path}")
    print("\nThis JSON, and ONLY this JSON, is the source for Table 1. Do not")
    print("hand-copy these numbers into any other script's hardcoded dict.")
    print(f"\nARC per-seed results (inspect for outliers before reporting the mean):")
    for seed, r in zip(args.seeds, arc_per_seed):
        print(f"  seed {seed}: {r}  mean={np.nanmean(list(r.values())):.4f}")


if __name__ == "__main__":
    main()
