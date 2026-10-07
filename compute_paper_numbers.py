"""
compute_paper_numbers.py -- recomputes, from the CORRECTED v3 data (real BFS labels, no leak, failed searches excluded),
every paper number that does not need real sentence-transformer embeddings, trained ARC checkpoints or a GPU.

  Run from the folder that contains data/planning_v3 (from plan_step1_data_prep_v3.py):
      PYTHONHASHSEED=0 python compute_paper_numbers.py
  Output: paper_numbers.json  +  LaTeX-ready lines printed to stdout.

Only instances with valid==True (solved by BFS within the budget) are used anywhere. Test domains are scored with
pre-declared baselines only (nothing here is tuned on them).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data/planning_v3")
eps = json.loads((D / "episodes.json").read_text())
Xs_all = np.load(D / "X_surf.npy"); y_all = np.load(D / "y_nsteps.npy").astype(float)
v_all = np.load(D / "y_valid.npy"); tt_all = np.load(D / "task_types.npy", allow_pickle=True)
TR = ["depot", "rovers", "satellite"]; TE = ["blocksworld", "logistics", "mystery_blocksworld"]; VAL = "gripper"
SEVEN = ["blocksworld", "depot", "gripper", "logistics", "mystery_blocksworld", "rovers", "satellite"]
NAME = {"blocksworld": "BW", "logistics": "LOG", "mystery_blocksworld": "MBW"}

# per-instance structural scalars used by the cheap surrogates
n_obj = np.array([e["n_objects"] for e in eps], float)
n_goal = np.array([len(e["goal_facts"]) for e in eps], float)
n_init = np.array([len(e["init_facts"]) for e in eps], float)
m = v_all
Xs, y, tt = Xs_all[m], y_all[m], tt_all[m]
n_obj, n_goal, n_init = n_obj[m], n_goal[m], n_init[m]
tt = np.array([str(t) for t in tt])
OUT = {}

def rho(a, b):
    r = stats.spearmanr(a, b)[0]
    return float(abs(r)) if not np.isnan(r) else float("nan")

def fisher(r, n, z=1.96):
    r = min(max(r, -0.9999), 0.9999); zz = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return float(np.tanh(zz - z * se)), float(np.tanh(zz + z * se))

# ---------------------------------------------------------------- dataset ----
print("== dataset (valid = BFS-solved within T=12, N=5000) ==")
OUT["dataset"] = {}
for d in SEVEN:
    k = tt == d; nt = int((tt_all == d).sum())
    OUT["dataset"][d] = dict(total=nt, valid=int(k.sum()), valid_frac=float(k.sum() / nt),
                             obj_min=int(n_obj[k].min()), obj_max=int(n_obj[k].max()),
                             nstar_min=int(y[k].min()), nstar_median=float(np.median(y[k])), nstar_max=int(y[k].max()))
    o = OUT["dataset"][d]
    print(f"{d:22s} total {nt}  valid {o['valid']:3d} ({100*o['valid_frac']:.1f}%)  |O| {o['obj_min']}-{o['obj_max']}  n* {o['nstar_min']}/{o['nstar_median']:.0f}/{o['nstar_max']}")

# ------------------------------------------------- Table 1 deterministic rows ----
tr = np.isin(tt, TR); sc = StandardScaler().fit(Xs[tr]); A = sc.transform(Xs)
def ridge_pred(ytype="raw", alpha=1.0, train=TR):
    trm = np.isin(tt, train); s = StandardScaler().fit(Xs[trm]); AA = s.transform(Xs); yt = y.copy()
    if ytype == "domain":
        for d in train: k = tt == d; yt[k] = (y[k] - y[k].mean()) / (y[k].std() + 1e-6)
    r = Ridge(alpha=alpha).fit(AA[trm], yt[trm]); return r.predict(AA)
P_ridge, P_ridgez = ridge_pred("raw"), ridge_pred("domain")
rows = {"Object cardinality": n_obj, "Ridge(x_s)": P_ridge, "Ridge(x_s, domain-z target)": P_ridgez}
print("\n== Table 1 deterministic rows (|rho|, test domains) ==")
OUT["table1"] = {}
for name, sc_ in rows.items():
    r = {d: rho(sc_[tt == d], y[tt == d]) for d in TE}; r["mean"] = float(np.mean([r[d] for d in TE])); OUT["table1"][name] = r
    print(f"{name:30s} BW {r['blocksworld']:.3f} LOG {r['logistics']:.3f} MBW {r['mystery_blocksworld']:.3f} mean {r['mean']:.3f}")

# ----------------------------------------- Fisher CIs + paired bootstrap ----
print("\n== 95% Fisher CIs (N = valid instances) ==")
OUT["fisher"] = {}
for name in ["Object cardinality", "Ridge(x_s)"]:
    OUT["fisher"][name] = {}
    for d in TE:
        n = int((tt == d).sum()); r = OUT["table1"][name][d]; lo, hi = fisher(r, n)
        OUT["fisher"][name][d] = dict(rho=r, lo=lo, hi=hi, n=n)
        print(f"{name:20s} {NAME[d]:4s} N={n:3d} rho={r:.3f} CI [{lo:.3f}, {hi:.3f}]")
rng = np.random.default_rng(0); print("\n== paired bootstrap, Delta = |rho_Ridge| - |rho_|O||, 2000 resamples ==")
OUT["boot_ridge_minus_obj"] = {}
for d in TE:
    k = np.where(tt == d)[0]; ds = []
    for _ in range(2000):
        i = rng.choice(k, len(k), replace=True)
        a, b = rho(P_ridge[i], y[i]), rho(n_obj[i], y[i])
        if not (np.isnan(a) or np.isnan(b)): ds.append(a - b)
    lo, hi = np.percentile(ds, [2.5, 97.5]); pt = OUT["table1"]["Ridge(x_s)"][d] - OUT["table1"]["Object cardinality"][d]
    OUT["boot_ridge_minus_obj"][d] = dict(delta=float(pt), lo=float(lo), hi=float(hi))
    print(f"{NAME[d]:4s} delta {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

# ------------------------------------------------------ learned baselines ----
print("\n== learned baselines on x_s only (trained on source domains, pooled raw target) ==")
OUT["learned"] = {}
def fit_eval(model, name):
    model.fit(A[tr], y[tr]); p = model.predict(A)
    r = {d: rho(p[tt == d], y[tt == d]) for d in TE}; r["mean"] = float(np.mean([r[d] for d in TE])); OUT["learned"][name] = r
    print(f"{name:26s} BW {r['blocksworld']:.3f} LOG {r['logistics']:.3f} MBW {r['mystery_blocksworld']:.3f} mean {r['mean']:.3f}")
fit_eval(RandomForestRegressor(300, random_state=0, n_jobs=1), "Random Forest (x_s)")
fit_eval(SVR(kernel="rbf"), "SVR-RBF (x_s)")
fit_eval(HistGradientBoostingRegressor(random_state=0), "Grad. boosting (x_s)")
fit_eval(Ridge(alpha=1.0), "Ridge (x_s)")
try:
    import xgboost as xgb
    fit_eval(xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1), "XGBoost (x_s)")
    # LambdaMART-style ranker: groups = source domains, graded relevance = within-domain decile of n*
    order = np.argsort(tt[tr], kind="stable"); Atr, ytr, ttr = A[tr][order], y[tr][order], tt[tr][order]
    rel = np.zeros(len(ytr), int)
    for d in TR:
        k = ttr == d; rel[k] = np.clip((stats.rankdata(ytr[k]) - 1) / k.sum() * 10, 0, 9).astype(int)
    groups = [int((ttr == d).sum()) for d in TR]  # contiguous after stable sort by name order below
    names_sorted = sorted(TR); groups = [int((ttr == d).sum()) for d in names_sorted]
    rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1)
    rk.fit(Atr, rel, group=groups); p = rk.predict(A)
    r = {d: rho(p[tt == d], y[tt == d]) for d in TE}; r["mean"] = float(np.mean([r[d] for d in TE])); OUT["learned"]["LambdaMART-style (x_s)"] = r
    print(f"{'LambdaMART-style (x_s)':26s} BW {r['blocksworld']:.3f} LOG {r['logistics']:.3f} MBW {r['mystery_blocksworld']:.3f} mean {r['mean']:.3f}")
except Exception as e:
    print("xgboost rows skipped:", e)

# ------------------------------------------------------- cheap surrogates ----
print("\n== cheap structural surrogates (|rho|, test domains) ==")
sur = {"Goal count": n_goal, "Initial-state size": n_init, "Initial-state density (n_init/|O|^2)": n_init / n_obj ** 2,
       "Object-goal ratio (|O|/n_goal)": n_obj / np.maximum(n_goal, 1), "Object cardinality": n_obj}
OUT["cheap"] = {}
for name, s_ in sur.items():
    r = {d: rho(s_[tt == d], y[tt == d]) for d in TE}; r["mean"] = float(np.mean([r[d] for d in TE])); OUT["cheap"][name] = r
    print(f"{name:38s} BW {r['blocksworld']:.3f} LOG {r['logistics']:.3f} MBW {r['mystery_blocksworld']:.3f} mean {r['mean']:.3f}")

# ------------------------------------------------------- within-size ----
print("\n== within object-count level (generator levels B1..B5): |rho| of Ridge(x_s) vs n*; |O| is constant => 0 ==")
OUT["within"] = {}
for d in TE:
    k = tt == d; levels = sorted(set(n_obj[k])); row = []
    for lv in levels:
        kk = k & (n_obj == lv); nn = int(kk.sum()); nd = len(set(y[kk]))
        r = rho(P_ridge[kk], y[kk]) if (nn >= 5 and nd > 1) else float("nan")
        row.append(dict(level=float(lv), n=nn, n_distinct_nstar=nd, rho=r))
    OUT["within"][d] = row
    print(f"{NAME[d]:4s} " + "  ".join(f"|O|={int(r['level'])}: N={r['n']:3d} #n*={r['n_distinct_nstar']} rho={'  nan' if np.isnan(r['rho']) else format(r['rho'],'.2f')}" for r in row))

# ------------------------------------------------------- leave-one-domain-out ----
print("\n== leave-one-domain-out over the seven domains: Ridge(x_s) vs |O| ==")
OUT["loo"] = {}
for h in SEVEN:
    trn = [d for d in SEVEN if d != h]; p = ridge_pred("raw", train=trn); k = tt == h
    OUT["loo"][h] = dict(ridge=rho(p[k], y[k]), obj=rho(n_obj[k], y[k]))
    print(f"{h:22s} Ridge {OUT['loo'][h]['ridge']:.3f}   |O| {OUT['loo'][h]['obj']:.3f}   delta {OUT['loo'][h]['ridge']-OUT['loo'][h]['obj']:+.3f}")
rr = np.array([OUT["loo"][h]["ridge"] for h in SEVEN]); oo = np.array([OUT["loo"][h]["obj"] for h in SEVEN])
OUT["loo_summary"] = dict(ridge_mean=float(rr.mean()), ridge_std=float(rr.std()), obj_mean=float(oo.mean()), obj_std=float(oo.std()),
                          ridge_wins=int((rr > oo).sum()))
print(f"mean Ridge {rr.mean():.3f} (std {rr.std():.3f})  |O| {oo.mean():.3f} (std {oo.std():.3f})  Ridge > |O| on {(rr>oo).sum()}/7")

# ------------------------------------------------------- x_s geometry ----
print("\n== geometry of standardized x_s within each domain (pooled StandardScaler over all 7 domains) ==")
k7 = np.isin(tt, SEVEN); Z = StandardScaler().fit_transform(Xs[k7]); t7 = tt[k7]
OUT["xs_geometry"] = {}
for d in SEVEN:
    M = Z[t7 == d]; Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12); S = Mn @ Mn.T
    cos = float((S.sum() - len(S)) / (len(S) * (len(S) - 1))); sv = np.linalg.svd(M, compute_uv=False); p = sv / sv.sum()
    er = float(np.exp(-(p * np.log(p + 1e-12)).sum())); OUT["xs_geometry"][d] = dict(mean_cos=cos, eff_rank=er)
    print(f"{d:22s} mean pairwise cosine {cos:+.3f}  effective rank {er:.2f}")
cs = [OUT["xs_geometry"][d]["mean_cos"] for d in SEVEN]; ers = [OUT["xs_geometry"][d]["eff_rank"] for d in SEVEN]
OUT["xs_geometry_summary"] = dict(cos_min=min(cs), cos_max=max(cs), er_min=min(ers), er_max=max(ers), er_mean=float(np.mean(ers)))
print(f"summary: cosine {min(cs):+.3f} .. {max(cs):+.3f}; effective rank {min(ers):.1f} .. {max(ers):.1f} (mean {np.mean(ers):.1f})")

json.dump(OUT, open("paper_numbers.json", "w"), indent=2); print("\nsaved paper_numbers.json")
