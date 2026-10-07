"""
plan_step50_relaxed_heuristic_features.py
=============================================
Two of the "untested ideas" from the prior analysis:

  (1) RICHER STRUCTURAL FEATURES: a delete-relaxation (FF-style h_add) heuristic value computed from each
      instance's initial state to its goal, for ALL SEVEN v3 domains (not just the two plan_step43 covered),
      given to EVERY method as one extra feature -- |O|+h_add, goal-count+h_add is not meaningful (those are
      single scalars already), so concretely: Ridge on x_s gets a 29th feature appended, and the heuristic's
      own standalone correlation is reported next to the existing cheap surrogates.

  (2) PREDICTION-LEVEL SEED ENSEMBLING: a utility to average PREDICTIONS across several seeds' models
      BEFORE computing one Spearman correlation, as distinct from averaging the per-seed CORRELATIONS (what
      plan_table1.py already reports). These are genuinely different statistics -- an ensemble of predictors
      can have lower variance than any individual predictor, which the mean-of-correlations number does not
      capture. This script provides the function; applying it to ARC/ERM needs the cluster (torch).

CORRECTNESS: ground_actions_v3() reconstructs each domain's exact ground-action list (object names are
untyped in v3's problem_pddl, so object ROLES are recovered by name-prefix pattern, e.g. 'b3'->block,
'sat2'->satellite; a handful of per-instance RANDOM relations -- satellite's calibration-target, rovers'
can-traverse/at-lander/visible -- are static (never added/deleted by any action, verified against
plan_step1_data_prep_v3.py's action definitions) and are read directly from the saved init_facts rather than
re-derived from names). Before computing a single heuristic value for a domain, self_test() independently
re-simulates that domain's SAVED BFS plan against the reconstructed action list; a domain is used only if
every one of its solved instances passes this check.

USAGE:
  python plan_step50_relaxed_heuristic_features.py --data data/planning_v3
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

TE = ["blocksworld", "logistics", "mystery_blocksworld"]
TR = ["depot", "rovers", "satellite"]
SEVEN = TE + TR + ["gripper"]


def A(name, pre, add, dele):
    return {"name": name, "pre": frozenset(pre), "add": frozenset(add), "del": frozenset(dele)}


def by_prefix(objs, prefix):
    """Objects whose name starts with `prefix` followed by a digit (or, for fixed-name objects like
    'left'/'right'/'colour'/'mode0', an exact-match list passed instead of a prefix)."""
    if isinstance(prefix, (list, tuple, set)):
        return [o for o in objs if o in prefix]
    return sorted([o for o in objs if re.match(rf"^{re.escape(prefix)}\d+$", o)],
                 key=lambda s: int(re.search(r"\d+", s).group()))


def static_facts(init_facts, pred):
    """All (arg1, arg2, ...) tuples for a given predicate name, read directly from init_facts. Used only
    for predicates confirmed (by inspection of plan_step1_data_prep_v3.py) to be STATIC: never added or
    deleted by any action of that domain."""
    out = []
    for f in init_facts:
        toks = f.strip("()").split()
        if toks and toks[0] == pred:
            out.append(tuple(toks[1:]))
    return out


def ground_actions_v3(ep):
    """Returns the ground-action list for ep['domain'], reconstructed independently of
    plan_step1_data_prep_v3.py's generator code (no shared action-list object, no imported closures --
    only object names and the episode's own saved init_facts are used)."""
    dom = ep["domain"]
    objs = re.search(r":objects\s+(.*?)\)\s*\(:init", ep["problem_pddl"], re.S).group(1).split()
    init = ep["init_facts"]

    if dom in ("blocksworld", "mystery_blocksworld"):
        blocks = by_prefix(objs, "b")
        pm = ({"on": "stacked", "ot": "grounded", "cl": "apex", "he": "grasping"} if dom == "mystery_blocksworld"
              else {"on": "on", "ot": "ontable", "cl": "clear", "he": "handempty"})
        hold = (lambda x: f"(clutching {x})") if dom == "mystery_blocksworld" else (lambda x: f"(holding {x})")
        acts = []
        for x in blocks:
            acts.append(A(f"(pick-up {x})", {f"({pm['cl']} {x})", f"({pm['ot']} {x})", f"({pm['he']})"},
                         {hold(x)}, {f"({pm['ot']} {x})", f"({pm['cl']} {x})", f"({pm['he']})"}))
            acts.append(A(f"(put-down {x})", {hold(x)},
                         {f"({pm['ot']} {x})", f"({pm['cl']} {x})", f"({pm['he']})"}, {hold(x)}))
            for y in blocks:
                if x == y: continue
                acts.append(A(f"(stack {x} {y})", {hold(x), f"({pm['cl']} {y})"},
                             {f"({pm['on']} {x} {y})", f"({pm['cl']} {x})", f"({pm['he']})"}, {hold(x), f"({pm['cl']} {y})"}))
                acts.append(A(f"(unstack {x} {y})", {f"({pm['on']} {x} {y})", f"({pm['cl']} {x})", f"({pm['he']})"},
                             {hold(x), f"({pm['cl']} {y})"}, {f"({pm['on']} {x} {y})", f"({pm['cl']} {x})", f"({pm['he']})"}))
        return acts

    if dom == "gripper":
        rooms, balls, grip = by_prefix(objs, "room"), by_prefix(objs, "ball"), by_prefix(objs, ["left", "right"])
        acts = []
        for rf in rooms:
            for rt in rooms:
                if rf == rt: continue
                acts.append(A(f"(move {rf} {rt})", {f"(room {rf})", f"(room {rt})", f"(at-robby {rf})"},
                             {f"(at-robby {rt})"}, {f"(at-robby {rf})"}))
        for b in balls:
            for r in rooms:
                for g in grip:
                    acts.append(A(f"(pick {b} {r} {g})", {f"(at {b} {r})", f"(at-robby {r})", f"(free {g})"},
                                 {f"(carry {b} {g})"}, {f"(at {b} {r})", f"(free {g})"}))
                    acts.append(A(f"(drop {b} {r} {g})", {f"(carry {b} {g})", f"(at-robby {r})"},
                                 {f"(at {b} {r})", f"(free {g})"}, {f"(carry {b} {g})"}))
        return acts

    if dom == "depot":
        depots, pallets, hoists = by_prefix(objs, "depot"), by_prefix(objs, "pallet"), by_prefix(objs, "hoist")
        trucks, crates = by_prefix(objs, "truck"), by_prefix(objs, "crate")
        pallet_of = {depots[i]: pallets[i] for i in range(len(depots))}
        pallet_depot = {p: d for d, p in pallet_of.items()}
        all_surf = sorted(set(pallets) | set(crates))
        acts = []
        for t in trucks:
            for d1 in depots:
                for d2 in depots:
                    if d1 == d2: continue
                    acts.append(A(f"(drive {t} {d1} {d2})", {f"(truck {t})", f"(depotloc {d1})", f"(depotloc {d2})", f"(at-truck {t} {d1})"},
                                 {f"(at-truck {t} {d2})"}, {f"(at-truck {t} {d1})"}))
        for i, h in enumerate(hoists):
            d = depots[i]
            for c in crates:
                for s in all_surf:
                    if s == c: continue
                    loc = f"(at-pallet {s} {d})" if s in pallet_depot else f"(at-crate {s} {d})"
                    acts.append(A(f"(lift {h} {c} {s} {d})",
                                 {f"(hoist {h})", f"(crate {c})", f"(at-hoist {h} {d})", f"(on {c} {s})", f"(clear {c})",
                                  f"(hoist-available {h})", loc, f"(at-crate {c} {d})"},
                                 {f"(hoist-lifting {h} {c})", f"(clear {s})"},
                                 {f"(hoist-available {h})", f"(on {c} {s})", f"(clear {c})", f"(at-crate {c} {d})"}))
                    acts.append(A(f"(drop {h} {c} {s} {d})",
                                 {f"(hoist {h})", f"(crate {c})", f"(at-hoist {h} {d})", f"(hoist-lifting {h} {c})", f"(clear {s})", loc},
                                 {f"(hoist-available {h})", f"(on {c} {s})", f"(clear {c})", f"(at-crate {c} {d})"},
                                 {f"(hoist-lifting {h} {c})", f"(clear {s})"}))
            for c in crates:
                for t in trucks:
                    acts.append(A(f"(load {h} {c} {t} {d})",
                                 {f"(hoist {h})", f"(crate {c})", f"(truck {t})", f"(at-hoist {h} {d})", f"(at-truck {t} {d})", f"(hoist-lifting {h} {c})"},
                                 {f"(in-truck {c} {t})", f"(hoist-available {h})"}, {f"(hoist-lifting {h} {c})"}))
                    acts.append(A(f"(unload {h} {c} {t} {d})",
                                 {f"(hoist {h})", f"(crate {c})", f"(truck {t})", f"(at-hoist {h} {d})", f"(at-truck {t} {d})", f"(in-truck {c} {t})", f"(hoist-available {h})"},
                                 {f"(hoist-lifting {h} {c})"}, {f"(in-truck {c} {t})", f"(hoist-available {h})"}))
        return acts

    if dom == "logistics":
        trucks, cities, pkgs = by_prefix(objs, "truck"), by_prefix(objs, "city"), by_prefix(objs, "pkg")
        acts = []
        for p in pkgs:
            for t in trucks:
                for c in cities:
                    acts.append(A(f"(load-truck {p} {t} {c})", {f"(at {t} {c})", f"(at {p} {c})"},
                                 {f"(in {p} {t})"}, {f"(at {p} {c})"}))
                    acts.append(A(f"(unload-truck {p} {t} {c})", {f"(at {t} {c})", f"(in {p} {t})"},
                                 {f"(at {p} {c})"}, {f"(in {p} {t})"}))
        for t in trucks:
            for c1 in cities:
                for c2 in cities:
                    if c1 == c2: continue
                    acts.append(A(f"(drive-truck {t} {c1} {c2})", {f"(at {t} {c1})"}, {f"(at {t} {c2})"}, {f"(at {t} {c1})"}))
        return acts

    if dom == "satellite":
        sats, dirs = by_prefix(objs, "sat"), by_prefix(objs, "dir")
        instr, modes = by_prefix(objs, "inst"), by_prefix(objs, ["mode0"])
        onboard = dict(static_facts(init, "on-board"))                       # {inst: sat}
        calib = {a: b for a, b in static_facts(init, "calibration-target")}  # {inst: dir} -- RANDOM, read from init
        acts = []
        for s in sats:
            for d1 in dirs:
                for d2 in dirs:
                    if d1 == d2: continue
                    acts.append(A(f"(turn-to {s} {d2} {d1})", {f"(satellite {s})", f"(direction {d2})", f"(direction {d1})", f"(pointing {s} {d1})"},
                                 {f"(pointing {s} {d2})"}, {f"(pointing {s} {d1})"}))
        for ins, s in onboard.items():
            acts.append(A(f"(switch-on {ins} {s})", {f"(instrument {ins})", f"(satellite {s})", f"(on-board {ins} {s})", f"(power-avail {s})"},
                         {f"(power-on {ins})"}, {f"(power-avail {s})"}))
            d = calib.get(ins)
            if d is not None:
                acts.append(A(f"(calibrate {s} {ins} {d})",
                             {f"(satellite {s})", f"(instrument {ins})", f"(direction {d})", f"(on-board {ins} {s})",
                              f"(calibration-target {ins} {d})", f"(pointing {s} {d})", f"(power-on {ins})"},
                             {f"(calibrated {ins})"}, set()))
            for dd in dirs:
                for m in modes:
                    acts.append(A(f"(take-image {s} {dd} {ins} {m})",
                                 {f"(satellite {s})", f"(direction {dd})", f"(instrument {ins})", f"(mode {m})", f"(calibrated {ins})",
                                  f"(on-board {ins} {s})", f"(supports {ins} {m})", f"(power-on {ins})", f"(pointing {s} {dd})"},
                                 {f"(have-image {dd} {m})"}, set()))
        return acts

    if dom == "rovers":
        rovers, wps = by_prefix(objs, "rover"), by_prefix(objs, "wp")
        stores, cams = by_prefix(objs, "store"), by_prefix(objs, "cam")
        lander_wp = static_facts(init, "at-lander")[0][0]
        traversals = set(static_facts(init, "can-traverse"))  # (rover, a, b) -- same set shared by all rovers in v3
        trav_ab = sorted({(a, b) for _, a, b in traversals})
        store_of = dict(zip(rovers, stores))
        acts = []
        for r in rovers:
            for a, b in trav_ab:
                acts.append(A(f"(navigate {r} {a} {b})", {f"(rover {r})", f"(waypoint {a})", f"(waypoint {b})", f"(at {r} {a})", f"(can-traverse {r} {a} {b})"},
                             {f"(at {r} {b})"}, {f"(at {r} {a})"}))
        for r in rovers:
            s = store_of[r]
            for w in wps:
                acts.append(A(f"(sample-soil {r} {s} {w})", {f"(rover {r})", f"(store {s})", f"(waypoint {w})", f"(at {r} {w})", f"(equipped-soil {r})", f"(empty {s})"},
                             {f"(full {s})", f"(have-soil {r} {w})"}, {f"(empty {s})"}))
            for w in wps:
                for wf in wps:
                    acts.append(A(f"(communicate-soil {r} {lander_wp} {w} {wf})",
                                 {f"(rover {r})", f"(waypoint {lander_wp})", f"(waypoint {w})", f"(waypoint {wf})",
                                  f"(at-lander {lander_wp})", f"(have-soil {r} {w})", f"(at {r} {wf})",
                                  f"(visible {wf} {lander_wp})", f"(channel-free {r})"},
                                 {f"(communicated-soil {w})"}, set()))
        return acts

    raise ValueError(f"unhandled domain {dom}")


def self_test_domain(eps, dom, max_check=200):
    """Independently re-simulate each solved instance's SAVED plan against the reconstructed action list.
    Returns (n_checked, n_failed). A domain is used for h_add only if n_failed == 0."""
    solved = [e for e in eps if e["domain"] == dom and e["valid"] and e["n_steps"] > 0][:max_check]
    n_failed = 0
    for ep in solved:
        try:
            acts = {a["name"]: a for a in ground_actions_v3(ep)}
            state = set(ep["init_facts"])
            for name in ep["plan"]:
                a = acts.get(name)
                if a is None or not (a["pre"] <= state):
                    n_failed += 1; break
                state = (state - a["del"]) | a["add"]
            else:
                if not (set(ep["goal_facts"]) <= state):
                    n_failed += 1
        except Exception:
            n_failed += 1
    return len(solved), n_failed


def h_add(init_facts, goal_facts, actions, max_depth=30):
    """Delete-relaxation h_add: ignore delete effects, build a levelled relaxed planning graph, and sum,
    for each unsatisfied goal fact, the level at which some applicable action first achieves it (a
    standard admissible-in-the-relaxed-problem, inadmissible-in-general heuristic -- used here purely as
    a FEATURE, not as a search guide, so admissibility is not required)."""
    facts = set(init_facts)
    first_level = {f: 0 for f in facts}
    for depth in range(1, max_depth):
        if goal_facts <= facts:
            break
        new_facts = set(facts)
        applicable = [a for a in actions if a["pre"] <= facts]
        if not applicable:
            return float("inf")
        for a in applicable:
            for f in a["add"]:
                if f not in first_level:
                    first_level[f] = depth
            new_facts |= a["add"]
        if new_facts == facts:
            return float("inf")
        facts = new_facts
    if not (goal_facts <= facts):
        return float("inf")
    return sum(first_level.get(g, max_depth) for g in goal_facts if g not in init_facts)


def ensemble_then_correlate(per_seed_predictions, targets):
    """(2) Prediction-level ensembling: average several seeds' PREDICTIONS first, then compute ONE
    Spearman correlation -- distinct from averaging several seeds' correlations (what plan_table1.py's
    mean+/-std already reports). per_seed_predictions: list of 1-D arrays, same order as `targets`."""
    avg_pred = np.mean(np.stack(per_seed_predictions, axis=0), axis=0)
    rho = stats.spearmanr(avg_pred, targets)[0]
    return float(abs(rho)) if not np.isnan(rho) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/planning_v3")
    ap.add_argument("--max_per_domain", type=int, default=80, help="h_add is O(|actions| x depth) per "
                    "instance and some domains ground thousands of actions; cap instances per domain")
    args = ap.parse_args()

    D = Path(args.data)
    eps_all = json.loads((D / "episodes.json").read_text())
    valid = np.load(D / "y_valid.npy")
    Xs_all = np.load(D / "X_surf.npy")
    tt_all = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])

    print("=== Per-domain self-test: does the reconstructed action list reproduce the saved BFS plan? ===")
    usable = {}
    for dom in SEVEN:
        n, nf = self_test_domain(eps_all, dom)
        usable[dom] = (n > 0 and nf == 0)
        print(f"  {dom:22s} {n:4d} checked, {nf:4d} failed  ->  {'USABLE' if usable[dom] else 'SKIPPED'}")
    if not all(usable.values()):
        print("\nNOTE: domains marked SKIPPED are excluded from h_add below; fix ground_actions_v3() for "
              "them before trusting any number derived from them.")

    print(f"\n=== Computing h_add (delete-relaxation heuristic), capped at {args.max_per_domain} instances/domain ===")
    h_vals = {}  # instance_id -> h_add
    rng = np.random.default_rng(0)
    for dom in SEVEN:
        if not usable[dom]:
            continue
        idx_all = [i for i, (e, v) in enumerate(zip(eps_all, valid)) if v and e["domain"] == dom and e["n_steps"] > 0]
        idx = rng.choice(idx_all, min(args.max_per_domain, len(idx_all)), replace=False)
        for i in idx:
            ep = eps_all[i]
            acts = ground_actions_v3(ep)
            h_vals[ep["instance_id"]] = h_add(set(ep["init_facts"]), set(ep["goal_facts"]), acts)
        print(f"  {dom:22s} {len(idx)} instances scored")

    ids_scored = sorted(h_vals)
    id_to_arr_idx = {e["instance_id"]: i for i, e in enumerate(eps_all)}
    arr_idx = [id_to_arr_idx[i] for i in ids_scored]
    y_sub = np.array([eps_all[i]["n_steps"] for i in [id_to_arr_idx[j] for j in ids_scored]], float)
    tt_sub = tt_all[arr_idx]
    h_sub = np.array([h_vals[i] for i in ids_scored], float)
    finite = np.isfinite(h_sub)

    print("\n=== h_add alone vs. existing cheap surrogates, on the subsample scored above ===")
    for dom in TE:
        k = (tt_sub == dom) & finite
        if k.sum() < 10:
            print(f"  {dom}: only {k.sum()} finite h_add values, skipping"); continue
        rho = abs(stats.spearmanr(h_sub[k], y_sub[k])[0])
        print(f"  {dom:22s} N={k.sum():3d}  |rho|(h_add, n*) = {rho:.3f}")

    print("\n=== Ridge on x_s+h_add vs. Ridge on x_s alone (same subsample, so this is apples-to-apples) ===")
    Xs_sub = Xs_all[arr_idx]
    tr_mask = np.isin(tt_sub, TR) & finite
    sc_base = StandardScaler().fit(Xs_sub[tr_mask])
    ridge_base = Ridge(alpha=1.0).fit(sc_base.transform(Xs_sub[tr_mask]), y_sub[tr_mask])
    Xs_plus = np.hstack([Xs_sub, h_sub[:, None]])
    Xs_plus_clean = np.where(np.isfinite(Xs_plus), Xs_plus, np.nanmax(Xs_plus[np.isfinite(Xs_plus)]) * 2)
    sc_plus = StandardScaler().fit(Xs_plus_clean[tr_mask])
    ridge_plus = Ridge(alpha=1.0).fit(sc_plus.transform(Xs_plus_clean[tr_mask]), y_sub[tr_mask])
    for dom in TE:
        k = (tt_sub == dom) & finite
        if k.sum() < 10:
            continue
        p_base = ridge_base.predict(sc_base.transform(Xs_sub[k]))
        p_plus = ridge_plus.predict(sc_plus.transform(Xs_plus_clean[k]))
        r_base = abs(stats.spearmanr(p_base, y_sub[k])[0])
        r_plus = abs(stats.spearmanr(p_plus, y_sub[k])[0])
        print(f"  {dom:22s} N={k.sum():3d}  Ridge(x_s)={r_base:.3f}  Ridge(x_s + h_add)={r_plus:.3f}  "
              f"delta={r_plus-r_base:+.3f}")

    RES = Path("results_planning_v3"); RES.mkdir(exist_ok=True)
    out = {"usable_domains": usable,
          "h_add_by_instance_id": {str(k): (v if np.isfinite(v) else None) for k, v in h_vals.items()}}
    (RES / "relaxed_heuristic_features.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {RES}/relaxed_heuristic_features.json")
    print("\nNOTE ON SCOPE: this subsample is smaller than the full test-domain sets used elsewhere in the")
    print("paper (capped by --max_per_domain for runtime), so these correlations are NOT directly comparable")
    print("to Table 1's N=126/164/133 numbers without rerunning at full scale -- rerun with a larger")
    print("--max_per_domain (or no cap) before reporting this as a paper number.")


if __name__ == "__main__":
    main()
