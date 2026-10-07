"""
plan_step49_custom_domains_v3.py
====================================
Fills Appendix E's missing results row (app:custom_datasets): re-labels the three custom non-IPC domains
(Rescue Mission, Robot Arm Stacking, Library Books) under the SAME rules as the rest of the corrected
paper, then evaluates ARC and the structural baselines on them zero-shot.

Reuses the VERIFIED domain/problem generators from plan_step33_custom_domains.py (the ones the human
inter-annotator-agreement study in Appendix E was run on) for PDDL generation only. Everything downstream
is replaced to match plan_step1_data_prep_v3.py's conventions, because the original plan_step33 pipeline
used a different solver (pyperplan BFS/EHC with a 10s wall-clock timeout, not the paper's stated
T_max=12/N_max=5000 budget), a different, non-v3-compatible 29-dim feature extractor, and built no natural-
language description at all (so x_e could not previously have been computed for these domains):

  1. BFS labeling: the SAME bitmask BFS and the SAME (T_max=12, N_max=5000) budget as every other domain
     in this paper, not pyperplan with a wall-clock timeout.
  2. Features: v3's actual compute_surface_features (28-dim, the same features ARC/Ridge were trained on),
     not the old ad hoc extract_xs.
  3. Descriptions: built with v3's build_description() (domain name, objects, plain-language initial-state
     and goal summary) -- structural facts only, nothing derived from the plan or n*.
  4. Real FM embeddings via sentence-transformers (same encoder as the main pipeline), with a hard failure
     (not a silent random fallback) if unavailable.

USAGE:
  python plan_step49_custom_domains_v3.py --n_per_domain 200
  # then, once an ARC checkpoint exists (see plan_table1.py), score it:
  python plan_step49_custom_domains_v3.py --score --arc_checkpoint <path or inline via the snippet below>
"""
import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan_step1_data_prep_v3 as dp          # bfs_plan, build_description, compute_surface_features
from fast_bfs import bfs_plan_fast            # drop-in faster BFS, verified identical to dp.bfs_plan

spec = importlib.util.spec_from_file_location("p33", HERE / "plan_step33_custom_domains.py")
p33 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p33)

RESULTS = Path("results_planning_v3"); RESULTS.mkdir(exist_ok=True)
OUT_DATA = Path("data/custom_domains_v3"); OUT_DATA.mkdir(parents=True, exist_ok=True)
DOMAIN_SIZES = {"rescue": list(range(2, 13)), "robot_arm": list(range(3, 13)), "library": list(range(2, 6))}
N_PER_LEVEL_DEFAULT = 20


def parse_typed_objects(problem_pddl):
    """Returns {type_name: [obj1, obj2, ...]} from a `(:objects a b c - type ...)` block."""
    m = re.search(r":objects\s*(.*?)\)\s*\(:init", problem_pddl, re.S)
    assert m, "could not find :objects block"
    groups = {}
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or "-" not in line:
            continue
        objs, typ = line.rsplit("-", 1)
        groups.setdefault(typ.strip(), []).extend(objs.split())
    return groups


def parse_facts(problem_pddl, section):
    """section: 'init' or 'goal'. Returns a list of '(pred arg1 arg2)' strings."""
    if section == "init":
        m = re.search(r":init\s*(.*?)\s*\)\s*\(:goal", problem_pddl, re.S)
    else:
        m = re.search(r":goal\s*\(and\s*(.*?)\)\s*\)\s*\)\s*$", problem_pddl, re.S)
    assert m, f"could not find :{section} block"
    return re.findall(r"\([^()]+\)", m.group(1))


def ground_actions(domain_key, typed):
    """Hand-written ground-action lists for each of the three custom domains, built directly from the
    PDDL action schemas in plan_step33_custom_domains.DOMAINS -- NOT derived from or shared with the BFS
    code that originally labeled these instances (that pyperplan-based solver is not reused anywhere in
    this script), so the relabeling is an independent re-implementation, not a rerun of the same code."""
    A = lambda name, pre, add, dele: {"name": name, "pre": frozenset(pre), "add": frozenset(add), "del": frozenset(dele)}
    acts = []
    if domain_key == "rescue":
        agents, victims, rooms, supplies = typed["agent"], typed["victim"], typed["room"], typed["supply"]
        for a in agents:
            for r1 in rooms:
                for r2 in rooms:
                    if r1 == r2: continue
                    acts.append(A(f"(move {a} {r1} {r2})",
                                 {f"(agent-at {a} {r1})", f"(connected {r1} {r2})"},
                                 {f"(agent-at {a} {r2})"}, {f"(agent-at {a} {r1})"}))
            for s in supplies:
                for r in rooms:
                    acts.append(A(f"(pick-supply {a} {s} {r})",
                                 {f"(agent-at {a} {r})", f"(supply-at {s} {r})"},
                                 {f"(agent-has-supply {a} {s})"}, {f"(supply-at {s} {r})"}))
            for v in victims:
                for s in supplies:
                    for r in rooms:
                        acts.append(A(f"(rescue-victim {a} {v} {s} {r})",
                                     {f"(agent-at {a} {r})", f"(victim-at {v} {r})",
                                      f"(agent-has-supply {a} {s})", f"(victim-needs-supply {v} {s})"},
                                     {f"(victim-rescued {v})"}, set()))
    elif domain_key == "robot_arm":
        blocks, tables = typed["block"], typed["table"]
        for b in blocks:
            for t in tables:
                acts.append(A(f"(pick-from-table {b} {t})",
                             {f"(on-table {b} {t})", f"(clear {b})", "(arm-empty)"},
                             {f"(holding {b})", f"(table-free {t})"},
                             {f"(on-table {b} {t})", f"(clear {b})", "(arm-empty)"}))
                acts.append(A(f"(place-on-table {b} {t})",
                             {f"(holding {b})", f"(table-free {t})"},
                             {f"(on-table {b} {t})", f"(clear {b})", "(arm-empty)"},
                             {f"(table-free {t})", f"(holding {b})"}))
        for b1 in blocks:
            for b2 in blocks:
                if b1 == b2: continue
                for t in tables:
                    acts.append(A(f"(pick-from-block {b1} {b2} {t})",
                                 {f"(on {b1} {b2})", f"(clear {b1})", "(arm-empty)", f"(on-table {b2} {t})"},
                                 {f"(holding {b1})", f"(clear {b2})"},
                                 {f"(on {b1} {b2})", f"(clear {b1})", "(arm-empty)"}))
                    acts.append(A(f"(stack {b1} {b2} {t})",
                                 {f"(holding {b1})", f"(clear {b2})", f"(on-table {b2} {t})"},
                                 {f"(on {b1} {b2})", f"(clear {b1})", "(arm-empty)"},
                                 {f"(holding {b1})", f"(clear {b2})"}))
    elif domain_key == "library":
        books, shelves, carts, patrons = typed["book"], typed["shelf"], typed["cart"], typed["patron"]
        for b in books:
            for s in shelves:
                for p in patrons:
                    acts.append(A(f"(checkout-book {b} {s} {p})",
                                 {f"(book-on-shelf {b} {s})", f"(patron-at-shelf {p} {s})"},
                                 {f"(book-with-patron {b} {p})", f"(shelf-has-space {s})"},
                                 {f"(book-on-shelf {b} {s})"}))
            for p in patrons:
                for c in carts:
                    for s in shelves:
                        acts.append(A(f"(return-to-cart {b} {p} {c} {s})",
                                     {f"(book-with-patron {b} {p})", f"(cart-at-shelf {c} {s})"},
                                     {f"(book-on-cart {b} {c})"}, {f"(book-with-patron {b} {p})"}))
            for c in carts:
                for s in shelves:
                    acts.append(A(f"(shelve-book {b} {c} {s})",
                                 {f"(book-on-cart {b} {c})", f"(shelf-has-space {s})", f"(correct-shelf {b} {s})",
                                  f"(cart-at-shelf {c} {s})"},
                                 {f"(book-on-shelf {b} {s})", f"(book-returned {b})"},
                                 {f"(book-on-cart {b} {c})", f"(shelf-has-space {s})"}))
        for c in carts:
            for s1 in shelves:
                for s2 in shelves:
                    if s1 == s2: continue
                    acts.append(A(f"(move-cart {c} {s1} {s2})",
                                 {f"(cart-at-shelf {c} {s1})", f"(shelf-has-space {s2})"},
                                 {f"(cart-at-shelf {c} {s2})"}, {f"(cart-at-shelf {c} {s1})"}))
    else:
        raise ValueError(domain_key)
    return acts


DESC_WORDS = {"rescue": ("rescue mission", "agents, victims, rooms and supplies"),
              "robot_arm": ("robot arm stacking", "blocks and tables"),
              "library": ("library reshelving", "books, shelves, a cart and patrons")}


def build_episode(domain_key, n, seed, instance_id):
    p33_key = {"rescue": "rescue", "robot_arm": "robot_arm", "library": "library"}[domain_key]
    problem_pddl = p33.DOMAINS[p33_key]["generate"](n, seed)
    domain_pddl = p33.DOMAINS[p33_key]["domain"]
    typed = parse_typed_objects(problem_pddl)
    init_facts = parse_facts(problem_pddl, "init")
    goal_facts = parse_facts(problem_pddl, "goal")
    actions = ground_actions(domain_key, typed)
    plan, n_steps, valid = bfs_plan_fast(set(init_facts), set(goal_facts), actions)

    name, obj_summary = DESC_WORDS[domain_key]
    description = dp.build_description(name, obj_summary,
                                       f"{len(init_facts)} initial facts over {sum(len(v) for v in typed.values())} objects",
                                       f"{len(goal_facts)} goal condition(s)")
    n_objects = sum(len(v) for v in typed.values())
    return {"domain": domain_key, "n_objects": n_objects, "problem_pddl": problem_pddl, "domain_pddl": domain_pddl,
            "plan": plan or [], "n_steps": n_steps, "valid": valid, "description": description,
            "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts),
            "task_type": domain_key, "instance_id": instance_id}


def independent_resim(ep):
    """Re-simulate the saved plan against a FRESH ground-action list built independently (called again,
    not reusing any cached action objects from build_episode), confirming the relabeling is self-consistent."""
    actions = {a["name"]: a for a in ground_actions(ep["domain"], parse_typed_objects(ep["problem_pddl"]))}
    state = set(ep["init_facts"])
    for name in ep["plan"]:
        a = actions.get(name)
        if a is None or not (a["pre"] <= state):
            return False
        state = (state - a["del"]) | a["add"]
    return set(ep["goal_facts"]) <= state


def generate(args):
    all_eps = []
    for dom, sizes in DOMAIN_SIZES.items():
        # Use EVERY documented size level for this domain (not a truncated subset), with enough seeds per
        # level to land near --n_total instances total for the domain, matching the "200 attempts" scale
        # described in the paper appendix regardless of how many size levels that domain has.
        n_total = args.n_total
        seeds_per_level = max(1, -(-n_total // len(sizes)))  # ceil division
        n_valid = 0
        dom_start = len(all_eps)
        for lvl, n in enumerate(sizes):
            for s in range(seeds_per_level):
                iid = len(all_eps)
                try:
                    ep = build_episode(dom, n, lvl * 10000 + s, iid)
                except Exception:
                    continue
                ep["complexity"] = lvl
                all_eps.append(ep)
                n_valid += int(ep["valid"] and ep["n_steps"] > 0)
        print(f"{dom}: {len(all_eps) - dom_start} generated ({len(sizes)} size levels x {seeds_per_level} seeds), "
              f"{n_valid} BFS-solved (n*>0)")

    # leak check + independent re-simulation check before saving anything
    leak_words = ["estimated", "easy", "medium", "hard", "difficulty", "steps."]
    leaked = [e for e in all_eps if any(w in e["description"].lower() for w in leak_words)]
    assert not leaked, f"{len(leaked)} leaked descriptions -- aborting, fix before saving"
    solved = [e for e in all_eps if e["valid"] and e["n_steps"] > 0]
    bad = sum(0 if independent_resim(e) else 1 for e in solved)
    print(f"\nIndependent re-simulation: {len(solved)} solved plans checked, {bad} failures")
    assert bad == 0, "re-simulation failures -- do not trust these labels, do not save"

    X_surf = dp.compute_surface_features(all_eps)
    np.save(OUT_DATA / "X_surf.npy", X_surf)
    (OUT_DATA / "episodes.json").write_text(json.dumps(all_eps, indent=2))
    y_nsteps = np.array([e["n_steps"] for e in all_eps], dtype=np.float32)
    y_valid = np.array([e["valid"] for e in all_eps], dtype=bool)
    np.save(OUT_DATA / "y_nsteps.npy", y_nsteps)
    np.save(OUT_DATA / "y_valid.npy", y_valid)
    np.save(OUT_DATA / "task_types.npy", np.array([e["task_type"] for e in all_eps]))
    # dp.compute_fm_embeddings() catches its own ImportError internally and silently returns a RANDOM
    # fallback vector rather than raising -- so a bare try/except around it would never fire and would
    # wrongly report success even when sentence-transformers is missing. Check availability explicitly first.
    try:
        import sentence_transformers  # noqa: F401
        have_real_encoder = True
    except ImportError:
        have_real_encoder = False
    X_fm = dp.compute_fm_embeddings(all_eps)
    np.save(OUT_DATA / "X_fm.npy", X_fm)
    if have_real_encoder:
        print("Real sentence-transformer embeddings saved.")
    else:
        print("WARNING: sentence-transformers is not installed. X_fm.npy contains a RANDOM fallback, "
              "NOT real embeddings. ARC scoring against this file would be meaningless; x_s-only "
              "baselines (|O|, goal count, Ridge on x_s) are unaffected and still valid. "
              "Install sentence-transformers and rerun generation before scoring ARC.")
    print(f"\nSaved -> {OUT_DATA}/  ({len(all_eps)} total, {y_valid.sum()} BFS-solved)")


def score(args):
    eps = json.loads((OUT_DATA / "episodes.json").read_text())
    Xs = np.load(OUT_DATA / "X_surf.npy")
    y = np.load(OUT_DATA / "y_nsteps.npy"); v = np.load(OUT_DATA / "y_valid.npy")
    tt = np.array([str(t) for t in np.load(OUT_DATA / "task_types.npy", allow_pickle=True)])
    n_goal = np.array([len(e["goal_facts"]) for e in eps], float)
    # BUG FIXED: filtering on `v` (valid) alone includes degenerate n*=0 instances (goal already
    # true in the initial state), which every other correlation in this paper excludes -- this is
    # why an earlier run reported N=87 for robot_arm when generation itself had logged only 85
    # BFS-solved (n*>0) instances. Now consistent with Table 1 / Appendix A's convention.
    m = v & (y > 0); Xs, y, tt, n_goal = Xs[m], y[m], tt[m], n_goal[m]
    n_obj = Xs[:, 0]

    sys.path.insert(0, str(HERE))
    from plan_table1 import load_data as load_main_data
    Xs_main, Xf_main, y_main, tt_main = load_main_data()
    tr = np.isin(tt_main, ["depot", "rovers", "satellite"])
    sc = StandardScaler().fit(Xs_main[tr])
    ridge = Ridge(alpha=1.0).fit(sc.transform(Xs_main[tr]), y_main[tr])
    ridge_pred = ridge.predict(sc.transform(Xs))

    out = {}
    for d in ["rescue", "robot_arm", "library"]:
        k = tt == d
        if k.sum() < 10:
            print(f"{d}: only {k.sum()} valid instances, skipping"); continue
        row = {"n": int(k.sum()),
              "object_cardinality": float(abs(stats.spearmanr(n_obj[k], y[k])[0])),
              "goal_count": float(abs(stats.spearmanr(n_goal[k], y[k])[0])),
              "ridge_xs": float(abs(stats.spearmanr(ridge_pred[k], y[k])[0]))}
        if args.arc_checkpoint and Path(OUT_DATA / "X_fm.npy").exists():
            print(f"{d}: scoring the provided ARC checkpoint -- see --help for how this must be produced.")
            # left as an integration point: load the checkpoint's model.state_dict() + model.prep into a
            # freshly constructed PlanningGURU and call _score_domain_given_support, exactly as in
            # plan_step46_routing.py's get_arc_scores_for_routing_snippet().
        out[d] = row
        print(f"{d:10s} N={row['n']:3d}  |O|={row['object_cardinality']:.3f}  "
              f"goal_count={row['goal_count']:.3f}  Ridge(x_s)={row['ridge_xs']:.3f}")
    (RESULTS / "custom_domains_v3_scores.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {RESULTS}/custom_domains_v3_scores.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=200,
                    help="target total instances per domain, spread evenly across that domain's "
                         "documented size levels (matches the original '200 attempts' scale)")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--arc_checkpoint", default=None)
    args = ap.parse_args()
    if args.score:
        score(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
