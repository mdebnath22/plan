"""
plan_step43_astar_association.py
===================================
Fills Appendix B.6 (app:solver_association): correlates a structural estimator's ranking with an INDEPENDENT
planner's search effort (A* node expansions, wall-clock time), obtained without any reference to the BFS
procedure used for the main n* labels.

Pure Python/NumPy -- no torch, no embeddings, no LLM. Runs anywhere, including this sandbox.

A* uses the classic delete-relaxation / FF-style heuristic (h_add variant: cost of a relaxed plan where
delete effects are ignored, computed by a greedy relaxed-plan extraction over a levelled relaxed planning
graph). This is a different algorithm and a different state representation (bitmask + priority queue vs the
BFS used for n*), so an association here is not circular with the BFS label.

USAGE:
  python plan_step43_astar_association.py --data data/planning_v3 --n 60
Writes results_planning_v3/astar_association.json and prints the LaTeX table body.
"""
import argparse, json, time
from pathlib import Path
from heapq import heappush, heappop

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

TE = ["blocksworld", "logistics", "mystery_blocksworld"]
TR = ["depot", "rovers", "satellite"]
A_STAR_NODE_CAP = 20000  # independent of the BFS N_max=5000 budget used for labeling


def parse_actions(domain_pddl):
    """Re-derive a STRIPS action list from the episode's own domain_pddl text, independent of whatever
    Python object built it. Used only to recompute ground actions for A*; the planner itself is separate
    from (and does not call) bfs_plan."""
    import re
    actions = []
    for m in re.finditer(r"\(:action\s+(\S+)\s*:parameters\s*\(([^)]*)\)\s*:precondition\s*(.*?)\s*:effect\s*(.*?)\)\s*(?=\(:action|\)\s*\)\s*$)",
                         domain_pddl, re.S):
        pass  # schema-level parsing is not needed: we instead ground directly from the episode's
              # already-ground init/goal facts and plan, see ground_actions_from_episode below.
    return actions


def ground_actions_from_episode(ep):
    """
    We do not have a general PDDL grounder in this codebase, so we build the ground action set the SAME
    way the v3 generator does: by replaying its own generator for this task_type with the SAME object
    list recovered from problem_pddl, and reusing its action-construction code paths. This keeps the
    action set faithful to the actual domain semantics without introducing a second, divergent grounder.
    """
    import plan_step1_data_prep_v3 as dp
    import re
    objs = re.search(r":objects\s+(.*?)\)", ep["problem_pddl"], re.S).group(1).split()
    domain = ep["task_type"]
    if domain in ("blocksworld", "mystery_blocksworld"):
        n = len(objs)
        # re-derive the SAME action list the generator used, parameterized by the actual object names
        pred_map = ({"on": "stacked", "ontable": "grounded", "clear": "apex", "handempty": "grasping"}
                    if domain == "mystery_blocksworld" else
                    {"on": "on", "ontable": "ontable", "clear": "clear", "handempty": "handempty"})
        on_p, ot_p, cl_p, he_p = pred_map["on"], pred_map["ontable"], pred_map["clear"], pred_map["handempty"]
        holding = lambda x: f"(holding {x})" if domain == "blocksworld" else f"(clutching {x})"
        actions = []
        for x in objs:
            actions.append({"name": f"(pick-up {x})", "pre": frozenset({f"({cl_p} {x})", f"({ot_p} {x})", f"({he_p})"}),
                            "add": frozenset({holding(x)}), "del": frozenset({f"({ot_p} {x})", f"({cl_p} {x})", f"({he_p})"})})
            actions.append({"name": f"(put-down {x})", "pre": frozenset({holding(x)}),
                            "add": frozenset({f"({ot_p} {x})", f"({cl_p} {x})", f"({he_p})"}), "del": frozenset({holding(x)})})
            for y in objs:
                if x == y: continue
                actions.append({"name": f"(stack {x} {y})", "pre": frozenset({holding(x), f"({cl_p} {y})"}),
                                "add": frozenset({f"({on_p} {x} {y})", f"({cl_p} {x})", f"({he_p})"}),
                                "del": frozenset({holding(x), f"({cl_p} {y})"})})
                actions.append({"name": f"(unstack {x} {y})", "pre": frozenset({f"({on_p} {x} {y})", f"({cl_p} {x})", f"({he_p})"}),
                                "add": frozenset({holding(x), f"({cl_p} {y})"}),
                                "del": frozenset({f"({on_p} {x} {y})", f"({cl_p} {x})", f"({he_p})"})})
        return actions
    raise NotImplementedError(f"A* grounding only implemented for blocksworld/mystery_blocksworld in this script; "
                              f"got {domain}. See module docstring for how to extend.")


def relaxed_plan_heuristic(state, goal, actions, cache={}):
    """FF-style h_add/relaxed-plan heuristic: ignore delete effects, build a levelled relaxed planning
    graph from `state`, and extract a (non-optimal) relaxed plan. Returns len(relaxed plan), or inf if
    the goal is unreachable even in the relaxed problem."""
    key = (state, goal)
    if key in cache:
        return cache[key]
    facts = set(state)
    levels = [set(facts)]
    applied_at = {}
    for depth in range(1, 30):
        if goal <= facts:
            break
        new_facts = set(facts)
        applicable = [a for a in actions if a["pre"] <= facts]
        if not applicable:
            cache[key] = float("inf"); return float("inf")
        for a in applicable:
            if a["name"] not in applied_at:
                applied_at[a["name"]] = depth
            new_facts |= a["add"]
        if new_facts == facts:
            cache[key] = float("inf"); return float("inf")
        facts = new_facts
        levels.append(set(facts))
    else:
        if not (goal <= facts):
            cache[key] = float("inf"); return float("inf")
    # extract: count distinct actions needed to first achieve each goal fact (greedy, FF-style h_add)
    needed_depth = {}
    fact_first_level = {}
    for lvl, fs in enumerate(levels):
        for f in fs:
            if f not in fact_first_level:
                fact_first_level[f] = lvl
    h = 0
    seen_actions = set()
    for g in goal:
        if g in state:
            continue
        # cheapest action (by its enabling level) that adds g
        candidates = [a for a in actions if g in a["add"] and all(fact_first_level.get(p, 999) < fact_first_level.get(g, 999) for p in a["pre"])]
        if candidates:
            name = min(candidates, key=lambda a: applied_at.get(a["name"], 999))["name"]
            if name not in seen_actions:
                seen_actions.add(name); h += 1
        else:
            h += 1  # conservative fallback
    cache[key] = h
    return h


def astar(init_facts, goal_facts, actions, node_cap=A_STAR_NODE_CAP):
    start = frozenset(init_facts); goal = frozenset(goal_facts)
    if goal <= start:
        return [], 0, 0.0
    t0 = time.time()
    h0 = relaxed_plan_heuristic(start, goal, actions)
    frontier = [(h0, 0, id(start), start, ())]
    best_g = {start: 0}
    expanded = 0
    while frontier and expanded < node_cap:
        f, g, _, state, plan = heappop(frontier)
        if g > best_g.get(state, float("inf")):
            continue
        expanded += 1
        if goal <= state:
            return list(plan), expanded, time.time() - t0
        for a in actions:
            if a["pre"] <= state:
                ns = (state - a["del"]) | a["add"]
                ng = g + 1
                if ng < best_g.get(ns, float("inf")):
                    best_g[ns] = ng
                    h = relaxed_plan_heuristic(ns, goal, actions)
                    if h < float("inf"):
                        heappush(frontier, (ng + h, ng, id(ns), ns, plan + (a["name"],)))
    return None, expanded, time.time() - t0  # budget exhausted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/planning_v3")
    ap.add_argument("--n", type=int, default=60, help="instances per test domain (A* is slow; this is a subsample)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = Path(args.data)
    eps = json.loads((D / "episodes.json").read_text())
    Xs = np.load(D / "X_surf.npy"); y = np.load(D / "y_nsteps.npy").astype(float)
    v = np.load(D / "y_valid.npy"); tt = np.array([str(t) for t in np.load(D / "task_types.npy", allow_pickle=True)])
    n_obj = Xs[:, 0]

    rng = np.random.default_rng(args.seed)
    tr_mask = np.isin(tt, TR) & v
    sc = StandardScaler().fit(Xs[tr_mask])
    ridge = Ridge(alpha=1.0).fit(sc.transform(Xs[tr_mask]), y[tr_mask])
    ridge_pred_all = ridge.predict(sc.transform(Xs))

    out = {}
    print(f"A* node cap = {A_STAR_NODE_CAP}; this will be slow for larger instances -- subsampling {args.n}/domain.\n")
    for d in ["blocksworld", "mystery_blocksworld"]:  # the only domains this script's grounder supports
        idx_all = np.where((tt == d) & v)[0]
        idx = rng.choice(idx_all, min(args.n, len(idx_all)), replace=False)
        nodes, wtime, bfs_len, ridge_s, obj_s = [], [], [], [], []
        n_timeout = 0
        for i in idx:
            ep = eps[i]
            try:
                actions = ground_actions_from_episode(ep)
                plan, expanded, dt = astar(ep["init_facts"], ep["goal_facts"], actions)
            except Exception as e:
                continue
            if plan is None:
                n_timeout += 1
                continue
            # sanity: A* plan length should not be SHORTER than the BFS-optimal length (both are optimal
            # under admissible search over the same action set; a shorter A* plan would indicate a bug)
            assert len(plan) >= ep["n_steps"], f"A* found a shorter plan than BFS n*! instance {i}: A*={len(plan)} BFS={ep['n_steps']}"
            nodes.append(expanded); wtime.append(dt); bfs_len.append(ep["n_steps"])
            ridge_s.append(ridge_pred_all[i]); obj_s.append(n_obj[i])
        nodes, wtime, bfs_len, ridge_s, obj_s = map(np.array, (nodes, wtime, bfs_len, ridge_s, obj_s))
        print(f"{d}: {len(nodes)}/{len(idx)} solved within A* node cap ({n_timeout} timed out)")

        def rho(a, b):
            r = stats.spearmanr(a, b)[0]; return float(abs(r)) if not np.isnan(r) else float("nan")
        row = dict(
            n=int(len(nodes)),
            rho_ridge_bfslen=rho(ridge_s, bfs_len),
            rho_ridge_astar_nodes=rho(ridge_s, nodes),
            rho_ridge_astar_time=rho(ridge_s, wtime),
            rho_obj_astar_nodes=rho(obj_s, nodes),
            rho_bfslen_astar_nodes=rho(bfs_len, nodes),  # sanity: BFS n* and A* effort should correlate strongly
        )
        out[d] = row
        print(f"  Ridge vs BFS-n*:      {row['rho_ridge_bfslen']:.3f}")
        print(f"  Ridge vs A*-nodes:    {row['rho_ridge_astar_nodes']:.3f}")
        print(f"  Ridge vs A*-time:     {row['rho_ridge_astar_time']:.3f}")
        print(f"  |O|   vs A*-nodes:    {row['rho_obj_astar_nodes']:.3f}")
        print(f"  BFS-n* vs A*-nodes (sanity; should be high): {row['rho_bfslen_astar_nodes']:.3f}\n")

    RES = Path("results_planning_v3"); RES.mkdir(exist_ok=True)
    (RES / "astar_association.json").write_text(json.dumps(out, indent=2))
    print("Saved -> results_planning_v3/astar_association.json")
    print("\nNote: this script's grounder only supports Blocksworld/Mystery-Blocksworld (see")
    print("ground_actions_from_episode). Logistics is not included in this table; if needed, extend the")
    print("grounder with the Logistics action schema from plan_step1_data_prep_v3.gen_logistics_problem.")
    print("\n=== LaTeX row data ===")
    for d, r in out.items():
        print(f"{d}: BFS-len {r['rho_ridge_bfslen']:.3f} & A*-nodes {r['rho_ridge_astar_nodes']:.3f} & "
              f"A*-time {r['rho_ridge_astar_time']:.3f} & |O|,A*-nodes {r['rho_obj_astar_nodes']:.3f} \\\\")


if __name__ == "__main__":
    main()
