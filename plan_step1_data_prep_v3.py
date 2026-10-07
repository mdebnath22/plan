"""
plan_step1_data_prep_v3.py
============================
Corrected data generation pipeline. See PREPROCESSING_STEPS.md for the
full list of what changed and why. Summary of fixes vs. the original
plan_step1_data_prep.py:

  1. Depot is now a REAL, distinct domain (hoist/crate/pallet/truck),
     not a relabeled copy of Gripper.
  2. Every domain's n_steps comes from an actual BFS search under the
     SAME explicit budget B=(T_max=12, N_max=5000) -- a step-depth cap
     AND a node-expansion cap. No domain uses a closed-form formula
     (n_sats*5+..., n_rovers*6+...) or a greedy heuristic (old Gripper)
     as a stand-in for a real plan length.
  3. On BFS failure (budget exhausted with no goal state found):
     n_steps = 0, valid = False. NOT a random integer. Every
     downstream consumer of these labels MUST filter to valid==True
     before computing any correlation, regression, or baseline.
  4. Instance descriptions contain ONLY pre-planning structural facts
     (domain name, objects, initial state, goal) -- no step count, no
     difficulty adjective, nothing derived from n_steps. This removes
     the target leak into x_e (and therefore x_r).
  5. The domain-identity hash feature (old x_s[14]) is removed from
     the syntactic feature extractor.
  6. Reproducibility: action lists are built from SORTED iteration over
     sets (Depot surfaces, Rovers traversals). Previously set order
     depended on PYTHONHASHSEED, so plan strings (and potentially
     borderline budget outcomes) could differ between regenerations.

USAGE:
  python plan_step1_data_prep_v3.py --n_per_level 40 --force
"""
import argparse
import json
import re
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/planning_v3")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Uniform search budget (matches the paper's stated B=(T_max,N_max))
T_MAX = 12       # max plan depth (steps)
N_MAX = 5000     # max states expanded before declaring failure

TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
VAL_DOMAINS   = ["gripper"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
ALL_DOMAINS   = TEST_DOMAINS + VAL_DOMAINS + TRAIN_DOMAINS


# ============================================================================
# Domain PDDL definitions
# ============================================================================

BLOCKSWORLD_DOMAIN = """
(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x))
  (:action pick-up :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))
  (:action put-down :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (handempty) (ontable ?x) (clear ?x)))
  (:action stack :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (handempty) (on ?x ?y) (clear ?x)))
  (:action unstack :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (not (on ?x ?y)) (not (handempty)) (holding ?x) (clear ?y))))
"""

MYSTERY_BW_DOMAIN = """
(define (domain mystery_blocksworld)
  (:requirements :strips)
  (:predicates (stacked ?x ?y) (grounded ?x) (apex ?x) (grasping) (clutching ?x))
  (:action grasp :parameters (?x)
    :precondition (and (apex ?x) (grounded ?x) (grasping))
    :effect (and (not (grounded ?x)) (not (apex ?x)) (not (grasping)) (clutching ?x)))
  (:action release :parameters (?x)
    :precondition (clutching ?x)
    :effect (and (not (clutching ?x)) (grasping) (grounded ?x) (apex ?x)))
  (:action place :parameters (?x ?y)
    :precondition (and (clutching ?x) (apex ?y))
    :effect (and (not (clutching ?x)) (not (apex ?y)) (grasping) (stacked ?x ?y) (apex ?x)))
  (:action lift :parameters (?x ?y)
    :precondition (and (stacked ?x ?y) (apex ?x) (grasping))
    :effect (and (not (stacked ?x ?y)) (not (grasping)) (clutching ?x) (apex ?y))))
"""

GRIPPER_DOMAIN = """
(define (domain gripper)
  (:requirements :strips)
  (:predicates (room ?r) (ball ?b) (gripperobj ?g) (at-robby ?r)
               (at ?b ?r) (free ?g) (carry ?o ?g))
  (:action move :parameters (?from ?to)
    :precondition (and (room ?from) (room ?to) (at-robby ?from))
    :effect (and (at-robby ?to) (not (at-robby ?from))))
  (:action pick :parameters (?obj ?room ?gripperobj)
    :precondition (and (ball ?obj) (room ?room) (gripperobj ?gripperobj)
                       (at ?obj ?room) (at-robby ?room) (free ?gripperobj))
    :effect (and (carry ?obj ?gripperobj) (not (at ?obj ?room)) (not (free ?gripperobj))))
  (:action drop :parameters (?obj ?room ?gripperobj)
    :precondition (and (ball ?obj) (room ?room) (gripperobj ?gripperobj)
                       (carry ?obj ?gripperobj) (at-robby ?room))
    :effect (and (at ?obj ?room) (free ?gripperobj) (not (carry ?obj ?gripperobj)))))
"""

DEPOT_DOMAIN = """
(define (domain depot)
  (:requirements :strips)
  (:predicates
    (depotloc ?d) (truck ?t) (hoist ?h) (crate ?c) (pallet ?p)
    (at-truck ?t ?d) (at-hoist ?h ?d) (at-pallet ?p ?d) (at-crate ?c ?d)
    (on ?c ?s) (clear ?s) (in-truck ?c ?t)
    (hoist-available ?h) (hoist-lifting ?h ?c))
  (:action drive :parameters (?t ?from ?to)
    :precondition (and (truck ?t) (depotloc ?from) (depotloc ?to) (at-truck ?t ?from))
    :effect (and (at-truck ?t ?to) (not (at-truck ?t ?from))))
  (:action lift :parameters (?h ?c ?s ?d)
    :precondition (and (hoist ?h) (crate ?c) (at-hoist ?h ?d)
                       (on ?c ?s) (clear ?c) (hoist-available ?h))
    :effect (and (hoist-lifting ?h ?c) (not (hoist-available ?h))
                 (not (on ?c ?s)) (clear ?s) (not (clear ?c))))
  (:action drop :parameters (?h ?c ?s ?d)
    :precondition (and (hoist ?h) (crate ?c) (at-hoist ?h ?d)
                       (hoist-lifting ?h ?c) (clear ?s))
    :effect (and (hoist-available ?h) (not (hoist-lifting ?h ?c))
                 (on ?c ?s) (not (clear ?s)) (clear ?c)))
  (:action load :parameters (?h ?c ?t ?d)
    :precondition (and (hoist ?h) (crate ?c) (truck ?t)
                       (at-hoist ?h ?d) (at-truck ?t ?d) (hoist-lifting ?h ?c))
    :effect (and (in-truck ?c ?t) (hoist-available ?h) (not (hoist-lifting ?h ?c))))
  (:action unload :parameters (?h ?c ?t ?d)
    :precondition (and (hoist ?h) (crate ?c) (truck ?t)
                       (at-hoist ?h ?d) (at-truck ?t ?d) (in-truck ?c ?t)
                       (hoist-available ?h))
    :effect (and (hoist-lifting ?h ?c) (not (in-truck ?c ?t)) (not (hoist-available ?h)))))
"""

LOGISTICS_DOMAIN = """
(define (domain logistics)
  (:requirements :strips :typing)
  (:predicates (in-city ?loc ?city) (at ?obj ?loc) (in ?pkg ?veh))
  (:action load-truck :parameters (?pkg ?truck ?loc)
    :precondition (and (at ?truck ?loc) (at ?pkg ?loc))
    :effect (and (not (at ?pkg ?loc)) (in ?pkg ?truck)))
  (:action unload-truck :parameters (?pkg ?truck ?loc)
    :precondition (and (at ?truck ?loc) (in ?pkg ?truck))
    :effect (and (not (in ?pkg ?truck)) (at ?pkg ?loc)))
  (:action drive-truck :parameters (?truck ?from ?to ?city)
    :precondition (and (at ?truck ?from) (in-city ?from ?city) (in-city ?to ?city))
    :effect (and (not (at ?truck ?from)) (at ?truck ?to))))
"""

SATELLITE_DOMAIN = """
(define (domain satellite)
  (:requirements :strips)
  (:predicates (satellite ?s) (direction ?d) (instrument ?i) (mode ?m)
               (pointing ?s ?d) (on-board ?i ?s) (supports ?i ?m)
               (power-avail ?s) (power-on ?i) (calibrated ?i)
               (have-image ?d ?m) (calibration-target ?i ?d))
  (:action turn-to :parameters (?s ?d-new ?d-prev)
    :precondition (and (satellite ?s) (direction ?d-new) (direction ?d-prev) (pointing ?s ?d-prev))
    :effect (and (pointing ?s ?d-new) (not (pointing ?s ?d-prev))))
  (:action switch-on :parameters (?i ?s)
    :precondition (and (instrument ?i) (satellite ?s) (on-board ?i ?s) (power-avail ?s))
    :effect (and (power-on ?i) (not (calibrated ?i)) (not (power-avail ?s))))
  (:action switch-off :parameters (?i ?s)
    :precondition (and (instrument ?i) (satellite ?s) (on-board ?i ?s) (power-on ?i))
    :effect (and (not (power-on ?i)) (power-avail ?s)))
  (:action calibrate :parameters (?s ?i ?d)
    :precondition (and (satellite ?s) (instrument ?i) (direction ?d) (on-board ?i ?s)
                       (calibration-target ?i ?d) (pointing ?s ?d) (power-on ?i))
    :effect (calibrated ?i))
  (:action take-image :parameters (?s ?d ?i ?m)
    :precondition (and (satellite ?s) (direction ?d) (instrument ?i) (mode ?m)
                       (calibrated ?i) (on-board ?i ?s) (supports ?i ?m)
                       (power-on ?i) (pointing ?s ?d))
    :effect (have-image ?d ?m)))
"""

ROVERS_DOMAIN = """
(define (domain rovers)
  (:requirements :strips)
  (:predicates (rover ?r) (waypoint ?w) (store ?s) (camera ?c) (mode ?m)
               (objective ?ob) (at ?r ?w) (at-lander ?w) (can-traverse ?r ?w1 ?w2)
               (equipped-soil ?r) (equipped-imaging ?r) (empty ?s)
               (have-soil ?r ?w) (full ?s) (calibrated ?c ?r) (supports ?c ?m)
               (visible ?w1 ?w2) (have-image ?r ?ob ?m) (communicated-soil ?w)
               (communicated-image ?ob ?m) (on-board ?c ?r) (channel-free ?r))
  (:action navigate :parameters (?r ?from ?to)
    :precondition (and (rover ?r) (waypoint ?from) (waypoint ?to) (at ?r ?from) (can-traverse ?r ?from ?to))
    :effect (and (at ?r ?to) (not (at ?r ?from))))
  (:action sample-soil :parameters (?r ?s ?w)
    :precondition (and (rover ?r) (store ?s) (waypoint ?w) (at ?r ?w) (equipped-soil ?r) (empty ?s))
    :effect (and (full ?s) (not (empty ?s)) (have-soil ?r ?w)))
  (:action communicate-soil :parameters (?r ?lander ?w ?from)
    :precondition (and (rover ?r) (waypoint ?lander) (waypoint ?w) (waypoint ?from)
                       (at-lander ?lander) (have-soil ?r ?w) (at ?r ?from)
                       (visible ?from ?lander) (channel-free ?r))
    :effect (communicated-soil ?w))
  (:action take-image :parameters (?r ?w ?ob ?c ?m)
    :precondition (and (rover ?r) (waypoint ?w) (objective ?ob) (camera ?c) (mode ?m)
                       (at ?r ?w) (calibrated ?c ?r) (on-board ?c ?r) (supports ?c ?m) (equipped-imaging ?r))
    :effect (have-image ?r ?ob ?m)))
"""

DOMAIN_PDDL = {
    "blocksworld": BLOCKSWORLD_DOMAIN,
    "mystery_blocksworld": MYSTERY_BW_DOMAIN,
    "gripper": GRIPPER_DOMAIN,
    "depot": DEPOT_DOMAIN,
    "logistics": LOGISTICS_DOMAIN,
    "satellite": SATELLITE_DOMAIN,
    "rovers": ROVERS_DOMAIN,
}

COMPLEXITY_N = {
    "blocksworld": [2, 3, 4, 6, 8],
    "mystery_blocksworld": [2, 3, 4, 6, 8],
    "gripper": [2, 3, 4, 6, 8],
    "depot": [2, 3, 4, 5, 6],
    "logistics": [2, 3, 4, 5, 6],
    "satellite": [1, 2, 3, 4, 5],
    "rovers": [1, 2, 3, 4, 5],
}


# ============================================================================
# Uniform BFS solver with an explicit dual (step, node) budget.
# EVERY domain's n_steps comes from this single function. No formulas.
# ============================================================================

class STRIPSState:
    __slots__ = ("facts",)
    def __init__(self, facts):
        self.facts = frozenset(facts)
    def satisfies(self, conditions):
        return conditions <= self.facts
    def apply(self, add_list, del_list):
        return STRIPSState((self.facts - del_list) | add_list)
    def __hash__(self): return hash(self.facts)
    def __eq__(self, o): return self.facts == o.facts


def bfs_plan(initial_facts, goal_facts, actions, t_max=T_MAX, n_max=N_MAX):
    """
    Uniform-budget BFS planner used for ALL domains.
    actions: list of dicts with keys pre (frozenset), add (frozenset), del (frozenset), name (str)
    Returns (plan, n_steps, valid). valid=False and n_steps=0 if the goal
    is not reached within t_max plan-depth AND n_max expanded states.
    NO random fallback, NO formula fallback, under any circumstance.
    """
    start = STRIPSState(initial_facts)
    goal = frozenset(goal_facts)

    if goal <= start.facts:
        return [], 0, True

    queue = deque([(start, [])])
    visited = {start}
    expanded = 0

    while queue and expanded < n_max:
        state, plan = queue.popleft()
        expanded += 1
        if len(plan) >= t_max:
            continue
        for action in actions:
            if state.satisfies(action["pre"]):
                new_state = state.apply(action["add"], action["del"])
                if new_state in visited:
                    continue
                new_plan = plan + [action["name"]]
                if goal <= new_state.facts:
                    return new_plan, len(new_plan), True
                visited.add(new_state)
                queue.append((new_state, new_plan))

    return None, 0, False


def build_description(domain, objects_summary, init_summary, goal_summary):
    return (f"Domain: {domain}. Objects: {objects_summary}. "
            f"Initial state: {init_summary}. Goal: {goal_summary}.")


def gen_blocksworld_problem(n_blocks, rng, domain="blocksworld"):
    blocks = [f"b{i}" for i in range(1, n_blocks + 1)]
    perm = list(rng.permutation(blocks))
    init = {"ontable": set(), "on": set(), "clear": set(), "handempty": True}
    placed = []
    for b in perm:
        if placed and rng.random() < 0.4:
            base = rng.choice(placed)
            init["on"].add((b, base))
            init["clear"].discard(base)
        else:
            init["ontable"].add(b)
        init["clear"].add(b)
        placed.append(b)

    n_goals = min(max(1, n_blocks // 2), 3)
    goal_pairs = set()
    attempts = 0
    while len(goal_pairs) < n_goals and attempts < 50:
        a, b = rng.choice(blocks, 2, replace=False)
        goal_pairs.add((a, b))
        attempts += 1

    if domain == "mystery_blocksworld":
        pred_map = {"on": "stacked", "ontable": "grounded", "clear": "apex", "handempty": "grasping"}
    else:
        pred_map = {"on": "on", "ontable": "ontable", "clear": "clear", "handempty": "handempty"}
    on_p, ot_p, cl_p, he_p = pred_map["on"], pred_map["ontable"], pred_map["clear"], pred_map["handempty"]

    init_facts = set()
    for b in init["ontable"]: init_facts.add(f"({ot_p} {b})")
    for (a, b) in init["on"]: init_facts.add(f"({on_p} {a} {b})")
    for b in init["clear"]: init_facts.add(f"({cl_p} {b})")
    init_facts.add(f"({he_p})")
    goal_facts = {f"({on_p} {a} {b})" for a, b in goal_pairs}

    holding = lambda x: f"(holding {x})" if domain == "blocksworld" else f"(clutching {x})"
    actions = []
    for x in blocks:
        actions.append({"name": f"(pick-up {x})", "pre": frozenset({f"({cl_p} {x})", f"({ot_p} {x})", f"({he_p})"}),
                        "add": frozenset({holding(x)}), "del": frozenset({f"({ot_p} {x})", f"({cl_p} {x})", f"({he_p})"})})
        actions.append({"name": f"(put-down {x})", "pre": frozenset({holding(x)}),
                        "add": frozenset({f"({ot_p} {x})", f"({cl_p} {x})", f"({he_p})"}), "del": frozenset({holding(x)})})
        for y in blocks:
            if x == y: continue
            actions.append({"name": f"(stack {x} {y})",
                            "pre": frozenset({holding(x), f"({cl_p} {y})"}),
                            "add": frozenset({f"({on_p} {x} {y})", f"({cl_p} {x})", f"({he_p})"}),
                            "del": frozenset({holding(x), f"({cl_p} {y})"})})
            actions.append({"name": f"(unstack {x} {y})",
                            "pre": frozenset({f"({on_p} {x} {y})", f"({cl_p} {x})", f"({he_p})"}),
                            "add": frozenset({holding(x), f"({cl_p} {y})"}),
                            "del": frozenset({f"({on_p} {x} {y})", f"({cl_p} {x})", f"({he_p})"})})

    plan, n_steps, valid = bfs_plan(init_facts, goal_facts, actions)

    unsat = sum(1 for g in goal_facts if g not in init_facts)
    description = build_description(
        domain, f"{n_blocks} blocks",
        f"{unsat} of {len(goal_facts)} goal relation(s) not yet satisfied; "
        f"{len(init['ontable'])} block(s) on the table",
        f"achieve {len(goal_facts)} stacking relation(s)")

    problem_pddl = (f"(define (problem bw-n{n_blocks}) (:domain {domain}) "
                     f"(:objects {' '.join(blocks)}) (:init {' '.join(sorted(init_facts))}) "
                     f"(:goal (and {' '.join(sorted(goal_facts))})))")

    return {"domain": domain, "n_objects": n_blocks, "problem_pddl": problem_pddl,
            "domain_pddl": DOMAIN_PDDL[domain].strip(), "plan": plan or [],
            "n_steps": n_steps, "valid": valid, "description": description,
            "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts)}


def gen_gripper_problem(n_balls, rng):
    n_rooms = max(2, n_balls // 2)
    rooms = [f"room{i}" for i in range(1, n_rooms + 1)]
    balls = [f"ball{i}" for i in range(1, n_balls + 1)]
    grippers = ["left", "right"]
    init_room = rooms[0]
    ball_locs = {b: rng.choice(rooms) for b in balls}
    goal_room = rooms[-1]
    goal_balls = list(rng.choice(balls, min(n_balls, 3), replace=False))

    init_facts = ({f"(room {r})" for r in rooms} | {f"(ball {b})" for b in balls}
                  | {f"(gripperobj {g})" for g in grippers} | {f"(at-robby {init_room})"}
                  | {f"(at {b} {ball_locs[b]})" for b in balls} | {f"(free {g})" for g in grippers})
    goal_facts = {f"(at {b} {goal_room})" for b in goal_balls}

    actions = []
    for r_from in rooms:
        for r_to in rooms:
            if r_from == r_to: continue
            actions.append({"name": f"(move {r_from} {r_to})",
                            "pre": frozenset({f"(room {r_from})", f"(room {r_to})", f"(at-robby {r_from})"}),
                            "add": frozenset({f"(at-robby {r_to})"}), "del": frozenset({f"(at-robby {r_from})"})})
    for b in balls:
        for r in rooms:
            for g in grippers:
                actions.append({"name": f"(pick {b} {r} {g})",
                                "pre": frozenset({f"(at {b} {r})", f"(at-robby {r})", f"(free {g})"}),
                                "add": frozenset({f"(carry {b} {g})"}), "del": frozenset({f"(at {b} {r})", f"(free {g})"})})
                actions.append({"name": f"(drop {b} {r} {g})",
                                "pre": frozenset({f"(carry {b} {g})", f"(at-robby {r})"}),
                                "add": frozenset({f"(at {b} {r})", f"(free {g})"}), "del": frozenset({f"(carry {b} {g})"})})

    plan, n_steps, valid = bfs_plan(init_facts, goal_facts, actions)

    unsat = sum(1 for b in goal_balls if ball_locs[b] != goal_room)
    description = build_description(
        "gripper", f"{n_balls} balls, {n_rooms} rooms, a 2-gripper robot",
        f"robot at {init_room}; {unsat} of {len(goal_balls)} target ball(s) not yet at {goal_room}",
        f"move {len(goal_balls)} ball(s) to {goal_room}")
    problem_pddl = (f"(define (problem gripper-n{n_balls}) (:domain gripper) "
                     f"(:objects {' '.join(rooms + balls + grippers)}) "
                     f"(:init {' '.join(sorted(init_facts))}) (:goal (and {' '.join(sorted(goal_facts))})))")

    return {"domain": "gripper", "n_objects": n_balls + n_rooms, "problem_pddl": problem_pddl,
            "domain_pddl": DOMAIN_PDDL["gripper"].strip(), "plan": plan or [], "n_steps": n_steps,
            "valid": valid, "description": description, "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts)}


def gen_depot_problem(n_crates, rng):
    """Genuine Depot: stacking (like Blocksworld) + inter-depot transport
    (like Logistics), via hoists and trucks. Structurally distinct from
    Gripper, which has no stacking at all.

    Correctness note: lift/drop require the crate AND the target surface
    to be co-located with the hoist at the same depot (via at-pallet /
    at-crate location facts). Without this check, the BFS solver can
    "teleport" a crate onto a pallet at a different depot without ever
    using a truck, which silently defeats the entire multi-dimensional
    difficulty design of this domain -- caught and fixed here."""
    n_depots = max(2, n_crates // 2)
    depots = [f"depot{i}" for i in range(1, n_depots + 1)]
    pallets = [f"pallet{i}" for i in range(1, n_depots + 1)]
    hoists = [f"hoist{i}" for i in range(1, n_depots + 1)]
    trucks = [f"truck{i}" for i in range(1, max(1, n_depots // 2) + 1)]
    crates = [f"crate{i}" for i in range(1, n_crates + 1)]

    crate_depot = {c: depots[i % n_depots] for i, c in enumerate(crates)}
    pallet_of = {depots[i]: pallets[i] for i in range(n_depots)}
    pallet_depot = {pallet_of[d]: d for d in depots}  # pallet -> its home depot

    init_facts = set()
    for d in depots: init_facts.add(f"(depotloc {d})")
    for t in trucks: init_facts.add(f"(truck {t})"); init_facts.add(f"(at-truck {t} {depots[0]})")
    for i, h in enumerate(hoists):
        init_facts.add(f"(hoist {h})"); init_facts.add(f"(at-hoist {h} {depots[i]})")
        init_facts.add(f"(hoist-available {h})")
    for i, p in enumerate(pallets):
        init_facts.add(f"(pallet {p})"); init_facts.add(f"(at-pallet {p} {depots[i]})")
    for c in crates:
        init_facts.add(f"(crate {c})")
        init_facts.add(f"(on {c} {pallet_of[crate_depot[c]]})")
        init_facts.add(f"(at-crate {c} {crate_depot[c]})")  # explicit crate location
    occupied_surfaces = {pallet_of[crate_depot[c]] for c in crates}
    for p in pallets:
        if p not in occupied_surfaces:
            init_facts.add(f"(clear {p})")
    for c in crates:
        init_facts.add(f"(clear {c})")

    n_goal_crates = min(n_crates, max(1, n_crates // 2))
    goal_crates = list(rng.choice(crates, n_goal_crates, replace=False))
    goal_facts = set()
    for c in goal_crates:
        other_depots = [d for d in depots if d != crate_depot[c]]
        if not other_depots:
            continue
        target_depot = rng.choice(other_depots)
        goal_facts.add(f"(on {c} {pallet_of[target_depot]})")
    if not goal_facts:
        c0 = crates[0]
        target = [d for d in depots if d != crate_depot[c0]][0]
        goal_facts.add(f"(on {c0} {pallet_of[target]})")

    def surface_loc_fact(s, d):
        """The fact asserting surface s is located at depot d, whichever
        kind of surface it is. This is what the original buggy version
        omitted from lift/drop preconditions."""
        if s in pallet_depot:
            return f"(at-pallet {s} {d})"
        return f"(at-crate {s} {d})"

    actions = []
    for t in trucks:
        for d_from in depots:
            for d_to in depots:
                if d_from == d_to: continue
                actions.append({"name": f"(drive {t} {d_from} {d_to})",
                                "pre": frozenset({f"(truck {t})", f"(depotloc {d_from})", f"(depotloc {d_to})", f"(at-truck {t} {d_from})"}),
                                "add": frozenset({f"(at-truck {t} {d_to})"}), "del": frozenset({f"(at-truck {t} {d_from})"})})
    all_surfaces = sorted(set(pallets) | set(crates))  # sorted: set order is hash-seed dependent
    for i, h in enumerate(hoists):
        d = depots[i]
        for c in crates:
            for s in all_surfaces:
                if s == c: continue
                # FIX: require the surface to actually be at this depot
                loc_fact = surface_loc_fact(s, d)
                actions.append({"name": f"(lift {h} {c} {s} {d})",
                                "pre": frozenset({f"(hoist {h})", f"(crate {c})", f"(at-hoist {h} {d})", f"(on {c} {s})",
                                                   f"(clear {c})", f"(hoist-available {h})", loc_fact, f"(at-crate {c} {d})"}),
                                "add": frozenset({f"(hoist-lifting {h} {c})", f"(clear {s})"}),
                                "del": frozenset({f"(hoist-available {h})", f"(on {c} {s})", f"(clear {c})", f"(at-crate {c} {d})"})})
                actions.append({"name": f"(drop {h} {c} {s} {d})",
                                "pre": frozenset({f"(hoist {h})", f"(crate {c})", f"(at-hoist {h} {d})",
                                                   f"(hoist-lifting {h} {c})", f"(clear {s})", loc_fact}),
                                "add": frozenset({f"(hoist-available {h})", f"(on {c} {s})", f"(clear {c})", f"(at-crate {c} {d})"}),
                                "del": frozenset({f"(hoist-lifting {h} {c})", f"(clear {s})"})})
        for c in crates:
            for t in trucks:
                actions.append({"name": f"(load {h} {c} {t} {d})",
                                "pre": frozenset({f"(hoist {h})", f"(crate {c})", f"(truck {t})", f"(at-hoist {h} {d})", f"(at-truck {t} {d})", f"(hoist-lifting {h} {c})"}),
                                "add": frozenset({f"(in-truck {c} {t})", f"(hoist-available {h})"}),
                                "del": frozenset({f"(hoist-lifting {h} {c})"})})
                actions.append({"name": f"(unload {h} {c} {t} {d})",
                                "pre": frozenset({f"(hoist {h})", f"(crate {c})", f"(truck {t})", f"(at-hoist {h} {d})", f"(at-truck {t} {d})", f"(in-truck {c} {t})", f"(hoist-available {h})"}),
                                "add": frozenset({f"(hoist-lifting {h} {c})"}),
                                "del": frozenset({f"(in-truck {c} {t})", f"(hoist-available {h})"})})

    plan, n_steps, valid = bfs_plan(init_facts, goal_facts, actions)

    description = build_description(
        "depot", f"{n_crates} crates, {n_depots} depots, {len(trucks)} truck(s), {len(hoists)} hoist(s)",
        "each crate starts on its home depot's pallet",
        f"relocate {len(goal_facts)} crate(s) to a different depot's pallet (requires truck transport)")
    problem_pddl = (f"(define (problem depot-n{n_crates}) (:domain depot) "
                     f"(:objects {' '.join(depots+pallets+hoists+trucks+crates)}) "
                     f"(:init {' '.join(sorted(init_facts))}) (:goal (and {' '.join(sorted(goal_facts))})))")

    return {"domain": "depot", "n_objects": n_crates + n_depots, "problem_pddl": problem_pddl,
            "domain_pddl": DOMAIN_PDDL["depot"].strip(), "plan": plan or [], "n_steps": n_steps,
            "valid": valid, "description": description, "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts)}


def gen_logistics_problem(n_packages, rng):
    n_cities = max(2, n_packages // 2)
    cities = [f"city{i}" for i in range(1, n_cities + 1)]
    trucks = [f"truck{i}" for i in range(1, n_cities + 1)]
    packages = [f"pkg{i}" for i in range(1, n_packages + 1)]
    pkg_locs = {p: rng.choice(cities) for p in packages}
    pkg_goals = {p: rng.choice(cities) for p in packages}
    truck_locs = {t: cities[i] for i, t in enumerate(trucks)}

    init_facts = ({f"(in-city {c} {c})" for c in cities}
                  | {f"(at {t} {truck_locs[t]})" for t in trucks}
                  | {f"(at {p} {pkg_locs[p]})" for p in packages})
    goal_facts = {f"(at {p} {pkg_goals[p]})" for p in packages if pkg_locs[p] != pkg_goals[p]}
    if not goal_facts:
        goal_facts = {f"(at {packages[0]} {cities[-1]})"}

    actions = []
    for p in packages:
        for t in trucks:
            for c in cities:
                actions.append({"name": f"(load-truck {p} {t} {c})",
                                "pre": frozenset({f"(at {t} {c})", f"(at {p} {c})"}),
                                "add": frozenset({f"(in {p} {t})"}), "del": frozenset({f"(at {p} {c})"})})
                actions.append({"name": f"(unload-truck {p} {t} {c})",
                                "pre": frozenset({f"(at {t} {c})", f"(in {p} {t})"}),
                                "add": frozenset({f"(at {p} {c})"}), "del": frozenset({f"(in {p} {t})"})})
    for t in trucks:
        for c_from in cities:
            for c_to in cities:
                if c_from == c_to: continue
                actions.append({"name": f"(drive-truck {t} {c_from} {c_to})",
                                "pre": frozenset({f"(at {t} {c_from})", f"(in-city {c_from} {c_from})", f"(in-city {c_to} {c_to})"}),
                                "add": frozenset({f"(at {t} {c_to})"}), "del": frozenset({f"(at {t} {c_from})"})})

    plan, n_steps, valid = bfs_plan(init_facts, goal_facts, actions)

    description = build_description(
        "logistics", f"{n_packages} packages, {n_cities} cities, {len(trucks)} trucks",
        f"{len(packages)-len(goal_facts)} package(s) already at goal",
        f"deliver {len(goal_facts)} package(s) to their destination cities")
    problem_pddl = (f"(define (problem log-n{n_packages}) (:domain logistics) "
                     f"(:objects {' '.join(trucks+cities+packages)}) "
                     f"(:init {' '.join(sorted(init_facts))}) (:goal (and {' '.join(sorted(goal_facts))})))")

    return {"domain": "logistics", "n_objects": n_packages + n_cities, "problem_pddl": problem_pddl,
            "domain_pddl": DOMAIN_PDDL["logistics"].strip(), "plan": plan or [], "n_steps": n_steps,
            "valid": valid, "description": description, "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts)}


def gen_satellite_problem(n_sats, rng):
    n_dir = max(3, n_sats * 2)
    sats = [f"sat{i}" for i in range(1, n_sats + 1)]
    dirs = [f"dir{i}" for i in range(1, n_dir + 1)]
    instr = [f"inst{i}" for i in range(1, n_sats + 1)]
    modes = ["mode0"]
    calib = {ins: rng.choice(dirs) for ins in instr}
    goals = [(rng.choice(dirs), modes[0]) for _ in range(min(n_sats, 2))]

    init_facts = ({f"(satellite {s})" for s in sats} | {f"(direction {d})" for d in dirs}
                  | {f"(instrument {i})" for i in instr} | {f"(mode {m})" for m in modes}
                  | {f"(on-board {instr[i]} {sats[i]})" for i in range(n_sats)}
                  | {f"(supports {ins} {modes[0]})" for ins in instr}
                  | {f"(power-avail {s})" for s in sats}
                  | {f"(pointing {s} {rng.choice(dirs)})" for s in sats}
                  | {f"(calibration-target {ins} {calib[ins]})" for ins in instr})
    goal_facts = {f"(have-image {d} {m})" for d, m in goals}

    actions = []
    for s in sats:
        for d_prev in dirs:
            for d_new in dirs:
                if d_prev == d_new: continue
                actions.append({"name": f"(turn-to {s} {d_new} {d_prev})",
                                "pre": frozenset({f"(satellite {s})", f"(direction {d_new})", f"(direction {d_prev})", f"(pointing {s} {d_prev})"}),
                                "add": frozenset({f"(pointing {s} {d_new})"}), "del": frozenset({f"(pointing {s} {d_prev})"})})
    for i, ins in enumerate(instr):
        s = sats[i]
        actions.append({"name": f"(switch-on {ins} {s})",
                        "pre": frozenset({f"(instrument {ins})", f"(satellite {s})", f"(on-board {ins} {s})", f"(power-avail {s})"}),
                        "add": frozenset({f"(power-on {ins})"}), "del": frozenset({f"(power-avail {s})"})})
        for d in dirs:
            if calib[ins] != d: continue
            actions.append({"name": f"(calibrate {s} {ins} {d})",
                            "pre": frozenset({f"(satellite {s})", f"(instrument {ins})", f"(direction {d})", f"(on-board {ins} {s})", f"(calibration-target {ins} {d})", f"(pointing {s} {d})", f"(power-on {ins})"}),
                            "add": frozenset({f"(calibrated {ins})"}), "del": frozenset()})
        for d in dirs:
            for m in modes:
                actions.append({"name": f"(take-image {s} {d} {ins} {m})",
                                "pre": frozenset({f"(satellite {s})", f"(direction {d})", f"(instrument {ins})", f"(mode {m})", f"(calibrated {ins})", f"(on-board {ins} {s})", f"(supports {ins} {m})", f"(power-on {ins})", f"(pointing {s} {d})"}),
                                "add": frozenset({f"(have-image {d} {m})"}), "del": frozenset()})

    plan, n_steps, valid = bfs_plan(init_facts, goal_facts, actions)

    description = build_description(
        "satellite", f"{n_sats} satellite(s), {n_dir} directions, {len(instr)} instrument(s)",
        "instruments uncalibrated and powered off",
        f"capture images of {len(goals)} target direction(s)")
    problem_pddl = (f"(define (problem sat-n{n_sats}) (:domain satellite) "
                     f"(:objects {' '.join(sats+dirs+instr+modes)}) "
                     f"(:init {' '.join(sorted(init_facts))}) (:goal (and {' '.join(sorted(goal_facts))})))")

    return {"domain": "satellite", "n_objects": n_sats + n_dir, "problem_pddl": problem_pddl,
            "domain_pddl": DOMAIN_PDDL["satellite"].strip(), "plan": plan or [], "n_steps": n_steps,
            "valid": valid, "description": description, "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts)}


def gen_rovers_problem(n_rovers, rng):
    n_wp = max(3, n_rovers * 2)
    rovers = [f"rover{i}" for i in range(1, n_rovers + 1)]
    wps = [f"wp{i}" for i in range(1, n_wp + 1)]
    stores = [f"store{i}" for i in range(1, n_rovers + 1)]
    cams = [f"cam{i}" for i in range(1, n_rovers + 1)]
    modes = ["colour"]
    lander_wp = wps[0]
    rover_wps = {r: rng.choice(wps) for r in rovers}
    traversals = set()
    for wp in wps:
        targets = rng.choice(wps, min(2, n_wp), replace=False)
        for t in targets:
            if t != wp:
                traversals.add((wp, t)); traversals.add((t, wp))
    soil_goals = [rng.choice(wps)]

    init_facts = ({f"(rover {r})" for r in rovers} | {f"(waypoint {w})" for w in wps}
                  | {f"(store {s})" for s in stores} | {f"(camera {c})" for c in cams}
                  | {f"(mode {m})" for m in modes} | {f"(at {r} {rover_wps[r]})" for r in rovers}
                  | {f"(at-lander {lander_wp})"}
                  | {f"(can-traverse {r} {a} {b})" for r in rovers for a, b in traversals}
                  | {f"(equipped-soil {r})" for r in rovers} | {f"(equipped-imaging {r})" for r in rovers}
                  | {f"(empty {stores[i]})" for i in range(n_rovers)}
                  | {f"(on-board {cams[i]} {rovers[i]})" for i in range(n_rovers)}
                  | {f"(supports {cams[i]} {modes[0]})" for i in range(n_rovers)}
                  | {f"(calibrated {cams[i]} {rovers[i]})" for i in range(n_rovers)}
                  | {f"(visible {a} {lander_wp})" for a, _ in traversals if lander_wp != a}
                  | {f"(channel-free {r})" for r in rovers})
    goal_facts = {f"(communicated-soil {w})" for w in soil_goals}

    actions = []
    for r in rovers:
        for a, b in sorted(traversals):  # sorted: set order is hash-seed dependent
            actions.append({"name": f"(navigate {r} {a} {b})",
                            "pre": frozenset({f"(rover {r})", f"(waypoint {a})", f"(waypoint {b})", f"(at {r} {a})", f"(can-traverse {r} {a} {b})"}),
                            "add": frozenset({f"(at {r} {b})"}), "del": frozenset({f"(at {r} {a})"})})
    for i, r in enumerate(rovers):
        s = stores[i]
        for w in wps:
            actions.append({"name": f"(sample-soil {r} {s} {w})",
                            "pre": frozenset({f"(rover {r})", f"(store {s})", f"(waypoint {w})", f"(at {r} {w})", f"(equipped-soil {r})", f"(empty {s})"}),
                            "add": frozenset({f"(full {s})", f"(have-soil {r} {w})"}), "del": frozenset({f"(empty {s})"})})
        for w in wps:
            for w_from in wps:
                actions.append({"name": f"(communicate-soil {r} {lander_wp} {w} {w_from})",
                                "pre": frozenset({f"(rover {r})", f"(waypoint {lander_wp})", f"(waypoint {w})", f"(waypoint {w_from})",
                                                   f"(at-lander {lander_wp})", f"(have-soil {r} {w})", f"(at {r} {w_from})",
                                                   f"(visible {w_from} {lander_wp})", f"(channel-free {r})"}),
                                "add": frozenset({f"(communicated-soil {w})"}), "del": frozenset()})

    plan, n_steps, valid = bfs_plan(init_facts, goal_facts, actions)

    description = build_description(
        "rovers", f"{n_rovers} rover(s), {n_wp} waypoints",
        "rovers positioned at their starting waypoints",
        f"collect and communicate soil data from {len(soil_goals)} waypoint(s)")
    problem_pddl = (f"(define (problem rov-n{n_rovers}) (:domain rovers) "
                     f"(:objects {' '.join(rovers+wps+stores+cams+modes)}) "
                     f"(:init {' '.join(sorted(init_facts))}) (:goal (and {' '.join(sorted(goal_facts))})))")

    return {"domain": "rovers", "n_objects": n_rovers + n_wp, "problem_pddl": problem_pddl,
            "domain_pddl": DOMAIN_PDDL["rovers"].strip(), "plan": plan or [], "n_steps": n_steps,
            "valid": valid, "description": description, "goal_facts": sorted(goal_facts), "init_facts": sorted(init_facts)}


GENERATORS = {
    "blocksworld": lambda n, rng: gen_blocksworld_problem(n, rng, "blocksworld"),
    "mystery_blocksworld": lambda n, rng: gen_blocksworld_problem(n, rng, "mystery_blocksworld"),
    "gripper": gen_gripper_problem,
    "depot": gen_depot_problem,
    "logistics": gen_logistics_problem,
    "satellite": gen_satellite_problem,
    "rovers": gen_rovers_problem,
}


def compute_surface_features(records):
    """28-dim syntactic features. Domain-identity hash REMOVED (fix #5)."""
    N = len(records)
    feats = np.zeros((N, 28), dtype=np.float32)
    for i, r in enumerate(records):
        desc = r["description"]; words = desc.lower().split()
        problem = r.get("problem_pddl", ""); init_f = r.get("init_facts", []); goal_f = r.get("goal_facts", [])
        feats[i, 0] = float(r.get("n_objects", 0))
        feats[i, 1] = float(len(init_f))
        feats[i, 2] = float(len(goal_f))
        feats[i, 3] = float(len(goal_f)) / max(float(len(init_f)), 1)
        feats[i, 4] = float(r.get("n_objects", 1)) / max(len(goal_f), 1)
        feats[i, 5] = float(len(words))
        sents = [s.strip() for s in re.split(r'[.!?]', desc) if s.strip()]
        feats[i, 6] = float(len(sents))
        feats[i, 7] = float(len(set(words))) / max(len(words), 1)
        def count_preds(facts):
            preds = set()
            for f in facts:
                m = re.match(r'\((\S+)', f)
                if m: preds.add(m.group(1))
            return len(preds)
        feats[i, 8] = float(count_preds(init_f))
        feats[i, 9] = float(count_preds(goal_f))
        arities = []
        for f in goal_f:
            args = re.findall(r'\??\w+', f.strip("()"))
            arities.append(max(0, len(args) - 1))
        feats[i, 10] = float(np.mean(arities)) if arities else 0
        feats[i, 11] = float(max(arities)) if arities else 0
        types = re.findall(r'- (\w+)', problem)
        feats[i, 12] = float(len(set(types)))
        feats[i, 13] = float(r.get("domain_pddl", "").count(":action"))
        sl = [len(s.split()) for s in sents] or [0]
        feats[i, 14] = float(np.mean(sl))
        feats[i, 15] = float(np.max(sl))
        obj_in_both = len(set(init_f) & set(goal_f))
        feats[i, 16] = float(obj_in_both)
        feats[i, 17] = float(len(problem)) / 500.0
        feats[i, 18] = float(problem.count("not"))
        feats[i, 19] = float(len(goal_f) * r.get("n_objects", 1)) / 10.0
        feats[i, 20] = float(desc.count(","))
        feats[i, 21] = float(len(words) > 40)
        feats[i, 22] = float(len(goal_f) > 2)
        feats[i, 23] = float(len(init_f)) / max(float(r.get("n_objects", 1)), 1)
        feats[i, 24] = float(count_preds(init_f)) / max(float(r.get("n_objects", 1)), 1)
        feats[i, 25] = float(len(set(init_f)))
        feats[i, 26] = float(len(re.findall(r'\d+', problem)))
        feats[i, 27] = float(problem.count("depotloc") > 0)
    return np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=0.0)


def compute_fm_embeddings(records, model_name="all-mpnet-base-v2"):
    descriptions = [r["description"] for r in records]
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_name)
        emb = st_model.encode(descriptions, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        return emb.astype(np.float32)
    except Exception as e:
        print(f"    sentence-transformers failed ({e}), using random fallback (flagged, not silent)")
        rng2 = np.random.default_rng(0)
        X_fm = rng2.normal(0, 1, (len(records), 768)).astype(np.float32)
        X_fm /= np.linalg.norm(X_fm, axis=1, keepdims=True)
        return X_fm


def generate_dataset(domains, n_per_level, rng):
    records = []
    instance_id = 0
    for domain in domains:
        gen = GENERATORS.get(domain)
        if gen is None:
            print(f"  No generator for {domain}, skipping"); continue
        n_levels = COMPLEXITY_N.get(domain, [2, 3, 4, 5, 6])
        for level, n_obj in enumerate(n_levels):
            collected = 0
            attempts = 0
            batch_start = len(records)
            while collected < n_per_level and attempts < n_per_level * 5:
                attempts += 1
                try:
                    ep = gen(n_obj, rng)
                except Exception:
                    continue
                ep["instance_id"] = instance_id
                ep["complexity"] = level
                ep["task_type"] = domain
                instance_id += 1
                records.append(ep)
                collected += 1
            n_valid = sum(1 for r in records[batch_start:] if r["valid"])
            print(f"  {domain} level {level} (n_obj={n_obj}): {collected} generated, "
                  f"{n_valid} BFS-solved within budget")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_level", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    print(f"Generating with uniform budget T_max={T_MAX}, N_max={N_MAX}")
    records = generate_dataset(ALL_DOMAINS, args.n_per_level, rng)

    n_valid_total = sum(1 for r in records if r["valid"])
    print(f"\nTotal: {len(records)}  |  BFS-solved within budget: {n_valid_total} "
          f"({100*n_valid_total/len(records):.1f}%)")
    print("Instances with valid=False have n_steps=0 SENTINEL values and MUST be "
          "excluded from any downstream Spearman/regression computation.")

    ep_path = DATA_DIR / "episodes.json"
    ep_path.write_text(json.dumps(records, indent=2))
    print(f"Saved -> {ep_path}")

    X_surf = compute_surface_features(records)
    np.save(DATA_DIR / "X_surf.npy", X_surf)
    print(f"Surface features: {X_surf.shape} (domain-identity hash removed)")

    X_fm = compute_fm_embeddings(records)
    np.save(DATA_DIR / "X_fm.npy", X_fm)
    print(f"FM embeddings: {X_fm.shape} (descriptions contain no leaked step-count)")

    task_arr = np.array([r["task_type"] for r in records])
    y_nsteps = np.array([r["n_steps"] for r in records], dtype=np.float32)
    y_valid = np.array([r["valid"] for r in records], dtype=bool)
    np.save(DATA_DIR / "y_nsteps.npy", y_nsteps)
    np.save(DATA_DIR / "y_valid.npy", y_valid)
    np.save(DATA_DIR / "task_types.npy", task_arr)

    registry = {
        "n_total": len(records), "surf_dim": int(X_surf.shape[1]), "fm_dim": int(X_fm.shape[1]),
        "t_max": T_MAX, "n_max": N_MAX,
        "test_domains": TEST_DOMAINS, "val_domains": VAL_DOMAINS, "train_domains": TRAIN_DOMAINS,
        "valid_fraction_by_domain": {
            d: float(np.mean(y_valid[task_arr == d])) for d in ALL_DOMAINS if (task_arr == d).any()
        },
    }
    (DATA_DIR / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"\nValid (BFS-solved) fraction by domain:")
    for d, f in registry["valid_fraction_by_domain"].items():
        print(f"  {d:<22} {f*100:.1f}%")
    print(f"\nAll data -> {DATA_DIR}/")


if __name__ == "__main__":
    main()
