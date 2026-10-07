"""
plan_validator.py
==================
Genuine PDDL plan validator: executes actions against the initial
state, checks preconditions, and verifies the goal is reached.
Extracted from plan_step15_qwen72b_eval.py (the validator that
produced the paper's Table 7 numbers: Qwen-72B 23.5% BW / 0% LOG / 22% MBW).

This replaces the syntax-only checkers in plan_step31/plan_step37,
which only checked whether action names came from the domain
vocabulary and could be fooled by syntactically valid but
semantically empty plans (e.g. repeating a legal action forever).

Supports: blocksworld, mystery_blocksworld, logistics.
"""
import re


def _f(*parts):
    return " ".join(str(p).lower() for p in parts)


# Action semantics: name -> (arity, fn(*args) -> (preconditions, remove, add))
_ACTION_SEMANTICS = {
    # blocksworld
    "pick-up":  (1, lambda x:    ({_f("clear",x),_f("ontable",x),_f("handempty")},
                                    {_f("ontable",x),_f("clear",x),_f("handempty")},
                                    {_f("holding",x)})),
    "put-down": (1, lambda x:    ({_f("holding",x)},
                                    {_f("holding",x)},
                                    {_f("handempty"),_f("ontable",x),_f("clear",x)})),
    "stack":    (2, lambda x,y:  ({_f("holding",x),_f("clear",y)},
                                    {_f("holding",x),_f("clear",y)},
                                    {_f("handempty"),_f("on",x,y),_f("clear",x)})),
    "unstack":  (2, lambda x,y:  ({_f("on",x,y),_f("clear",x),_f("handempty")},
                                    {_f("on",x,y),_f("clear",x),_f("handempty")},
                                    {_f("holding",x),_f("clear",y)})),
    # mystery_blocksworld (scrambled predicate/action names)
    "grasp":    (1, lambda x:    ({_f("apex",x),_f("grounded",x),_f("grasping")},
                                    {_f("grounded",x),_f("apex",x),_f("grasping")},
                                    {_f("clutching",x)})),
    "release":  (1, lambda x:    ({_f("clutching",x)},
                                    {_f("clutching",x)},
                                    {_f("grasping"),_f("grounded",x),_f("apex",x)})),
    "place":    (2, lambda x,y:  ({_f("clutching",x),_f("apex",y)},
                                    {_f("clutching",x),_f("apex",y)},
                                    {_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
    "lift":     (2, lambda x,y:  ({_f("stacked",x,y),_f("apex",x),_f("grasping")},
                                    {_f("stacked",x,y),_f("apex",x),_f("grasping")},
                                    {_f("clutching",x),_f("apex",y)})),
    # logistics
    "load-truck":      (3, lambda p,t,l: ({_f("at",t,l),_f("at",p,l)},
                                            {_f("at",p,l)},
                                            {_f("in",p,t)})),
    "unload-truck":    (3, lambda p,t,l: ({_f("at",t,l),_f("in",p,t)},
                                            {_f("in",p,t)},
                                            {_f("at",p,l)})),
    "load-airplane":   (3, lambda p,a,l: ({_f("at",a,l),_f("at",p,l)},
                                            {_f("at",p,l)},
                                            {_f("in",p,a)})),
    "unload-airplane": (3, lambda p,a,l: ({_f("at",a,l),_f("in",p,a)},
                                            {_f("in",p,a)},
                                            {_f("at",p,l)})),
    # NOTE: the v3 Logistics generator (plan_step1_data_prep_v3.gen_logistics_problem) uses a
    # SIMPLIFIED 3-argument drive-truck(?truck ?from ?to), with (in-city L L) as a self-referential
    # fact rather than a separate city object -- the domain PDDL shown to the LLM only ever contains
    # this 3-argument action. The original 4-argument entry below matched the OLD domain used for the
    # now-withdrawn Table 7 numbers and would reject every syntactically correct plan an LLM produces
    # against the CURRENT (v3) domain file with "arity:drive-truck(need 4 got 3)". Replaced with the
    # 3-argument semantics matching the domain file actually used in this round of experiments.
    "drive-truck":     (3, lambda t,c_from,c_to: ({_f("at",t,c_from)},
                                                   {_f("at",t,c_from)},
                                                   {_f("at",t,c_to)})),
    "fly-airplane":    (3, lambda a,s,d: ({_f("at",a,s),_f("airport",s),_f("airport",d)},
                                            {_f("at",a,s)},
                                            {_f("at",a,d)})),
}


_TOKEN_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")  # a clean PDDL-style identifier: letters/digits/hyphens only


def parse_plan_lines(response):
    """
    Extract action tokens from an LLM response.

    BUG FIXED: the previous version treated ANY line starting with "(" as an action call verbatim, and
    ANY other non-empty line whose FIRST WORD merely looked like an identifier as a "bare format" action
    (wrapping the whole line in parens). Since that identifier regex matches almost any capitalized
    English word, ordinary chain-of-thought prose -- "To solve the problem...", "This gives...", "Here is
    the plan:" -- was silently parsed as an attempted action call with a garbage name ("to", "this",
    "here", ...), which validate_plan() then reported as an immediate unknown_action failure on the FIRST
    such line, discarding the real plan that followed. Confirmed directly: a response containing a fully
    correct plan ("(pick-up b2)" then "(stack b2 b1)") inside a code fence, preceded by the sentence "To
    solve the problem...", was marked invalid with reason "unknown_action:to" by the old parser.

    Fix: a line is now accepted as an action call only if EVERY token inside it (not just the first) is a
    clean identifier with no punctuation, which excludes ordinary sentences (which contain commas,
    periods, or multi-word phrases a real PDDL action never has). The unparenthesized "bare format"
    fallback is additionally restricted to lines whose first token is already a KNOWN action name in
    _ACTION_SEMANTICS, since that is the only case where guessing a model skipped its parentheses is
    actually safe -- it can no longer manufacture a new, unknown action name out of prose.
    """
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    actions = []
    for line in lines:
        if line.startswith("(") and line.endswith(")"):
            toks = line[1:-1].split()
            if toks and all(_TOKEN_RE.match(t) for t in toks):
                actions.append(line)
        elif not line.startswith("("):
            toks = line.split()
            if (toks and all(_TOKEN_RE.match(t) for t in toks)
                    and toks[0].lower() in _ACTION_SEMANTICS):
                actions.append(f"({line})")
    return actions


def validate_plan(actions, episode: dict):
    """
    Execute plan against PDDL initial state and check goal.
    Returns (valid: bool, error_type: str|None).

    actions: list of strings like "(pick-up b1)" or "(load-truck p1 t1 l1)"
    episode: dict with 'init_facts' and 'goal_facts' (lists of fact strings)
    """
    if not actions:
        return False, "empty_plan"

    init_facts = episode.get("init_facts", [])
    goal_facts = episode.get("goal_facts", [])
    if not init_facts or not goal_facts:
        # No ground truth available — accept if parse succeeded
        return True, None

    state = {_f(*f.strip().strip("()").split()) for f in init_facts}
    goals = {_f(*g.strip().strip("()").split()) for g in goal_facts}

    for act_str in actions:
        toks = act_str.strip().strip("()").split()
        if not toks:
            continue
        name = toks[0].lower()
        args = [t.lower() for t in toks[1:]]

        if name not in _ACTION_SEMANTICS:
            return False, f"unknown_action:{name}"

        arity, fn = _ACTION_SEMANTICS[name]
        if len(args) != arity:
            return False, f"arity:{name}(need {arity} got {len(args)})"

        pre, rem, add = fn(*args)
        if not pre.issubset(state):
            return False, f"precondition_fail:{name}"

        state = (state - rem) | add

    ok = goals.issubset(state)
    return ok, (None if ok else "goal_not_reached")


def is_valid(response, episode, domain_pddl=None):
    """
    Full pipeline: parse LLM response into plan actions, then
    validate via state simulation. Compatible drop-in replacement
    for the syntax-only is_valid() used in plan_step31/plan_step37.

    Returns (valid: bool, reason: str) matching the existing
    reason vocabulary: "error", "refusal", "empty_plan",
    "unknown_action:X", "precondition_fail:X", "arity:X",
    "goal_not_reached", "valid".
    """
    if not response or response.startswith("ERROR:"):
        return False, "error"

    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b|i cannot|unable to|impossible", txt, re.I):
        return False, "refusal"

    actions = parse_plan_lines(txt)
    if not actions:
        return False, "empty_plan"

    valid, reason = validate_plan(actions, episode)
    return valid, (reason if reason else "valid")
