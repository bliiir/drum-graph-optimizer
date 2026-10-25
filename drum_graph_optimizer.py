#!/usr/bin/env python3
"""Drumming technique + kit layout as a two-layer weighted graph.

No third-party packages. Runs on Python 3.10+.

Inner graph: event-by-event motor states; beam-search for good stickings, groove
handings, lead switches and limb coordination for each musical phrase.
Outer graph: kit configurations + acquired skills; Dijkstra explores practice/
reconfiguration investments. Each reachable node is scored over a repertoire,
including nodes reachable only after a locally unhelpful prerequisite skill.

IMPORTANT: Cost weights are hypotheses, *not* neuroscience findings. Calibrate
with your own recordings, perceived effort, timing error, and body comfort.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import heapq
import itertools
import json
import math
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# THE USER'S WORLD: adjustable, deliberately modest defaults.
# Costs are arbitrary "effort points"; practice expenses are in hours.
# Benefit_factor translates one saved effort point per practice/performance
# hour into practice-hour equivalents. Both scales require calibration.
# ---------------------------------------------------------------------------
DEFAULT_PARAMETERS = {
    # For other starting kits, use e.g. {"ride": -1, "crash": -1}.
    "initial_kit": {},
    # Mark a skill as already acquired only if it is comfortably fluent.
    "initial_skills": [],
    "custom_patterns": [],  # optionally supply rhythmic events as JSON
    "budget_hours": 34.0,
    "benefit_factor": 0.065,
    "horizons_hours": [30, 200, 800],
    "beam_width": 44,
    "max_layout_edits": 2,
    "max_new_skills": 3,
    "movement_weight": 0.24,
    "direction_lead_weight": 0.50,  # left-to-right favors R arriving first
    "wrong_side_weight": 0.23,
    "crossed_arms_penalty": 0.10,
    "quick_repeat_penalty": 0.11,
    "three_strokes_penalty": 0.78,
    "cymbal_hand_switch_penalty": 0.60,
    "lead_switch_penalty_untrained": 0.48,
    "lead_switch_penalty_trained": 0.055,
    # Foot/hand co-ordination: these are *individual calibration guesses*,
    # not fixed neurological laws. Positive = harder, negative = easier.
    "hand_foot_pairing": {
        "L_HH_R_SN_K": 0.04,
        "R_HH_L_SN_K": 0.24,
        "R_RIDE_L_SN_K": 0.18,
        "L_RIDE_R_SN_K": 0.06,
    },
    # Right HH time + left snare / right foot coordination can be developed.
    "pairing_gain_from_training": 0.17,
    # All neural / familiarity weights can be overwritten in a JSON profile.
    # "Untrained" means not yet developed by THIS drummer, not biologically
    # incapable. Lower is better. No sign/hand combination is universal.
    "neural_stroke_penalties": {
        "HH_L": 0.045, "HH_R_untrained": 0.48, "HH_R_trained": 0.065,
        "RIDE_L": 0.10, "RIDE_R_untrained": 0.45, "RIDE_R_trained": 0.065,
        "SN_R": 0.055, "SN_L_untrained": 0.30, "SN_L_trained": 0.055,
        "TOM_L": 0.035, "TOM_R_untrained": 0.18, "TOM_R_trained": 0.045,
        "LEFT_KICK_untrained": 0.70, "LEFT_KICK_trained": 0.085,
        "RIGHT_HAT_FOOT_untrained": 0.50, "RIGHT_HAT_FOOT_trained": 0.085,
        "RIGHT_FILL_INIT_untrained": 0.38, "RIGHT_FILL_INIT_trained": 0.045,
    },
    "skill_hours": {
        "RIGHT_HH": 11.0,
        "RIGHT_RIDE": 8.0,
        "LEFT_SNARE": 9.0,
        "RIGHT_FILL_LEAD": 9.0,
        "LEAD_SWITCH": 6.0,
        "CROSSED_HAND_FOOT": 7.0,
        "LEFT_KICK": 12.0,
        "RIGHT_HAT_FOOT": 10.0,
    },
    # Includes monetary/set-up inconvenience converted into *equivalent*
    # practice hours. E.g., a remote HH pedal costs money, not just time.
    "layout_hours": {
        "hh": 11.0,
        "ride": 2.0,
        "crash": 1.5,
        "floor": 9.0,
        "toms": 6.0,
        "snare": 3.0,
        "kick_foot": 10.0,
        "hat_foot": 8.0,
    },
    "repertoire_weights": {
        "straight_hh": 0.25,
        "syncopated_hh": 0.16,
        "right_ride": 0.18,
        "rightward_fill": 0.16,
        "leftward_fill": 0.07,
        "round_trip_fill": 0.10,
        "groove_to_fill": 0.08,
    },
}

SKILLS = (
    "RIGHT_HH", "RIGHT_RIDE", "LEFT_SNARE", "RIGHT_FILL_LEAD",
    "LEAD_SWITCH", "CROSSED_HAND_FOOT", "LEFT_KICK", "RIGHT_HAT_FOOT",
)
SKILL_BITS = {name: (1 << i) for i, name in enumerate(SKILLS)}
SKILL_REQUIRES = {
    "LEAD_SWITCH": ("RIGHT_FILL_LEAD",),
    "CROSSED_HAND_FOOT": ("RIGHT_HH", "LEFT_SNARE"),
}


@dataclass(frozen=True)
class Kit:
    hh: int = -1      # -1=drummer's left, +1=right
    ride: int = 1
    crash: int = 1
    floor: int = 1
    toms: int = 0     # 0=high tom left, 1=mirrored
    snare: int = 0    # -1=slightly left, 0=middle, +1=slightly right
    kick_foot: int = 1     # kick pedal operated by right/left foot
    hat_foot: int = -1     # hi-hat pedal operated by left/right foot

    def position(self, drum: str) -> tuple[float, float]:
        if drum == "HH": return 2.15 * self.hh, 0.88
        if drum == "RIDE": return 2.10 * self.ride, 1.50
        if drum == "CRASH": return 2.38 * self.crash, 1.87
        if drum == "SN": return 0.43 * self.snare, -0.28
        if drum == "T1": return (0.61 if self.toms else -0.61), 1.30
        if drum == "T2": return (-0.49 if self.toms else 0.49), 1.39
        if drum == "FT": return 1.83 * self.floor, 0.01
        raise ValueError(f"Unknown drum: {drum}")

    def label(self) -> str:
        side = lambda v: "L" if v == -1 else "R"
        return (f"HH:{side(self.hh)} Ride:{side(self.ride)} "
                f"Crash:{side(self.crash)} Floor:{side(self.floor)} "
                f"Toms:{'mirror' if self.toms else 'standard'} "
                f"Snare:{'L' if self.snare == -1 else ('R' if self.snare == 1 else 'C')} "
                f"KickFoot:{side(self.kick_foot)} HatFoot:{side(self.hat_foot)}")


STANDARD_KIT = Kit()


@dataclass(frozen=True)
class State:
    kit: Kit = STANDARD_KIT
    learned: int = 0

    def has(self, skill: str) -> bool:
        return bool(self.learned & SKILL_BITS[skill])

    def label(self) -> str:
        learned = ",".join(s for s in SKILLS if self.has(s)) or "established L lead"
        return f"{self.kit.label()}\nSkills: {learned}"


@dataclass(frozen=True)
class Step:
    hits: tuple[str, ...]   # 0-2 hand-played objects per subdivision
    kick: bool = False      # bass foot: determined by current kit
    hat_pedal: bool = False # optional hi-hat pedal note
    group: str = ""        # start of fill/tom phrase (for lead-switch accounting)


@dataclass(frozen=True)
class Phrase:
    name: str
    steps: tuple[Step, ...]
    weight: float


def fill(*drum_and_counts: tuple[str, int]) -> tuple[Step, ...]:
    steps = []
    for drum, count in drum_and_counts:
        for i in range(count):
            steps.append(Step((drum,), group=drum if i == 0 else ""))
    return tuple(steps)


def groove(cymbal: str, syncopated: bool = False) -> tuple[Step, ...]:
    seq = []
    for i in range(16):   # One bar of 16ths, HH/ride on 8ths
        hits = []
        if i % 2 == 0: hits.append(cymbal)
        if i in (4, 12): hits.append("SN")
        kick = i in ((0, 4, 7, 8, 12, 14) if syncopated else (0, 8))
        seq.append(Step(tuple(hits), kick=kick,
                        hat_pedal=(cymbal == "RIDE" and i in (4, 12))))
    return tuple(seq)


def make_repertoire(weights: dict[str, float], custom_patterns: Optional[list] = None) -> tuple[Phrase, ...]:
    cases = {
        "straight_hh": groove("HH"),
        "syncopated_hh": groove("HH", syncopated=True),
        "right_ride": groove("RIDE", syncopated=True),
        "rightward_fill": fill(("SN", 4), ("T1", 4), ("FT", 4), ("CRASH", 1)),
        "leftward_fill": fill(("FT", 4), ("T2", 4), ("SN", 4), ("CRASH", 1)),
        "round_trip_fill": fill(("SN", 2), ("T1", 2), ("FT", 2), ("T2", 2), ("SN", 2), ("CRASH", 1)),
        "groove_to_fill": groove("HH")[:8] + fill(("SN", 2), ("T1", 2), ("FT", 2), ("CRASH", 1)),
    }
    repertoire = [Phrase(k, cases[k], weights.get(k, 0.0)) for k in cases if weights.get(k, 0) > 0]
    allowed_drums = {"HH", "RIDE", "SN", "T1", "T2", "FT", "CRASH"}
    for entry in custom_patterns or []:
        steps = []
        last_group = None
        for event in entry["events"]:
            if isinstance(event, str):
                tokens = [x.strip().upper() for x in event.split("+")]
                hands = tuple(x for x in tokens if x in allowed_drums)
                unknown = set(tokens) - allowed_drums - {"KICK", "HAT_PEDAL"}
                if unknown: raise ValueError(f"Unknown tokens in custom event: {unknown}")
                # Automatically group successive same-drum notes for fill lead.
                group = hands[0] if len(hands) == 1 and hands[0] not in ("HH", "RIDE") else ""
                start = group if group != last_group else ""
                last_group = group or None
                steps.append(Step(hands, kick="KICK" in tokens,
                                  hat_pedal="HAT_PEDAL" in tokens, group=start))
            else:
                steps.append(Step(tuple(event.get("hits", [])),
                                  kick=bool(event.get("kick", False)),
                                  hat_pedal=bool(event.get("hat_pedal", False)),
                                  group=event.get("group", "")))
        if not steps: raise ValueError(f"Empty custom phrase: {entry['name']}")
        repertoire.append(Phrase(entry["name"], tuple(steps), float(entry.get("weight", 0.1))))
    return tuple(repertoire)


@dataclass(frozen=True)
class Motor:
    # Last played drum and tick for each hand, so "RRLL" gives the right
    # hand time to move during the two L notes.
    right_drum: str = "SN"
    left_drum: str = "SN"
    right_tick: int = -12
    left_tick: int = -12
    last_hand: str = ""
    run_length: int = 0
    cymbal_hand: str = ""
    group_lead: str = ""
    prev_group: str = ""


@dataclass
class Candidate:
    score: float
    motor: Motor
    sticking: tuple[str, ...]
    postures: tuple[str, ...]


def options_for(step: Step):
    # At most two different hand objects on the same beat.
    if len(step.hits) == 0: return ((),)
    if len(step.hits) == 1:
        return (("L",), ("R",))
    if len(step.hits) == 2:
        return (("L", "R"), ("R", "L"))
    raise ValueError("A Step can have at most two simultaneous hand hits")


def neural_penalty(drum: str, hand: str, state: State, params: dict) -> float:
    cost = params["neural_stroke_penalties"]
    if drum == "HH":
        return cost["HH_L"] if hand == "L" else cost["HH_R_trained" if state.has("RIGHT_HH") else "HH_R_untrained"]
    if drum == "RIDE":
        return cost["RIDE_L"] if hand == "L" else cost["RIDE_R_trained" if state.has("RIGHT_RIDE") else "RIDE_R_untrained"]
    if drum == "SN":
        return cost["SN_R"] if hand == "R" else cost["SN_L_trained" if state.has("LEFT_SNARE") else "SN_L_untrained"]
    if drum in ("T1", "T2", "FT", "CRASH"):
        return cost["TOM_L"] if hand == "L" else cost["TOM_R_trained" if state.has("RIGHT_FILL_LEAD") else "TOM_R_untrained"]
    return 0.0


def hand_stroke_cost(kit: Kit, drum: str, hand: str, idx: int,
                     motor: Motor, params: dict) -> float:
    x, y = kit.position(drum)
    last_drum = motor.right_drum if hand == "R" else motor.left_drum
    last_tick = motor.right_tick if hand == "R" else motor.left_tick
    x0, y0 = kit.position(last_drum)
    gap = max(1, idx - last_tick)
    dist = math.hypot(x-x0, y-y0)
    # Last strike far away + little time between hits is expensive.
    movement = params["movement_weight"] * dist / math.sqrt(gap)
    # Reaching a far side with the opposite arm is biomechanically demanding.
    side = 1 if hand == "R" else -1
    across = max(0.0, -(side*x) - 0.18)
    reach = params["wrong_side_weight"] * across**1.3
    return movement + reach


def advance(motor: Motor, step: Step, hands: tuple[str, ...],
            idx: int, state: State, params: dict,
            forced_cymbal_hand: Optional[str] = None,
            forced_first_fill_hand: Optional[str] = None) -> Optional[tuple[Motor, float, str]]:
    kit = state.kit
    mp = dict(zip(step.hits, hands))
    if forced_cymbal_hand and any(d in mp and mp[d] != forced_cymbal_hand for d in ("HH", "RIDE")):
        return None
    if forced_first_fill_hand and idx == 0 and step.hits and step.hits[0] not in ("HH", "RIDE"):
        if hands[0] != forced_first_fill_hand:
            return None

    cost = 0.0
    new = motor
    posture = "-"
    for drum, hand in mp.items():
        cost += neural_penalty(drum, hand, state, params)
        cost += hand_stroke_cost(kit, drum, hand, idx, motor, params)
        if hand == motor.last_hand and len(step.hits) == 1:
            cost += params["quick_repeat_penalty"]
            if motor.run_length >= 2:
                cost += params["three_strokes_penalty"]
        if drum in ("HH", "RIDE"):
            if motor.cymbal_hand and motor.cymbal_hand != hand:
                cost += params["cymbal_hand_switch_penalty"]
            new = replace(new, cymbal_hand=hand)
        if hand == "R": new = replace(new, right_drum=drum, right_tick=idx)
        else: new = replace(new, left_drum=drum, left_tick=idx)

    if len(step.hits) == 2:
        right_drum = step.hits[hands.index("R")]
        left_drum = step.hits[hands.index("L")]
        rx = kit.position(right_drum)[0]
        lx = kit.position(left_drum)[0]
        crossed = rx < lx
        posture = "crossed" if crossed else "open"
        if crossed: cost += params["crossed_arms_penalty"]
    elif len(step.hits) == 1:
        posture = "single"

    if step.group and len(step.hits) == 1:
        first = hands[0]
        if motor.prev_group:
            prev_x = kit.position(motor.prev_group)[0]
            new_x = kit.position(step.group)[0]
            travel_x = new_x - prev_x
            # The FIRST hand to arrive on a new drum carries an empirically
            # configurable directional challenge. Leftward -> L favored;
            # rightward -> R favored. This is a hypothesis, not a physical law.
            if (travel_x > 0 and first == "L") or (travel_x < 0 and first == "R"):
                cost += params["direction_lead_weight"] * abs(travel_x)
        if motor.group_lead and motor.group_lead != first:
            cost += (params["lead_switch_penalty_trained"] if state.has("LEAD_SWITCH")
                     else params["lead_switch_penalty_untrained"])
        new = replace(new, prev_group=step.group, group_lead=first)
        # Treat the ability to initiate with R, not every R stroke, as a
        # learnable fill capability. This is independent of HH/RIDE hand.
        if first == "R":
            cost += params["neural_stroke_penalties"]["RIGHT_FILL_INIT_trained" if state.has("RIGHT_FILL_LEAD") else "RIGHT_FILL_INIT_untrained"]

    # Hand–foot coordination costs: first-class, *measurable* input.
    if step.kick:
        if kit.kick_foot == -1:
            cost += params["neural_stroke_penalties"]["LEFT_KICK_trained" if state.has("LEFT_KICK") else "LEFT_KICK_untrained"]
        cym = next((d for d in ("HH", "RIDE") if d in mp), "")
        if cym and "SN" in mp:
            # Explicit kick-foot identifier; legacy _K defaults to right kick.
            key = f"{mp[cym]}_{cym}_{mp['SN']}_SN_K{'R' if kit.kick_foot == 1 else 'L'}"
            fallback = f"{mp[cym]}_{cym}_{mp['SN']}_SN_K"
            cost += params["hand_foot_pairing"].get(key, params["hand_foot_pairing"].get(fallback, 0.0))
            if (mp[cym] == "R" and mp["SN"] == "L"
                    and state.has("CROSSED_HAND_FOOT")):
                cost -= params["pairing_gain_from_training"]

    if step.hat_pedal:
        if step.kick and kit.hat_foot == kit.kick_foot:
            return None  # same foot cannot strike two independent pedals
        if kit.hat_foot == 1:
            cost += params["neural_stroke_penalties"]["RIGHT_HAT_FOOT_trained" if state.has("RIGHT_HAT_FOOT") else "RIGHT_HAT_FOOT_untrained"]

    if len(step.hits) == 1:
        h = hands[0]
        new = replace(new, last_hand=h,
                      run_length=(motor.run_length+1 if h == motor.last_hand else 1))
    elif len(step.hits) == 2:
        # Simultaneous strikes don't imply a sequential R/L sticking.
        new = replace(new, last_hand="", run_length=0)
    return new, cost, posture


def optimize_phrase(phrase: Phrase, state: State, params: dict,
                    forced_cymbal_hand: Optional[str] = None,
                    forced_first_fill_hand: Optional[str] = None) -> Candidate:
    width = int(params["beam_width"])
    beam = [Candidate(0.0, Motor(), (), ())]
    for idx, step in enumerate(phrase.steps):
        candidates = []
        for item in beam:
            for hands in options_for(step):
                next_step = advance(item.motor, step, hands, idx, state, params,
                                    forced_cymbal_hand, forced_first_fill_hand)
                if next_step is None: continue
                m, incremental, posture = next_step
                label = ("+".join(f"{d}:{h}" for d, h in zip(step.hits, hands))
                         or ("K" if step.kick else "rest"))
                candidates.append(Candidate(item.score+incremental, m,
                                            item.sticking+(label,), item.postures+(posture,)))
        if not candidates:
            raise ValueError(f"No feasible sticking at step {idx} for {phrase.name}")
        # Distinct motor states preserve multiple strategically different paths.
        best_for_motor = {}
        for c in candidates:
            old = best_for_motor.get(c.motor)
            if old is None or c.score < old.score:
                best_for_motor[c.motor] = c
        beam = sorted(best_for_motor.values(), key=lambda c: c.score)[:width]
    return min(beam, key=lambda c: c.score)


def evaluate_fixed_sticking(phrase: Phrase, state: State,
                            params: dict, hands: str) -> Candidate:
    """Score an exact single-stroke sticking like RRLLRRLLRRLLR."""
    if len(hands) != len(phrase.steps) or any(len(s.hits) != 1 for s in phrase.steps):
        raise ValueError("Fixed sticking requires exactly one hand stroke per event")
    m = Motor()
    score = 0.0
    played, poses = [], []
    for i, (step, h) in enumerate(zip(phrase.steps, hands)):
        if h not in ("R", "L"):
            raise ValueError("Sticking must contain only R and L")
        result = advance(m, step, (h,), i, state, params)
        if result is None: raise ValueError("Impossible fixed sticking")
        m, points, pose = result
        score += points
        played.append(f"{step.hits[0]}:{h}")
        poses.append(pose)
    return Candidate(score, m, tuple(played), tuple(poses))


def score_state(state: State, params: dict,
                repertoire: tuple[Phrase, ...]) -> tuple[float, dict[str, float]]:
    if state.kit.hat_foot == state.kit.kick_foot:
        # Infeasible temporary configuration: useful as a graph bridge for
        # switching BOTH pedals, but never a recommended finished layout.
        return 1000.0, {p.name: 1000.0 for p in repertoire}
    cost = {}
    for phrase in repertoire:
        cost[phrase.name] = optimize_phrase(phrase, state, params).score
    # Mean effort per representative phrase, weighted by musical importance.
    wt = sum(p.weight for p in repertoire)
    mean = sum(p.weight * cost[p.name] for p in repertoire) / wt
    return mean, cost


# ---------------------------------------------------------------------------
# Outer investment graph: a locally unprofitable action can be a prerequisite
# for access to a globally valuable future state (non-greedy traversal).
# ---------------------------------------------------------------------------
def starting_state(params: dict) -> State:
    kit = replace(STANDARD_KIT, **params.get("initial_kit", {}))
    learned = 0
    for skill in params.get("initial_skills", []):
        learned |= SKILL_BITS[skill]
    return State(kit, learned)


def layout_difference(kit: Kit, origin: Kit) -> int:
    return sum(getattr(kit, f) != getattr(origin, f)
               for f in ("hh", "ride", "crash", "floor", "toms", "snare", "kick_foot", "hat_foot"))


def state_neighbors(state: State, params: dict):
    for field in ("hh", "ride", "crash", "floor", "toms", "snare", "kick_foot", "hat_foot"):
        choices = (-1, 0, 1) if field == "snare" else ((0, 1) if field == "toms" else (-1, 1))
        for choice in choices:
            if choice == getattr(state.kit, field): continue
            dest = State(replace(state.kit, **{field:choice}), state.learned)
            yield dest, float(params["layout_hours"][field]), f"move {field} to {choice:+d}"
    for skill in SKILLS:
        if state.has(skill): continue
        if not all(state.has(s) for s in SKILL_REQUIRES.get(skill, ())): continue
        dest = State(state.kit, state.learned | SKILL_BITS[skill])
        yield dest, float(params["skill_hours"][skill]), f"learn {skill}"


def traverse_investments(params: dict):
    start = starting_state(params)
    best = {start: 0.0}
    previous = {}
    counter = itertools.count()
    heap = [(0.0, next(counter), start)]
    visited = []
    budget = float(params["budget_hours"])
    max_skills = int(params["max_new_skills"])
    max_layout = int(params["max_layout_edits"])
    existing_skills = start.learned
    while heap:
        spent, _, node = heapq.heappop(heap)
        if spent > best[node] + 1e-9: continue
        visited.append(node)
        for next_state, cost, action in state_neighbors(node, params):
            if layout_difference(next_state.kit, start.kit) > max_layout: continue
            if (next_state.learned & ~existing_skills).bit_count() > max_skills: continue
            total = spent + cost
            if total > budget + 1e-9: continue
            if total < best.get(next_state, float("inf")) - 1e-9:
                best[next_state] = total
                previous[next_state] = (node, action, cost)
                heapq.heappush(heap, (total, next(counter), next_state))
    return visited, best, previous


def actions_to(state: State, previous: dict) -> list[dict]:
    steps = []
    while state in previous:
        prev, label, hours = previous[state]
        steps.append({"action": label, "hours": hours})
        state = prev
    return steps[::-1]


def predict(params: dict):
    repertoire = make_repertoire(params["repertoire_weights"], params.get("custom_patterns"))
    nodes, hours, previous = traverse_investments(params)
    scores = {}
    for node in nodes:
        scores[node], _ = score_state(node, params, repertoire)
    start = starting_state(params)
    baseline = scores[start]
    horizons = [float(h) for h in params["horizons_hours"]]
    leaders = {}
    for horizon in horizons:
        # Profit is the projected value of improved execution, minus cost to
        # acquire skills and change layout. Crucially, non-greedy: costs of
        # intermediate states do not decide whether a path is explored.
        ranking = sorted(nodes, key=lambda node:
                         horizon * params["benefit_factor"] * (baseline-scores[node]) - hours[node],
                         reverse=True)
        rows = []
        for node in ranking[:8]:
            rows.append({
                "state": node.label(),
                "kit": asdict(node.kit),
                "learned": [s for s in SKILLS if node.has(s)],
                "investment_hours": round(hours[node], 2),
                "effort": round(scores[node], 4),
                "effort_reduction": round(baseline-scores[node], 4),
                "net_practice_hour_equivalents": round(
                    horizon*params["benefit_factor"]*(baseline-scores[node])-hours[node], 2),
                "route": actions_to(node, previous),
                "_state": node,
            })
        leaders[horizon] = rows
    return {
        "params": params, "repertoire": repertoire,
        "nodes_explored": len(nodes), "baseline": baseline,
        "scores": scores, "hours": hours, "previous": previous,
        "ranking": leaders, "start": start,
    }


def comparison_table(state: State, params: dict, repertoire: tuple[Phrase, ...]) -> list[dict]:
    summaries = []
    for name in ("straight_hh", "right_ride", "rightward_fill", "leftward_fill"):
        phrase = next(p for p in repertoire if p.name == name)
        direction = "cymbal" if name in ("straight_hh", "right_ride") else "fill"
        for hand in ("L", "R"):
            result = optimize_phrase(phrase, state, params,
                       forced_cymbal_hand=hand if direction == "cymbal" else None,
                       forced_first_fill_hand=hand if direction == "fill" else None)
            postures = {p for p in result.postures if p in ("crossed", "open")}
            summaries.append({"phrase": name, "hand": hand,
                              "effort": round(result.score, 3),
                              "posture": "/".join(sorted(postures)) or "-",
                              "sticking": " | ".join(result.sticking)})
    return summaries


def write_dot(path: Path, result: dict, horizon: float):
    # A readable subgraph of paths leading to the top 5 states at this horizon.
    rankings = result["ranking"][horizon][:5]
    predecessors = result["previous"]
    edges = {}
    selected = {result["start"]}
    for entry in rankings:
        node = entry["_state"]
        selected.add(node)
        while node in predecessors:
            old, action, h = predecessors[node]
            edges[(old, node)] = (action, h)
            selected.add(old)
            node = old
    idx = {node: f"n{i}" for i, node in enumerate(selected)}
    top = rankings[0]["_state"]
    esc = lambda s: str(s).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    lines = ["digraph MotorLearning {", "  rankdir=TB;", "  graph [fontname=Helvetica, bgcolor=white];",
             "  node [shape=box, style=rounded, fontname=Helvetica, fontsize=9];",
             "  edge [fontname=Helvetica, fontsize=9];"]
    for node in selected:
        short_skills = ",".join(s.replace("RIGHT_", "R_").replace("LEFT_", "L_")
                                for s in SKILLS if node.has(s)) or "baseline"
        cost = result["hours"][node]
        benefit = horizon * result["params"]["benefit_factor"] * (result["baseline"]-result["scores"][node]) - cost
        layout = f"HH:{'L' if node.kit.hh == -1 else 'R'}  Ride:{'L' if node.kit.ride == -1 else 'R'}  Crash:{'L' if node.kit.crash == -1 else 'R'}  Floor:{'L' if node.kit.floor == -1 else 'R'}"
        if node.kit.toms or node.kit.snare or node.kit.kick_foot != 1 or node.kit.hat_foot != -1:
            layout += f"\nToms:{node.kit.toms} Snare:{node.kit.snare} Kick:{node.kit.kick_foot} HatPedal:{node.kit.hat_foot}"
        label = f"{layout}\n{short_skills}\nInvest {cost:g}h | Net {benefit:+.1f}h"
        color = 'fillcolor="#D2EEE1", style="rounded,filled"' if node == top else \
                ('fillcolor="#E7F1F9", style="rounded,filled"' if node == result["start"] else '')
        lines.append(f'  {idx[node]} [label="{esc(label)}"{", " + color if color else ""}];')
    for (old, new), (action, c) in edges.items():
        lines.append(f'  {idx[old]} -> {idx[new]} [label="{esc(action)} ({c:g}h)"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deeply_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deeply_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON overrides of neurological, equipment and learning costs")
    p.add_argument("--patterns", type=Path, help="Extra custom phrases: JSON {\"patterns\": [{name, weight, events:[...]}]}")
    p.add_argument("--budget", type=float, help="Available practice-equivalent investment hours")
    p.add_argument("--beam", type=int, help="Sticking search width, larger improves quality but is slower")
    p.add_argument("--out", type=Path, default=Path("drum_optimization.json"))
    p.add_argument("--dot", type=Path, default=Path("drum_options.dot"))
    args = p.parse_args()
    params = json.loads(json.dumps(DEFAULT_PARAMETERS))
    if args.config: params = deeply_merge(params, json.loads(args.config.read_text(encoding="utf-8")))
    if args.patterns:
        data = json.loads(args.patterns.read_text(encoding="utf-8"))
        params["custom_patterns"] = list(params.get("custom_patterns", [])) + data["patterns"]
    if args.budget is not None: params["budget_hours"] = args.budget
    if args.beam is not None: params["beam_width"] = args.beam
    print("Searching kit/skill graph (this is an individual model, not a neurological truth)...")
    result = predict(params)
    repertoire = result["repertoire"]
    print(f"\nExplored {result['nodes_explored']} reachable configurations, baseline effort {result['baseline']:.3f}.")
    for horizon, rows in result["ranking"].items():
        print(f"\nHorizon: {horizon:g} playing hours, top 3 (relative practice-hour values):")
        for i, row in enumerate(rows[:3], 1):
            print(f"  {i}. net {row['net_practice_hour_equivalents']:+.1f}h; "
                  f"investment {row['investment_hours']:g}h; "
                  f"effort {row['effort']:.2f}")
            print(f"     {row['state'].replace(chr(10), ' | ')}")
            print("     Route:", " -> ".join(x['action'] for x in row['route']) or "do nothing")

    selected = result["ranking"][max(result["ranking"])][0]["_state"]
    print("\nBaseline A/B comparisons (L vs R cymbal hand or fill starter):")
    for row in comparison_table(result["start"], params, repertoire):
        print(f"  {row['phrase']:<17} {row['hand']}  effort={row['effort']:>6.2f}  {row['posture']}")
    print("\nLong-horizon leader A/B comparisons:")
    for row in comparison_table(selected, params, repertoire):
        print(f"  {row['phrase']:<17} {row['hand']}  effort={row['effort']:>6.2f}  {row['posture']}")

    # Compare the user's actual RRLL vs LLRR orchestration, not merely the
    # beam search's unconstrained optimum.
    fill_phrase = next(p for p in repertoire if p.name == "rightward_fill")
    fixed_comparison = {}
    for name, node in (("baseline", result["start"]), ("long_horizon_winner", selected)):
        fixed_comparison[name] = {}
        print(f"\nExact rightward tom fill on {name}:")
        for sticking in ("RRLL"*3+"R", "LLRR"*3+"L", "RLRL"*3+"R", "LRLR"*3+"L"):
            c = evaluate_fixed_sticking(fill_phrase, node, params, sticking)
            fixed_comparison[name][sticking] = round(c.score, 3)
            print(f"  {sticking} -> {c.score:.3f} effort points")

    serializable = {
        "disclaimer": "Effort and neurological penalties are personalized hypotheses, not measured facts.",
        "parameters": params,
        "explored_states": result["nodes_explored"],
        "baseline_effort": result["baseline"],
        "rankings_by_playing_hours": {
            str(h): [{k: v for k, v in row.items() if k != "_state"} for row in rows]
            for h, rows in result["ranking"].items()
        },
        "baseline_AB": comparison_table(result["start"], params, repertoire),
        "long_horizon_winner_AB": comparison_table(selected, params, repertoire),
        "exact_rightward_fill_stickings": fixed_comparison,
    }
    args.out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    write_dot(args.dot, result, max(result["ranking"]))
    print(f"\nWrote {args.out}, {args.dot}")


if __name__ == "__main__":
    main()
