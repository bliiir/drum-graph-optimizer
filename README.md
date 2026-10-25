# Drumming as a graph: kit, technique, learning

This is a *decision-support experiment*, not a scientifically validated model of neurology or ergonomics. It is designed for a drummer who plays left-hand lead and established open-handed hi-hat, but is learning right-hand hi-hat/ride, left-hand snare, and right-hand-led fills on a standard kit.

## Run

Requires Python 3.10+, no external Python packages:

```bash
python drum_graph_optimizer.py \
    --config drum_personal_profile.json \
    --patterns drum_custom_patterns.json \
    --budget 34 \
    --beam 44 \
    --out drum_optimization.json \
    --dot drum_options.dot
```

Optionally render graph with Graphviz: `dot -Tpng drum_options.dot -o drum_options.png`.

Run `python drum_graph_optimizer.py --help` for CLI arguments. `--config` is optional: internal defaults are provided. Set `initial_kit` and `initial_skills` to describe what you already have and play fluently; only *additional* skills count against `max_new_skills`. For example, `"initial_skills": ["RIGHT_RIDE"]` if right-hand ride is already automatic.

## What is searched?

**Physical kit:** independent left/right positions for HH, ride, crash, floor tom; mirrored rack-tom order; snare left/center/right; kick-foot and HH-foot sides. A right-side HH while pedal remains on the left represents a remote HH. This model permits equipment changes but assigns them a one-time cost. A configuration that requires one foot to operate both kick and HH pedal independently is treated as unplayable, although graph search can pass through it to discover a fully swapped setup.

**Within each rhythm:** which hand hits which drum, the next drum position, how long each hand has had to relocate, hand position/reach, crossing, directional lead of an incoming fill group, sticking doubles/triples, hand/foot pairing, and the ability to switch fill lead. The motor planner keeps multiple candidate motor states (beam search). Increase `--beam` to reduce search approximation. Note: this is not exhaustive enumeration of every possible hit sequence.

**Across practice investments:** train RIGHT_HH, RIGHT_RIDE, LEFT_SNARE, RIGHT_FILL_LEAD, LEAD_SWITCH, CROSSED_HAND_FOOT, LEFT_KICK and RIGHT_HAT_FOOT. Skill-learning edges reduce *modeled* performance penalties. LEAD_SWITCH requires RIGHT_FILL_LEAD; CROSSED_HAND_FOOT requires RIGHT_HH + LEFT_SNARE. A locally unattractive acquisition can still be part of an optimal long-term route, because the outer Dijkstra traversal is not greedy. The graph explores every state within the stated budget and layout/skill change caps.

**Repertoire:** straight HH, syncopated HH with kick, right-side ride, rightward tom fill, leftward tom fill, round-trip fill, and groove-to-fill. Adjust `repertoire_weights` to your actual playing. Add your own phrases in `drum_custom_patterns.json` and supply `--patterns drum_custom_patterns.json`: each event is one 16th subdivision, with one or two simultaneous hand targets and optional `KICK` or `HAT_PEDAL` separated by `+`. An example fill is `"SN", "SN", "T1", "T1", "FT", "FT", "CRASH"`. The engine searches the possible stickings for each phrase. It cannot literally enumerate every imaginable rhythm; provide representative phrases and assign realistic weights.

## Interpretation of scores

The execution scores have arbitrary units and *must not* be interpreted as milliseconds, muscle effort in joules, neurological load, or proven natural handedness. The sample weights encode hypotheses and your reported playing history, not empirical test results.

The outer objective is:

`net_value = future_playing_hours * benefit_factor * (baseline_execution_cost - candidate_execution_cost) - one_time_investment_hours`

The `benefit_factor` is an entirely subjective conversion of cost points to hours-equivalent value. It, the future hours horizon, and your actual repertoire heavily determine which configuration wins.

Do not choose a real setup from the default ranking. Calibrate first and test sensitivity to uncertain weights.

## How to calibrate your neurological and mechanical costs

1. Fill the provided `drum_calibration_template.csv`. For each task, record 3–5 trials, use fixed tempi, and alternate condition order between days. If possible record multitrack audio or video so you can measure timing variance, missed hits, and unnatural movement.
2. Separately rate body effort and discomfort. Never optimize only for speed; a setup that feels fast but causes tension is not optimal.
3. Use the **differences** between conditions to adjust `neural_stroke_penalties`, `hand_foot_pairing`, `crossed_arms_penalty`, and `direction_lead_weight` in your JSON. You may need different coefficients for different tempi. Training costs should be estimated from how much practice you need to achieve stable proficiency, rather than borrowed from someone else.
4. Re-run across horizons: e.g. 30 hours (short term), 200 hours (medium), 800 hours (long). Inspect whether the top option changes when you vary uncertain weights by ±50%. If it does, you have a question to test instead of an answer to assume.
5. Update the repertoire, especially if you play improvisational jazz/fusion: add polyrhythms, orchestrated accents, hi-hat-foot independence, tom displacement, and free improvisation segments.

### Key limits

- The physical kit uses simplified 2D positions, not precise human reach, forearm crossing or detailed stick trajectory/collision detection.
- Learned skills are binary acquired/not acquired. Real learning curves, fatigue, interference, retention and transfer are richer.
- Motor execution uses beam search, which might prune the globally best route.
- The candidate set is bounded by `max_layout_edits`, `max_new_skills`, and `budget_hours`; the best state means *best among reachable modeled states*, not globally or universally best.
- The neurologically correct values cannot be established from arm-crossing geometry alone. In particular, comfortable crossed-hand hand/foot coupling is entirely plausible and should be measured, not penalized by assumption.

## Why the graph framing matters

One can find an endpoint where `LEAD_SWITCH` is valuable, even if training RIGHT_FILL_LEAD by itself looks inefficient. The first skill is a gateway to the second. This is the kind of nonlocal investment decision a simplistic per-fill or per-minute ranking misses.
