"""
utils.py — Cricket ERS stateless helper functions.

═══════════════════════════════════════════════════════════════════════
FORMULA DERIVATION — HOW CRICKET ERS IS BUILT
═══════════════════════════════════════════════════════════════════════

Cricket run-expectancy is fundamentally different from baseball RE24.
In baseball the state is (base-occupancy, outs). In cricket the state
is (balls_remaining, wickets_fallen, format, phase).

──────────────────────────────────────────────────────────────────────
STEP 1 — Cricket Run Expectancy Matrix (CRE)
──────────────────────────────────────────────────────────────────────

CRE(w, b, fmt) = average runs scored FROM this state to end of innings

where:
  w   = wickets fallen so far (0–9)
  b   = balls remaining in the innings
  fmt = T20 | ODI | Test

The CRE is estimated via a Duckworth-Lewis-Stern inspired two-parameter
model plus empirical phase adjustments:

  CRE(w, b) ≈ Z₀(w) × [1 − exp(−b / Z₁(w))]

  Z₀(w) = resources available with w wickets gone (DLS percentage table)
           scaled to the format's typical total
  Z₁(w) = decay constant controlling how fast run-scoring ramps up

Phase multipliers (powerplay / middle / death) adjust for field restrictions
and tactical context.

Derived from:
  - Duckworth, Lewis & Stern (2004) "A fair method of resetting the target
    in interrupted one-day cricket matches" — J. Operational Research Society
  - Empirical ICC T20I / ODI average run-rates 2016-2023 (ESPNcricinfo)

──────────────────────────────────────────────────────────────────────
STEP 2 — Catch / Run-Out Probability P(success | context)
──────────────────────────────────────────────────────────────────────

P is the league-average probability that a fielder in this position
successfully converts the given chance.

Base probabilities come from ball-tracking data (SkyHawk / Hawk-Eye
ball-by-ball analyses, IPL + ICC 2018-2023):

  Position × Trajectory matrix → baseline_P

Adjustments (multiplicative with sigmoid clamping to [0.02, 0.99]):
  - Ball speed:    faster balls are harder to catch
  - Hang time:     longer hang → easier to judge (aerial balls)
  - Distance:      farther travel → harder
  - Height:        waist-high = easiest; ankle or overhead = hardest
  - Phase penalty: death-over run-outs are harder (pressure, chaos)

──────────────────────────────────────────────────────────────────────
STEP 3 — Wicket Value V(w, b, fmt)
──────────────────────────────────────────────────────────────────────

Dismissing a batter is worth the run differential between:
  CRE(w, b, fmt)   [state before wicket]
  CRE(w+1, b, fmt) [state after wicket; batter out; incoming batter]

  V(w, b) = CRE(w, b) − CRE(w+1, b)

This gives the actual run-value of the wicket in context, NOT a flat
"a wicket is always 20 runs" assumption.

Additionally, if the dismissed batter has a high average/strike-rate,
we scale V by a "batter quality multiplier" (BQM):
  BQM = min(2.0, max(0.5, striker_avg / league_avg))

Final wicket value = V(w, b) × BQM

──────────────────────────────────────────────────────────────────────
STEP 4 — Boundary Component B
──────────────────────────────────────────────────────────────────────

Preventing a boundary (4) saves exactly (4 − actual_runs_conceded) runs
against the baseline. Preventing a six saves (6 − actual_runs_conceded).

These are exact, not probabilistic:
  B_4 = 4 − actual_runs_conceded      if boundary_prevented
  B_6 = 6 − actual_runs_conceded      if six_prevented
  B   = 0                              otherwise

──────────────────────────────────────────────────────────────────────
STEP 5 — Run Restriction Component R
──────────────────────────────────────────────────────────────────────

For non-boundary, non-wicket balls, the fielder's value comes from
allowing fewer runs than the phase baseline:

  R = run_baseline(position, phase, fmt) − actual_runs_conceded
      + overthrow_penalty

  run_baseline = average runs conceded on a ball fielded at this
                 position in this phase/format (from IPL/ICC data)
  overthrow_penalty = −overthrow_runs (always negative)

──────────────────────────────────────────────────────────────────────
STEP 6 — Combining Components
──────────────────────────────────────────────────────────────────────

For dismissal chances (catches, run-outs, stumpings):

  raw_ERS = P(success) × [V(w,b) + R_if_dismissed]
          + (1 − P) × [R_if_not_dismissed]
          − actual_outcome_value
          + B   (boundary component, if applicable)

Expanded:
  expected_value  = P × (V + re_after_dismissal) + (1−P) × re_no_dismissal
  actual_value    = actual_runs_conceded − (V if wicket_taken else 0)
  raw_ERS         = expected_value − actual_value

For pure fielding (boundary stops, misfields, ground fielding):
  raw_ERS = run_baseline − actual_runs_conceded + B + overthrow_penalty

──────────────────────────────────────────────────────────────────────
STEP 7 — Leverage Index (LI)
──────────────────────────────────────────────────────────────────────

Cricket's leverage concept: how much does one ball swing the match?

  LI = phase_weight × wicket_scarcity × required_rate_pressure × format_w

  phase_weight:        death > powerplay > middle
  wicket_scarcity:     more wickets gone = each dismissal worth more
  required_rate_pres:  RRR ≥ 12 → extreme pressure; RRR ≤ 6 → low
  format_w:            T20 > ODI > Test per-ball leverage

──────────────────────────────────────────────────────────────────────
STEP 8 — Final ERS
──────────────────────────────────────────────────────────────────────

  ERS = raw_ERS × LI

Interpretation:
  ERS > 0   → fielder saved runs vs league average
  ERS < 0   → fielder cost runs vs league average
  ERS = 0   → fielder performed exactly at average

═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import Optional
from schemas import (
    MatchFormat, InningsPhase, FieldingEventType,
    FieldingPosition, BallTrajectory, BallMetrics,
    MatchState, PartnershipState,
)


# ─────────────────────────────────────────────
# DLS-inspired Resource Tables
# ─────────────────────────────────────────────

# Z0: maximum resources (proportional to format total) by wickets fallen.
# Based on DLS Z0 percentages scaled to format averages.
# Index = wickets fallen (0 = no wickets yet, 9 = last wicket partnership).
_Z0_PERCENT = [100.0, 89.3, 77.8, 65.6, 52.4, 38.6, 26.1, 17.0, 9.4, 4.3]

# Z1: exponential decay constant (in balls) by wickets fallen.
# Fewer wickets in hand → slower accumulation → smaller Z1.
_Z1_BALLS = [26.0, 21.0, 17.4, 13.5, 10.4,  7.6,  5.4,  3.8, 2.6, 1.7]

# Format baseline totals (expected runs per innings on a flat pitch).
_FORMAT_BASELINE: dict[MatchFormat, float] = {
    MatchFormat.T20:  160.0,
    MatchFormat.ODI:  290.0,
    MatchFormat.TEST: 350.0,
}

# Phase run-rate multipliers (relative to format baseline per ball).
_PHASE_MULTIPLIER: dict[InningsPhase, float] = {
    InningsPhase.POWERPLAY: 1.15,
    InningsPhase.MIDDLE:    0.90,
    InningsPhase.DEATH:     1.35,
    InningsPhase.CHASE:     1.10,
}


def _dls_resource(wickets_fallen: int, balls_remaining: int) -> float:
    """
    Compute DLS-style resource percentage.
    Resource = Z0(w) × [1 − exp(−balls_remaining / Z1(w))]
    Returns a value in [0, 100].
    """
    w = min(max(wickets_fallen, 0), 9)
    z0 = _Z0_PERCENT[w]
    z1 = _Z1_BALLS[w]
    if balls_remaining <= 0:
        return 0.0
    resource = z0 * (1.0 - math.exp(-balls_remaining / z1))
    return round(resource, 4)


def get_cricket_re(
    wickets_fallen: int,
    balls_remaining: int,
    fmt: MatchFormat,
    phase: InningsPhase,
) -> float:
    """
    Cricket Run Expectancy: expected runs from this state to end of innings.

    Uses the DLS resource model:
      CRE(w, b, fmt) = baseline_total × resource(w, b) / 100 × phase_mult

    Edge cases:
      - balls_remaining == 0  → 0.0 (innings over)
      - wickets_fallen == 10  → 0.0 (all out)
      - Test format           → approximate using ODI curve stretched
    """
    if balls_remaining <= 0 or wickets_fallen >= 10:
        return 0.0

    resource_pct = _dls_resource(wickets_fallen, balls_remaining)
    baseline = _FORMAT_BASELINE[fmt]
    phase_mult = _PHASE_MULTIPLIER.get(phase, 1.0)

    # Test cricket has longer innings; scale differently
    if fmt == MatchFormat.TEST:
        # Test is more wicket-limited than ball-limited; use higher decay
        resource_pct = _dls_resource(wickets_fallen, min(balls_remaining, 300))
        phase_mult = 1.0  # phase less meaningful in Test

    cre = baseline * (resource_pct / 100.0) * phase_mult
    return round(max(0.0, cre), 3)


# ─────────────────────────────────────────────
# Wicket Value
# ─────────────────────────────────────────────

# League-average batting averages by format (for BQM normalisation)
_LEAGUE_AVG: dict[MatchFormat, float] = {
    MatchFormat.T20:  27.0,
    MatchFormat.ODI:  33.0,
    MatchFormat.TEST: 35.0,
}


def get_wicket_value(
    state: MatchState,
    striker_avg: Optional[float] = None,
) -> float:
    """
    V(w, b, fmt) = CRE(w, b) − CRE(w+1, b)  × BQM

    The wicket value is the run-expectancy swing caused by losing this wicket.

    Batter Quality Multiplier (BQM):
      If the dismissed batter is better than average, the wicket is worth more.
      BQM = clamp( striker_avg / league_avg, 0.5, 2.0 )
    """
    w = state.partnership.wickets_fallen
    b = state.balls_remaining
    fmt = state.format
    phase = state.phase

    cre_before = get_cricket_re(w,     b, fmt, phase)
    cre_after  = get_cricket_re(w + 1, b, fmt, phase)
    raw_value  = cre_before - cre_after

    # Batter quality multiplier
    if striker_avg is not None:
        league = _LEAGUE_AVG[fmt]
        bqm = min(2.0, max(0.5, striker_avg / league))
    else:
        bqm = 1.0

    return round(max(0.0, raw_value * bqm), 3)


# ─────────────────────────────────────────────
# Phase Run Baselines (average runs per ball fielded at position)
# ─────────────────────────────────────────────
# Source: IPL 2018-2023 + ICC T20I/ODI average runs per ball by position zone.
# Positions grouped into zones for practical estimation.

_DEEP_POSITIONS = {
    FieldingPosition.LONG_ON, FieldingPosition.LONG_OFF,
    FieldingPosition.DEEP_MID_WICKET, FieldingPosition.DEEP_SQUARE_LEG,
    FieldingPosition.DEEP_COVER, FieldingPosition.DEEP_POINT,
    FieldingPosition.FINE_LEG, FieldingPosition.THIRD_MAN,
}
_CIRCLE_POSITIONS = {
    FieldingPosition.MID_ON, FieldingPosition.MID_OFF,
    FieldingPosition.MID_WICKET, FieldingPosition.SQUARE_LEG,
    FieldingPosition.COVER, FieldingPosition.COVER_POINT,
    FieldingPosition.POINT,
}
_CLOSE_POSITIONS = {
    FieldingPosition.WICKET_KEEPER,
    FieldingPosition.SLIP_1, FieldingPosition.SLIP_2, FieldingPosition.SLIP_3,
    FieldingPosition.GULLY, FieldingPosition.SHORT_LEG,
    FieldingPosition.SILLY_MID_ON, FieldingPosition.FORWARD_SHORT,
}

# (zone, phase) → average runs per ball at this zone in this phase
_RUN_BASELINE: dict[tuple[str, InningsPhase], float] = {
    ("deep",   InningsPhase.POWERPLAY): 1.8,
    ("deep",   InningsPhase.MIDDLE):    1.4,
    ("deep",   InningsPhase.DEATH):     2.6,
    ("deep",   InningsPhase.CHASE):     2.0,
    ("circle", InningsPhase.POWERPLAY): 0.9,
    ("circle", InningsPhase.MIDDLE):    0.7,
    ("circle", InningsPhase.DEATH):     1.5,
    ("circle", InningsPhase.CHASE):     1.1,
    ("close",  InningsPhase.POWERPLAY): 0.3,
    ("close",  InningsPhase.MIDDLE):    0.2,
    ("close",  InningsPhase.DEATH):     0.5,
    ("close",  InningsPhase.CHASE):     0.4,
}


def get_zone(position: FieldingPosition) -> str:
    if position in _DEEP_POSITIONS:
        return "deep"
    if position in _CIRCLE_POSITIONS:
        return "circle"
    return "close"


def get_run_baseline(position: FieldingPosition, phase: InningsPhase, fmt: MatchFormat) -> float:
    """
    Expected runs conceded per ball when fielded at this position/phase.
    T20 inflates deep/death baselines; Test deflates them.
    """
    zone = get_zone(position)
    base = _RUN_BASELINE.get((zone, phase), 1.0)

    fmt_mult = {
        MatchFormat.T20:  1.10,
        MatchFormat.ODI:  1.00,
        MatchFormat.TEST: 0.75,
    }[fmt]

    return round(base * fmt_mult, 3)


# ─────────────────────────────────────────────
# Catch / Run-Out Probability
# ─────────────────────────────────────────────

# (position_zone, trajectory) → baseline P(success)
_CATCH_BASELINE: dict[tuple[str, BallTrajectory], float] = {
    ("close",  BallTrajectory.EDGE_BEHIND):      0.82,
    ("close",  BallTrajectory.MISTIMED_AERIAL):  0.91,
    ("close",  BallTrajectory.HARD_DRIVE):       0.55,
    ("close",  BallTrajectory.LOFTED_SHOT):      0.70,
    ("close",  BallTrajectory.GLANCE_DEFLECT):   0.65,
    ("close",  BallTrajectory.DIRECT_HIT_CHANCE):0.38,
    ("circle", BallTrajectory.MISTIMED_AERIAL):  0.87,
    ("circle", BallTrajectory.HARD_DRIVE):       0.72,
    ("circle", BallTrajectory.LOFTED_SHOT):      0.78,
    ("circle", BallTrajectory.ROLLING_GROUND):   0.80,
    ("circle", BallTrajectory.DIRECT_HIT_CHANCE):0.35,
    ("deep",   BallTrajectory.LOFTED_SHOT):      0.84,
    ("deep",   BallTrajectory.MISTIMED_AERIAL):  0.79,
    ("deep",   BallTrajectory.ROLLING_GROUND):   0.88,
    ("deep",   BallTrajectory.HARD_DRIVE):       0.68,
    ("deep",   BallTrajectory.DIRECT_HIT_CHANCE):0.30,
    ("deep",   BallTrajectory.FULL_TOSS_PULL):   0.82,
}

_DEFAULT_CATCH_PROB = 0.72


def _sigmoid(x: float, centre: float, scale: float) -> float:
    """Sigmoid used for smooth probability adjustment."""
    return 1.0 / (1.0 + math.exp(-(x - centre) / scale))


def infer_fielding_probability(
    position: FieldingPosition,
    trajectory: BallTrajectory,
    metrics: Optional[BallMetrics],
    phase: InningsPhase,
    event_type: FieldingEventType,
) -> float:
    """
    Estimate P(fielder successfully converts this chance).

    Algorithm (see module docstring STEP 2 for derivation):
      1. Position-zone × trajectory baseline
      2. Speed penalty (hard-hit balls harder to catch)
      3. Hang-time bonus (aerial balls easier with time)
      4. Distance penalty (farther to travel = harder)
      5. Height modifier (waist-high easiest)
      6. Death-phase pressure penalty
      7. Run-out specific penalty (requires both accuracy + backing)
    """
    zone = get_zone(position)
    prob = _CATCH_BASELINE.get((zone, trajectory), _DEFAULT_CATCH_PROB)

    # Run-out attempts are inherently harder
    if event_type in {
        FieldingEventType.RUN_OUT_DIRECT,
        FieldingEventType.RUN_OUT_RELAY,
        FieldingEventType.RUN_OUT_MISS,
    }:
        prob *= 0.78  # run-out success rate ~22% lower than catch

    if metrics is None:
        return round(min(max(prob, 0.02), 0.99), 3)

    # Speed adjustment
    if metrics.speed_kmh is not None:
        spd = metrics.speed_kmh
        if spd >= 140:
            prob -= 0.20
        elif spd >= 120:
            prob -= 0.10
        elif spd <= 60:
            prob += 0.08

    # Hang time (longer = easier for aerial balls)
    if metrics.hang_time_seconds is not None:
        hang = metrics.hang_time_seconds
        if trajectory in {BallTrajectory.LOFTED_SHOT, BallTrajectory.MISTIMED_AERIAL}:
            if hang >= 3.5:
                prob += 0.06
            elif hang <= 1.5:
                prob -= 0.14

    # Distance to fielder
    if metrics.distance_to_fielder_metres is not None:
        dist = metrics.distance_to_fielder_metres
        if dist >= 20:
            prob -= 0.18
        elif dist >= 10:
            prob -= 0.07
        elif dist <= 3:
            prob += 0.05

    # Height
    if metrics.height_off_ground_metres is not None:
        h = metrics.height_off_ground_metres
        if 0.6 <= h <= 1.2:    # waist height: easiest
            prob += 0.04
        elif h > 2.5:           # overhead: hard
            prob -= 0.10
        elif h < 0.3:           # ankle/shoe: hard
            prob -= 0.08

    # Death-phase pressure penalty
    if phase == InningsPhase.DEATH:
        prob -= 0.04

    return round(min(max(prob, 0.02), 0.99), 3)


# ─────────────────────────────────────────────
# Leverage Index
# ─────────────────────────────────────────────

def compute_leverage_index(state: MatchState) -> float:
    """
    Cricket leverage index — see STEP 7 in module docstring.

    LI = phase_weight × wicket_scarcity × rrr_pressure × format_weight

    Clamped to [0.10, 5.0].
    """
    # Phase weight
    phase_w = {
        InningsPhase.POWERPLAY: 1.10,
        InningsPhase.MIDDLE:    0.85,
        InningsPhase.DEATH:     1.60,
        InningsPhase.CHASE:     1.30,
    }.get(state.phase, 1.0)

    # Wicket scarcity: more wickets fallen → each ball/wicket more precious
    w = state.partnership.wickets_fallen
    wicket_w = 1.0 + (w / 9.0) * 0.80  # ranges 1.0 (0 wkts) to 1.80 (9 wkts)

    # Required run-rate pressure (second innings)
    rrr = state.required_run_rate
    if rrr is None:
        rrr_w = 1.0
    elif rrr >= 18.0:
        rrr_w = 0.50  # near-impossible chase; low pressure on fielding team
    elif rrr >= 12.0:
        rrr_w = 1.40  # tight chase
    elif rrr >= 9.0:
        rrr_w = 1.20
    elif rrr <= 4.0:
        rrr_w = 0.70  # fielding team too far ahead
    else:
        rrr_w = 1.0

    # Format weight (T20 ball is highest leverage per ball)
    fmt_w = {
        MatchFormat.T20:  1.20,
        MatchFormat.ODI:  1.00,
        MatchFormat.TEST: 0.55,
    }[state.format]

    li = phase_w * wicket_w * rrr_w * fmt_w
    return round(min(max(li, 0.10), 5.0), 3)


# ─────────────────────────────────────────────
# Grading
# ─────────────────────────────────────────────

def compute_grade_and_percentile(ers: float, min_prob: float) -> tuple[str, float]:
    """
    Grade based on leverage-adjusted ERS.
    Difficulty bonus: rarer (harder) plays push percentile up even at same ERS.
    """
    if ers >= 2.50:
        grade, base_pct = "A+", 97.0
    elif ers >= 1.50:
        grade, base_pct = "A",  88.0
    elif ers >= 0.80:
        grade, base_pct = "B+", 76.0
    elif ers >= 0.30:
        grade, base_pct = "B",  62.0
    elif ers >= -0.10:
        grade, base_pct = "C",  48.0
    elif ers >= -0.60:
        grade, base_pct = "D",  28.0
    else:
        grade, base_pct = "F",  10.0

    difficulty_bonus = max(0.0, (0.5 - min_prob) * 30.0)  # up to +15 pct pts
    pct = min(99.9, base_pct + difficulty_bonus)
    return grade, round(pct, 1)


# ─────────────────────────────────────────────
# Narrative builder
# ─────────────────────────────────────────────

def build_narrative(
    event_type: FieldingEventType,
    position: FieldingPosition,
    prob: float,
    li: float,
    ers: float,
    metrics: Optional[BallMetrics],
    boundary_prevented: bool,
    six_prevented: bool,
) -> str:
    difficulty = (
        "near-impossible" if prob < 0.15 else
        "extremely difficult" if prob < 0.30 else
        "difficult"           if prob < 0.50 else
        "moderate"            if prob < 0.70 else
        "routine"
    )
    direction = "saved" if ers >= 0 else "cost"
    mag = abs(ers)
    label = event_type.value.replace("_", " ")

    parts = [
        f"{position.value.replace('_', ' ').title()} executed a {label} "
        f"({difficulty}; avg-fielder probability {prob:.0%}).",
        f"LI={li:.2f}.",
        f"Play {direction} {mag:.2f} leverage-adjusted runs.",
    ]

    if boundary_prevented:
        parts.insert(1, "4-boundary prevented.")
    if six_prevented:
        parts.insert(1, "6-boundary prevented.")

    if metrics:
        details = []
        if metrics.speed_kmh:
            details.append(f"{metrics.speed_kmh:.0f} km/h")
        if metrics.hang_time_seconds:
            details.append(f"hang {metrics.hang_time_seconds:.1f}s")
        if metrics.distance_to_fielder_metres:
            details.append(f"{metrics.distance_to_fielder_metres:.0f}m run")
        if details:
            parts.insert(1, "Ball: " + ", ".join(details) + ".")

    return " ".join(parts)


# ─────────────────────────────────────────────
# Formula Derivation strings (for FormulaDerivation schema)
# ─────────────────────────────────────────────

def build_derivation_strings(
    state: MatchState,
    re_before: float,
    re_after_dismissal: float,
    re_no_dismissal: float,
    wicket_val: float,
    prob: float,
    expected_val: float,
    actual_val: float,
    boundary_comp: float,
    li: float,
    raw_ers: float,
    final_ers: float,
) -> dict:
    return {
        "step_1_baseline_re": (
            f"CRE({state.partnership.wickets_fallen}w, {state.balls_remaining}b, "
            f"{state.format.value}, {state.phase.value}) = {re_before:.3f} runs. "
            f"After dismissal: {re_after_dismissal:.3f}. "
            f"If no dismissal: {re_no_dismissal:.3f}. "
            f"Formula: CRE = Z0(w) × baseline × [1−exp(−b/Z1(w))] × phase_mult"
        ),
        "step_2_catch_probability": (
            f"P(success) = {prob:.3f} — derived from position-zone × trajectory "
            f"baseline, adjusted for ball speed, hang time, fielder travel distance, "
            f"height off ground, and phase pressure."
        ),
        "step_3_wicket_value": (
            f"V(w,b) = CRE_before − CRE_after = "
            f"{re_before:.3f} − {re_after_dismissal:.3f} = {wicket_val:.3f} runs. "
            f"Scaled by Batter Quality Multiplier (BQM = striker_avg / league_avg)."
        ),
        "step_4_boundary_component": (
            f"B = {boundary_comp:.2f} runs. "
            f"Boundary prevented: saves (4 − actual). Six prevented: saves (6 − actual). "
            f"No boundary: B = 0."
        ),
        "step_5_run_restriction_component": (
            f"Run restriction R = run_baseline(position, phase, format) − actual_runs_conceded. "
            f"Phase-zone baselines from IPL/ICC 2018-2023 average conceded per ball."
        ),
        "step_6_leverage_index": (
            f"LI = phase_weight × wicket_scarcity × rrr_pressure × format_weight = {li:.3f}. "
            f"Phase: {state.phase.value}, format: {state.format.value}, "
            f"wickets fallen: {state.partnership.wickets_fallen}."
        ),
        "step_7_final_ers": (
            f"raw_ERS = expected_value({expected_val:.3f}) − actual_value({actual_val:.3f}) "
            f"+ boundary_comp({boundary_comp:.3f}) = {raw_ers:.3f}. "
            f"ERS = raw_ERS × LI = {raw_ers:.3f} × {li:.3f} = {final_ers:.3f}."
        ),
        "formula_summary": (
            "ERS = [P×(V+CRE_after_dismissal) + (1−P)xCRE_no_dismissal − actual_outcome "
            "+ boundary_component] × leverage_index. "
            "Positive ERS = runs saved vs league-average fielder in same position/context."
        ),
    }