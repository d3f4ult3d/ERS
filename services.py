"""
services.py — Cricket Expected Runs Saved calculation engine.

The full ERS formula is derived in utils.py. This module implements
the calculation pipeline, handling all edge cases for every event type.

Event handling matrix:
  ┌──────────────────────────┬──────────────────────────────────────────┐
  │ Event Type               │ Calculation Path                         │
  ├──────────────────────────┼──────────────────────────────────────────┤
  │ CATCH                    │ Dismissal: P × wicket_value + run restr. │
  │ CATCH_DROPPED            │ Missed dismissal: negative ERS from      │
  │                          │ expected wicket not taken                │
  │ RUN_OUT_DIRECT           │ Dismissal: P × wicket_value (run-out)    │
  │ RUN_OUT_RELAY            │ Same; probability lower (relay harder)   │
  │ RUN_OUT_MISS             │ Negative: chance squandered + runs given │
  │ STUMPING                 │ Dismissal: P × wicket_value (keeper)     │
  │ BOUNDARY_STOP            │ B_4 component; run restriction only      │
  │ BOUNDARY_SAVED_6         │ B_6 component; run restriction only      │
  │ BOUNDARY_MISSED          │ Negative: boundary conceded vs baseline  │
  │ MISFIELD_RUNS            │ Negative: extra runs vs run baseline     │
  │ DIVING_STOP              │ Positive: run saved vs baseline          │
  │ RELAY_THROW_GOOD         │ Positive: restricted runs vs baseline    │
  │ RELAY_THROW_POOR         │ Negative: extra runs vs baseline         │
  │ GROUND_FIELDING          │ Neutral to slightly positive             │
  │ OVERTHROW                │ Negative: overthrow penalty              │
  │ OBSTRUCTING_FIELD        │ Special: penalty handled separately      │
  └──────────────────────────┴──────────────────────────────────────────┘
"""

from __future__ import annotations

from schemas import (
    ERSRequest, ERSResponse, ERSEventBreakdown,
    ExpectedVsActual, FormulaDerivation,
    FieldingEventType, FieldingEventDetail,
    MatchState,
)
from utils import (
    get_cricket_re,
    get_wicket_value,
    get_run_baseline,
    infer_fielding_probability,
    compute_leverage_index,
    compute_grade_and_percentile,
    build_narrative,
    build_derivation_strings,
)


# ─────────────────────────────────────────────
# Dismissal event types
# ─────────────────────────────────────────────
_DISMISSAL_EVENTS = {
    FieldingEventType.CATCH,
    FieldingEventType.RUN_OUT_DIRECT,
    FieldingEventType.RUN_OUT_RELAY,
    FieldingEventType.STUMPING,
}

_MISSED_DISMISSAL_EVENTS = {
    FieldingEventType.CATCH_DROPPED,
    FieldingEventType.RUN_OUT_MISS,
}

_BOUNDARY_EVENTS = {
    FieldingEventType.BOUNDARY_STOP,
    FieldingEventType.BOUNDARY_SAVED_6,
    FieldingEventType.BOUNDARY_MISSED,
}


# ─────────────────────────────────────────────
# Per-event calculation
# ─────────────────────────────────────────────

def _compute_boundary_component(event: FieldingEventDetail) -> float:
    """
    B = runs saved purely from boundary prevention.
    A prevented 4 that became 2 runs → B = 4 − 2 = +2.
    A boundary missed that cost 4 → B = 0 (no prevention, baseline already accounts for 4).
    An overthrow to boundary → B = −overthrow_runs (handled in overthrow_penalty).
    """
    if event.boundary_prevented:
        return max(0.0, 4.0 - event.actual_runs_conceded)
    if event.six_prevented:
        return max(0.0, 6.0 - event.actual_runs_conceded)
    if event.event_type == FieldingEventType.BOUNDARY_MISSED:
        # Missed boundary: extra cost vs field-zone baseline
        # baseline doesn't assume 4 on every ball; difference captured in run-restriction
        return 0.0
    return 0.0


def _compute_overthrow_penalty(event: FieldingEventDetail) -> float:
    """Overthrow runs are a direct penalty (always ≤ 0)."""
    return -float(event.overthrow_runs)


def _compute_event_ers(
    event: FieldingEventDetail,
    state: MatchState,
    li: float,
) -> ERSEventBreakdown:
    """
    Core per-event ERS calculation.

    Returns an ERSEventBreakdown with all components filled.
    """
    w      = state.partnership.wickets_fallen
    b      = state.balls_remaining
    fmt    = state.format
    phase  = state.phase

    # ── Probability ──────────────────────────────────────────────────
    if event.catch_or_run_out_probability is not None:
        prob = event.catch_or_run_out_probability
    else:
        prob = infer_fielding_probability(
            event.fielder_position,
            event.ball_trajectory,
            event.ball_metrics,
            phase,
            event.event_type,
        )

    # ── Run Expectancy states ─────────────────────────────────────────
    re_before           = get_cricket_re(w,     b, fmt, phase)
    re_after_dismissal  = get_cricket_re(w + 1, b, fmt, phase)
    re_no_dismissal     = get_cricket_re(w,     b, fmt, phase)  # state unchanged if not out

    # ── Wicket value ──────────────────────────────────────────────────
    striker_avg = state.partnership.striker_batting_average
    wicket_val  = get_wicket_value(state, striker_avg)

    # ── Run baseline and boundary/overthrow components ────────────────
    run_baseline    = get_run_baseline(event.fielder_position, phase, fmt)
    boundary_comp   = _compute_boundary_component(event)
    overthrow_pen   = _compute_overthrow_penalty(event)
    actual_runs     = float(event.actual_runs_conceded)

    # ─────────────────────────────────────────────────────────────────
    # MAIN FORMULA BRANCHES
    # ─────────────────────────────────────────────────────────────────

    if event.event_type in _DISMISSAL_EVENTS:
        # ── Dismissal attempt (catch, run-out, stumping) ─────────────
        # Expected value = P × (wicket taken) + (1−P) × (no wicket)
        # Wicket value = CRE_before − CRE_after_dismissal
        # Expected if caught:     re_after_dismissal (inning continues with +1 wkt)
        # Expected if dropped:    re_no_dismissal (state unchanged)
        # expected_value = P × re_after_dismissal + (1−P) × re_no_dismissal
        # actual_value   = re_after_dismissal if wicket_taken else re_no_dismissal + actual_runs
        # raw_ERS = expected_value − actual_value + boundary_comp + overthrow_pen

        expected_value = (
            prob       * re_after_dismissal
            + (1 - prob) * (re_no_dismissal + run_baseline)
        )

        if event.wicket_taken:
            actual_value = re_after_dismissal + actual_runs
        else:
            # Fielder attempted but failed (shouldn't reach here via _DISMISSAL_EVENTS
            # unless wicket_taken=False on a catch attempt — treat as dropped)
            actual_value = re_no_dismissal + actual_runs

        raw_ers = (expected_value - actual_value) + boundary_comp + overthrow_pen

        wkt_saved     = wicket_val if event.wicket_taken else 0.0
        boundary_saved = boundary_comp
        overthrow_cost = overthrow_pen

    elif event.event_type in _MISSED_DISMISSAL_EVENTS:
        # ── Missed dismissal (dropped catch, run-out miss) ────────────
        # The fielder had a chance (high P) but failed.
        # ERS = what we expected − what happened (batter survives → negative)
        expected_value = (
            prob       * re_after_dismissal
            + (1 - prob) * (re_no_dismissal + run_baseline)
        )
        # Batter survived; runs conceded as well
        actual_value = re_no_dismissal + actual_runs

        raw_ers = (expected_value - actual_value) + boundary_comp + overthrow_pen

        wkt_saved     = 0.0  # no wicket taken
        boundary_saved = boundary_comp
        overthrow_cost = overthrow_pen

    elif event.event_type == FieldingEventType.BOUNDARY_MISSED:
        # ── Boundary missed ──────────────────────────────────────────
        # Baseline already bakes in a probability of a 4 at this position.
        # Actual = 4 runs. Extra cost = 4 - run_baseline (if 4 > baseline).
        expected_value = run_baseline
        actual_value   = actual_runs  # should be 4
        raw_ers        = expected_value - actual_value + overthrow_pen

        wkt_saved      = 0.0
        boundary_saved = 0.0
        overthrow_cost = overthrow_pen

    elif event.event_type in {
        FieldingEventType.BOUNDARY_STOP,
        FieldingEventType.BOUNDARY_SAVED_6,
    }:
        # ── Boundary prevented ───────────────────────────────────────
        # run_baseline might assume some chance of 4; boundary_comp captures exact saving.
        expected_value = run_baseline
        actual_value   = actual_runs
        raw_ers        = expected_value - actual_value + boundary_comp + overthrow_pen

        wkt_saved      = 0.0
        boundary_saved = boundary_comp
        overthrow_cost = overthrow_pen

    elif event.event_type in {
        FieldingEventType.MISFIELD_RUNS,
        FieldingEventType.RELAY_THROW_POOR,
    }:
        # ── Misfield / poor relay ────────────────────────────────────
        expected_value = run_baseline
        actual_value   = actual_runs
        raw_ers        = expected_value - actual_value + overthrow_pen

        wkt_saved      = 0.0
        boundary_saved = 0.0
        overthrow_cost = overthrow_pen

    elif event.event_type == FieldingEventType.OVERTHROW:
        # ── Overthrow ────────────────────────────────────────────────
        # Primary action succeeded (threw to stumps); overthrow is the penalty.
        # Base ERS for the throw itself is 0 (average play); penalty is the overthrow.
        raw_ers = overthrow_pen

        expected_value = 0.0
        actual_value   = float(event.overthrow_runs)
        wkt_saved      = 0.0
        boundary_saved = 0.0
        overthrow_cost = overthrow_pen

    elif event.event_type == FieldingEventType.OBSTRUCTING_FIELD:
        # ── Obstruction ──────────────────────────────────────────────
        # Fielder illegally blocked batter → umpire awards 5 penalty runs.
        # ERS = -(5 penalty runs above baseline)
        raw_ers        = run_baseline - (actual_runs + 5.0)
        expected_value = run_baseline
        actual_value   = actual_runs + 5.0
        wkt_saved      = 0.0
        boundary_saved = 0.0
        overthrow_cost = 0.0

    else:
        # ── Generic (diving stop, relay good, ground fielding) ───────
        expected_value = run_baseline
        actual_value   = actual_runs
        raw_ers        = expected_value - actual_value + boundary_comp + overthrow_pen

        wkt_saved      = wicket_val if event.wicket_taken else 0.0
        boundary_saved = boundary_comp
        overthrow_cost = overthrow_pen

    # ─────────────────────────────────────────────────────────────────
    # Leverage adjustment
    # ─────────────────────────────────────────────────────────────────
    lev_ers = raw_ers * li

    # ─────────────────────────────────────────────────────────────────
    # Assemble output structures
    # ─────────────────────────────────────────────────────────────────
    expected_vs_actual = ExpectedVsActual(
        expected_runs_avg_fielder = round(expected_value, 3),
        actual_runs_allowed       = round(actual_value, 3),
        runs_differential         = round(expected_value - actual_value, 3),
        wicket_value_saved        = round(wkt_saved, 3),
        boundary_value_saved      = round(boundary_saved, 3),
        overthrow_penalty         = round(overthrow_cost, 3),
    )

    narrative = build_narrative(
        event.event_type, event.fielder_position,
        prob, li, lev_ers,
        event.ball_metrics,
        event.boundary_prevented, event.six_prevented,
    )

    return ERSEventBreakdown(
        event_type                = event.event_type.value,
        fielder_position          = event.fielder_position.value,
        probability_used          = prob,
        leverage_index            = li,
        raw_ers                   = round(raw_ers, 4),
        leverage_adjusted_ers     = round(lev_ers, 4),
        expected_vs_actual        = expected_vs_actual,
        narrative                 = narrative,
    ), re_before, re_after_dismissal, re_no_dismissal, wicket_val, \
       expected_value, actual_value, boundary_comp, raw_ers, lev_ers


# ─────────────────────────────────────────────
# Top-level calculation
# ─────────────────────────────────────────────

def calculate_ers(request: ERSRequest) -> ERSResponse:
    """
    Main entry point. Processes all fielding events on a single ball,
    accumulates ERS, builds derivation proof, and returns full response.
    """
    state = request.match_state
    breakdowns: list[ERSEventBreakdown] = []

    total_raw_ers = 0.0
    total_lev_ers = 0.0
    min_prob      = 1.0

    # Store derivation data from the first (or most significant) event
    first_derivation_data: dict | None = None

    for event in request.fielding_events:
        li = compute_leverage_index(state)

        result = _compute_event_ers(event, state, li)
        (
            bd, re_before, re_after_d, re_no_d,
            wicket_val, exp_val, act_val,
            boundary_comp, raw_ers, lev_ers
        ) = result

        breakdowns.append(bd)
        total_raw_ers += raw_ers
        total_lev_ers += lev_ers
        min_prob       = min(min_prob, bd.probability_used)

        if first_derivation_data is None:
            first_derivation_data = {
                "re_before":          re_before,
                "re_after_dismissal": re_after_d,
                "re_no_dismissal":    re_no_d,
                "wicket_val":         wicket_val,
                "prob":               bd.probability_used,
                "expected_val":       exp_val,
                "actual_val":         act_val,
                "boundary_comp":      boundary_comp,
                "li":                 li,
                "raw_ers":            raw_ers,
                "final_ers":          lev_ers,
            }

    # Build derivation for the primary event
    deriv_data = first_derivation_data or {}
    deriv_strings = build_derivation_strings(
        state,
        deriv_data.get("re_before", 0.0),
        deriv_data.get("re_after_dismissal", 0.0),
        deriv_data.get("re_no_dismissal", 0.0),
        deriv_data.get("wicket_val", 0.0),
        deriv_data.get("prob", 0.0),
        deriv_data.get("expected_val", 0.0),
        deriv_data.get("actual_val", 0.0),
        deriv_data.get("boundary_comp", 0.0),
        deriv_data.get("li", 1.0),
        total_raw_ers,
        total_lev_ers,
    )
    formula_derivation = FormulaDerivation(**deriv_strings)

    grade, pct = compute_grade_and_percentile(total_lev_ers, min_prob)

    interpretation = _build_interpretation(
        total_lev_ers, grade, breakdowns, state, min_prob
    )

    return ERSResponse(
        ball_id             = request.ball_id,
        fielder_id          = request.fielder_id,
        format              = state.format.value,
        phase               = state.phase.value,
        expected_runs_saved = round(total_lev_ers, 4),
        raw_runs_saved      = round(total_raw_ers, 4),
        event_breakdown     = breakdowns,
        formula_derivation  = formula_derivation,
        grade               = grade,
        interpretation      = interpretation,
        percentile_estimate = pct,
    )


# ─────────────────────────────────────────────
# Interpretation builder
# ─────────────────────────────────────────────

def _build_interpretation(
    ers: float,
    grade: str,
    breakdowns: list[ERSEventBreakdown],
    state: MatchState,
    min_prob: float,
) -> str:
    quality = (
        "Elite, game-changing play"    if ers >= 2.50 else
        "Excellent fielding"           if ers >= 1.50 else
        "Above-average play"           if ers >= 0.80 else
        "Slightly above average"       if ers >= 0.30 else
        "Average fielding"             if ers >= -0.10 else
        "Below-average fielding"       if ers >= -0.60 else
        "Poor fielding — cost the team significantly"
    )

    difficulty = (
        "near-impossible" if min_prob < 0.15 else
        "very difficult"  if min_prob < 0.30 else
        "difficult"       if min_prob < 0.50 else
        "moderate"        if min_prob < 0.70 else
        "routine"
    )

    events_str = " + ".join(bd.event_type for bd in breakdowns)
    fmt_str    = state.format.value
    phase_str  = state.phase.value

    return (
        f"[{grade}] {quality} — {events_str} in {fmt_str} {phase_str} overs. "
        f"Difficulty: {difficulty} (min P={min_prob:.0%}). "
        f"ERS = {ers:+.3f} leverage-adjusted runs vs league-average baseline."
    )