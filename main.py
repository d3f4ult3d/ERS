"""
main.py — FastAPI application for the Cricket Expected Runs Saved (ERS) API.

Endpoints
---------
POST /ers/calculate          ERS for a single ball
POST /ers/batch              ERS for up to 50 balls
GET  /ers/cre-table          Cricket run-expectancy values for inspection
GET  /ers/formula            Full formula derivation documentation
GET  /health                 Health check
GET  /                       API info
"""

from __future__ import annotations

import json
import sys
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from schemas import (
    BallTrajectory,
    ERSRequest,
    ERSResponse,
    FieldingEventType,
    FieldingPosition,
    InningsPhase,
    MatchFormat,
    PitchCondition,
)
from services import calculate_ers
from utils import get_cricket_re, _Z0_PERCENT, _Z1_BALLS, _FORMAT_BASELINE

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Cricket Expected Runs Saved (ERS) API",
    description=(
        "A counterfactual cricket fielding value metric. For every ball, ERS estimates "
        "how many runs a fielder saved (or cost) relative to what a league-average fielder "
        "would have allowed in the identical match state. "
        "Uses DLS-inspired run-expectancy, ball-tracking probability models, "
        "and leverage-index weighting by format / phase / wickets / required-run-rate."
    ),
    version="1.0.0",
    contact={"name": "Cricket Analytics", "email": "analytics@example.com"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "ERS",       "description": "Expected Runs Saved calculation"},
        {"name": "Reference", "description": "Lookup tables and formula documentation"},
        {"name": "Utility",   "description": "Health and metadata"},
    ],
)


# ─────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────

class APIInfo(BaseModel):
    name: str
    version: str
    description: str
    docs: str
    formula: str
    health: str


class BatchERSRequest(BaseModel):
    balls: list[ERSRequest] = Field(..., min_length=1, max_length=50)


class BatchERSResponse(BaseModel):
    count: int
    results: list[ERSResponse]
    total_expected_runs_saved: float
    total_raw_runs_saved: float


class CRERow(BaseModel):
    wickets_fallen: int
    balls_remaining: int
    format: str
    phase: str
    expected_runs: float


class CRETableResponse(BaseModel):
    description: str
    formula: str
    rows: list[CRERow]


class FormulaDocResponse(BaseModel):
    title: str
    overview: str
    steps: list[dict]
    formula_summary: str
    references: list[str]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_model=APIInfo, tags=["Utility"])
async def root() -> APIInfo:
    return APIInfo(
        name="Cricket ERS API",
        version="1.0.0",
        description=(
            "Counterfactual fielding value for cricket. "
            "Positive ERS = fielder outperformed the average. "
            "Negative ERS = fielder underperformed."
        ),
        docs="/docs",
        formula="/ers/formula",
        health="/health",
    )


@app.get("/health", tags=["Utility"])
async def health_check() -> dict:
    return {"status": "ok", "service": "Cricket ERS API", "version": "1.0.0"}


@app.post(
    "/ers/calculate",
    response_model=ERSResponse,
    tags=["ERS"],
    summary="Calculate Expected Runs Saved for a single ball",
)
async def calculate_single(request: ERSRequest) -> ERSResponse:
    """
    ## Cricket ERS — Single Ball

    ### What it measures
    How many runs did this fielder save (or cost) compared to a league-average
    fielder in the **exact same match state**?

    ### Input
    - **match_state**: full context — format (T20/ODI/Test), phase, overs,
      wickets, score, target, pitch conditions, partnership quality
    - **fielding_events**: one or more actions on this ball
      (e.g., drop followed by overthrow = 2 events)

    ### Core calculation (see `/ers/formula` for full derivation)
    ```
    ERS = [P×(wicket_value + CRE_after)
           + (1−P)×CRE_no_dismissal
           − actual_outcome
           + boundary_component
           + overthrow_penalty] × leverage_index
    ```

    ### Interpretation
    - **ERS > 0** → saved runs vs average fielder
    - **ERS < 0** → cost runs vs average fielder
    - **Grade A+** at ≥ 2.50 leverage-adjusted runs saved

    ### Edge cases handled
    - Drop on last ball of a T20 chase (extreme leverage)
    - Overthrow off an otherwise good throw
    - Catch attempt on a tail-end batter (low batter quality multiplier)
    - Missed run-out that costs the only remaining wicket partnership
    - Obstruction (5-penalty-run cost calculated separately)
    - Test innings with effectively unlimited balls
    """
    try:
        return calculate_ers(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calculation error: {exc}")


@app.post(
    "/ers/batch",
    tags=["ERS"],
    summary="Calculate ERS for multiple balls in one request",
)
async def calculate_batch(batch: BatchERSRequest):
    """
    Process up to 50 balls in one request.
    Returns individual results plus aggregated totals.

    Partial failures return HTTP 207 with a `warnings` array.
    Useful for:
    - Full-innings fielding audit
    - Fielder comparison across a match
    - Session-level defensive analysis
    """
    results: list[ERSResponse] = []
    warnings: list[str] = []

    for i, ball in enumerate(batch.balls):
        try:
            results.append(calculate_ers(ball))
        except Exception as exc:
            warnings.append(f"Ball index {i} (id={ball.ball_id}): {exc}")

    if not results:
        raise HTTPException(
            status_code=422,
            detail={"message": "All balls failed to calculate", "errors": warnings},
        )

    total_ers = round(sum(r.expected_runs_saved for r in results), 4)
    total_raw = round(sum(r.raw_runs_saved for r in results), 4)

    response_data = BatchERSResponse(
        count=len(results),
        results=results,
        total_expected_runs_saved=total_ers,
        total_raw_runs_saved=total_raw,
    )

    if warnings:
        return JSONResponse(
            status_code=207,
            content={**response_data.model_dump(), "warnings": warnings},
        )
    return response_data


@app.get(
    "/ers/cre-table",
    response_model=CRETableResponse,
    tags=["Reference"],
    summary="Cricket Run Expectancy (CRE) lookup table",
)
async def get_cre_table(
    format: MatchFormat = Query(MatchFormat.T20, description="Match format"),
    phase:  InningsPhase = Query(InningsPhase.MIDDLE, description="Innings phase"),
    wickets_min: int = Query(0, ge=0, le=9),
    wickets_max: int = Query(9, ge=0, le=9),
    balls_step:  int = Query(10, ge=1, le=30,
                             description="Step size for balls_remaining in the table"),
) -> CRETableResponse:
    """
    Returns CRE values for the requested format/phase/wicket range.

    Use this to understand the run-expectancy baseline driving every ERS calculation.

    CRE = Z0(w) × baseline_total × [1 − exp(−b / Z1(w))] × phase_multiplier

    where Z0 and Z1 are DLS-inspired resource parameters.
    """
    if wickets_min > wickets_max:
        raise HTTPException(status_code=422, detail="wickets_min must be ≤ wickets_max")

    rows: list[CRERow] = []
    max_balls = {MatchFormat.T20: 120, MatchFormat.ODI: 300, MatchFormat.TEST: 450}[format]

    for w in range(wickets_min, wickets_max + 1):
        for b in range(0, max_balls + 1, balls_step):
            cre = get_cricket_re(w, b, format, phase)
            rows.append(CRERow(
                wickets_fallen=w,
                balls_remaining=b,
                format=format.value,
                phase=phase.value,
                expected_runs=cre,
            ))

    return CRETableResponse(
        description=(
            f"Cricket Run Expectancy for {format.value} format, {phase.value} phase. "
            f"Values represent average runs expected to score from this state to end of innings."
        ),
        formula="CRE = Z0(w) × baseline × [1 − exp(−b / Z1(w))] × phase_multiplier",
        rows=rows,
    )


@app.get(
    "/ers/formula",
    response_model=FormulaDocResponse,
    tags=["Reference"],
    summary="Full formula derivation and methodology documentation",
)
async def get_formula_documentation() -> FormulaDocResponse:
    """
    Returns the complete step-by-step derivation of the ERS formula,
    including data sources, assumptions, and references.
    """
    return FormulaDocResponse(
        title="Cricket Expected Runs Saved (ERS) — Formula Derivation",
        overview=(
            "ERS measures how many runs a fielder saved or cost relative to a "
            "league-average fielder in the identical match state. It is a counterfactual "
            "metric: we ask 'what would the average fielder have allowed here?' and "
            "compare that to what actually happened. "
            "The metric combines cricket run-expectancy (CRE), fielding probability models, "
            "wicket value, boundary components, and leverage indexing."
        ),
        steps=[
            {
                "step": 1,
                "name": "Cricket Run Expectancy (CRE)",
                "formula": "CRE(w, b, fmt) = Z0(w) × baseline_total × [1 − exp(−b / Z1(w))] × phase_mult",
                "explanation": (
                    "CRE is the expected runs to score from the current state to the end of the innings. "
                    "w = wickets fallen, b = balls remaining. "
                    "Z0 and Z1 are DLS resource parameters indexed by wickets. "
                    "phase_mult adjusts for powerplay (1.15×), middle (0.90×), death (1.35×). "
                    "Format baseline totals: T20=160, ODI=290, Test=350 runs."
                ),
            },
            {
                "step": 2,
                "name": "Fielding Probability P(success)",
                "formula": "P = baseline(position_zone × trajectory) × speed_adj × hang_adj × distance_adj × height_adj × phase_penalty",
                "explanation": (
                    "P is the probability an average fielder converts this chance. "
                    "Positions are grouped into zones: close (slips/keeper), circle (mid-on/point), "
                    "deep (long-on/fine-leg). Ball trajectory (edge, lofted, hard drive etc.) "
                    "sets the baseline. Adjustments: hard-hit balls (−0.10 to −0.20), "
                    "long hang time (+0.06), far distance (−0.07 to −0.18), "
                    "death-phase pressure (−0.04), run-out base penalty (×0.78)."
                ),
            },
            {
                "step": 3,
                "name": "Wicket Value V(w, b)",
                "formula": "V(w, b) = CRE(w, b) − CRE(w+1, b)  × BQM",
                "explanation": (
                    "The wicket value is the exact run-expectancy swing from losing this wicket in context. "
                    "It is NOT a flat 'wickets are always worth X runs'. "
                    "BQM (Batter Quality Multiplier) = clamp(striker_avg / league_avg, 0.5, 2.0). "
                    "Dismissing a top-order batter (avg 55) is worth ~2× more than a tail-ender (avg 15)."
                ),
            },
            {
                "step": 4,
                "name": "Boundary Component B",
                "formula": "B_4 = 4 − actual_runs  (if boundary_prevented);  B_6 = 6 − actual_runs  (if six_prevented)",
                "explanation": (
                    "Exact run saving from a boundary stop. "
                    "A fielder who prevents a 4 and concedes 2 saves exactly 2 runs. "
                    "A six saved that became a single saves 5 runs. "
                    "No boundary action → B = 0."
                ),
            },
            {
                "step": 5,
                "name": "Run Restriction Component R",
                "formula": "R = run_baseline(position, phase, format) − actual_runs_conceded + overthrow_penalty",
                "explanation": (
                    "For non-boundary, non-wicket balls, value comes from allowing fewer runs than baseline. "
                    "Position-zone × phase baselines (IPL/ICC 2018-2023): "
                    "deep/death: 2.6 runs/ball, circle/powerplay: 0.9, close/middle: 0.2. "
                    "Overthrow penalty = −overthrow_runs (always ≤ 0). "
                    "Obstruction: −5 penalty runs above baseline."
                ),
            },
            {
                "step": 6,
                "name": "Combining Components (dismissal chance)",
                "formula": (
                    "raw_ERS = [P × CRE_after_dismissal + (1−P) × (CRE_no_dismissal + run_baseline)] "
                    "− actual_outcome + B + overthrow_penalty"
                ),
                "explanation": (
                    "For dismissal events: expected value is the probability-weighted CRE. "
                    "For missed dismissals: the fielder had P probability of taking a wicket "
                    "but failed — the lost run-expectancy swing is penalised. "
                    "For boundary/run-restriction events: simpler baseline comparison."
                ),
            },
            {
                "step": 7,
                "name": "Leverage Index (LI)",
                "formula": "LI = phase_weight × wicket_scarcity × rrr_pressure × format_weight",
                "explanation": (
                    "LI scales ERS by how important this ball was to the match outcome. "
                    "phase_weight: death=1.60, powerplay=1.10, middle=0.85. "
                    "wicket_scarcity: 1.0 (0 wkts) → 1.80 (9 wkts). "
                    "rrr_pressure: RRR≥12 → 1.40, RRR≤4 → 0.70. "
                    "format_weight: T20=1.20, ODI=1.00, Test=0.55. "
                    "Clamped to [0.10, 5.0]."
                ),
            },
            {
                "step": 8,
                "name": "Final ERS",
                "formula": "ERS = raw_ERS × LI",
                "explanation": (
                    "ERS > 0: fielder saved runs vs average. "
                    "ERS < 0: fielder cost runs vs average. "
                    "Grade A+: ≥2.50 ERS (e.g., caught-and-bowled in death overs off a hard drive). "
                    "Grade F: ≤ −0.60 ERS (e.g., dropped sitter on last ball of T20)."
                ),
            },
        ],
        formula_summary=(
            "ERS = { P×[CRE(w+1,b)] + (1−P)×[CRE(w,b) + R] − actual_outcome + B } × LI\n"
            "where:\n"
            "  P   = P(fielder successfully converts the chance)\n"
            "  CRE = Cricket Run Expectancy from DLS resource model\n"
            "  w,b = wickets fallen, balls remaining\n"
            "  R   = run-restriction baseline for this position/phase/format\n"
            "  B   = boundary component (0 if no boundary involved)\n"
            "  LI  = leverage index\n"
        ),
        references=[
            "Duckworth, Lewis & Stern (2004) 'A fair method of resetting the target in interrupted one-day cricket matches', J. Operational Research Society 55(6)",
            "Preston & Thomas (2000) 'Batting strategy in limited overs cricket', The Statistician 49(1)",
            "Lemmer (2008) 'An analysis of players' performances in the first cricket Twenty20 World Cup series', South African Journal for Research in Sport",
            "ESPNcricinfo / Hawk-Eye ball-tracking data 2018-2023 (T20I, ODI, IPL averages)",
            "ICC Playing Conditions 2023 — fielding restrictions and Powerplay regulations",
        ],
    )


# ─────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────

@app.exception_handler(422)
async def validation_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
            "hint": "See /docs for full schema with examples, or /ers/formula for methodology.",
        },
    )


# ─────────────────────────────────────────────
# Raw terminal input
# ─────────────────────────────────────────────

def _prompt_text(label: str, default: Optional[str] = None) -> Optional[str]:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def _prompt_number(label: str, cast, default=None):
    while True:
        raw = _prompt_text(label, None if default is None else str(default))
        if raw is None or raw == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            print("Enter a valid number.")


def _prompt_bool(label: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    while True:
        raw = (_prompt_text(f"{label} (y/n)", default_text) or default_text).lower()
        if raw in {"y", "yes", "true", "1"}:
            return True
        if raw in {"n", "no", "false", "0"}:
            return False
        print("Enter y or n.")


def _prompt_enum(label: str, enum_cls):
    choices = list(enum_cls)
    print(f"\n{label} choices:")
    for idx, choice in enumerate(choices, start=1):
        print(f"  {idx}. {choice.value}")

    while True:
        raw = input(f"{label}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1].value
        try:
            return enum_cls(raw).value
        except ValueError:
            print("Choose a listed number or exact value.")


def _prompt_optional_float(label: str):
    raw = _prompt_text(label, "")
    if raw in {None, ""}:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, leaving blank.")
        return None


def _prompt_ball_metrics() -> Optional[dict]:
    if not _prompt_bool("Add ball-tracking metrics", False):
        return None

    metrics = {
        "speed_kmh": _prompt_optional_float("Ball speed km/h (blank if unknown)"),
        "hang_time_seconds": _prompt_optional_float("Hang time seconds (blank if unknown)"),
        "distance_to_fielder_metres": _prompt_optional_float("Distance to fielder metres (blank if unknown)"),
        "height_off_ground_metres": _prompt_optional_float("Height off ground metres (blank if unknown)"),
        "estimated_carry_metres": _prompt_optional_float("Estimated carry metres (blank if unknown)"),
        "impact_zone": _prompt_text("Impact zone (blank if unknown)", ""),
    }
    return {key: value for key, value in metrics.items() if value not in {None, ""}}


def run_raw_input_cli() -> None:
    print("\nCricket ERS raw input calculator")
    print("Press Enter on optional fields to skip them.\n")

    match_state = {
        "format": _prompt_enum("Match format", MatchFormat),
        "innings": _prompt_number("Innings", int, 1),
        "over": _prompt_number("Over number, 0-indexed", int, 0),
        "ball_in_over": _prompt_number("Ball in over (1-6)", int, 1),
        "phase": _prompt_enum("Innings phase", InningsPhase),
        "runs_scored": _prompt_number("Runs scored so far", int, 0),
        "target": _prompt_number("Target (blank for first innings)", int, None),
        "pitch_condition": _prompt_enum("Pitch condition", PitchCondition),
        "partnership": {
            "wickets_fallen": _prompt_number("Wickets fallen", int, 0),
            "current_partnership_runs": _prompt_number("Current partnership runs", int, 0),
            "striker_batting_average": _prompt_optional_float("Striker batting average (blank if unknown)"),
            "non_striker_batting_average": _prompt_optional_float("Non-striker batting average (blank if unknown)"),
            "striker_strike_rate": _prompt_optional_float("Striker strike rate (blank if unknown)"),
        },
        "batting_team_win_probability": _prompt_optional_float("Batting team win probability 0-1 (blank if unknown)"),
    }

    event_count = _prompt_number("Number of fielding events on this ball (1-5)", int, 1)
    fielding_events = []
    for event_idx in range(1, event_count + 1):
        print(f"\nFielding event {event_idx}")
        fielding_events.append({
            "event_type": _prompt_enum("Event type", FieldingEventType),
            "fielder_position": _prompt_enum("Fielder position", FieldingPosition),
            "ball_trajectory": _prompt_enum("Ball trajectory", BallTrajectory),
            "ball_metrics": _prompt_ball_metrics(),
            "wicket_taken": _prompt_bool("Wicket taken", False),
            "actual_runs_conceded": _prompt_number("Actual runs conceded", int, 0),
            "boundary_prevented": _prompt_bool("Boundary 4 prevented", False),
            "six_prevented": _prompt_bool("Six prevented", False),
            "overthrow_runs": _prompt_number("Overthrow runs", int, 0),
            "catch_or_run_out_probability": _prompt_optional_float("Known chance probability 0-1 (blank to infer)"),
            "is_pressure_moment": _prompt_bool("Pressure moment", False),
        })

    try:
        request = ERSRequest(
            ball_id=_prompt_text("\nBall ID (blank optional)", ""),
            fielder_id=_prompt_text("Fielder ID/name (blank optional)", ""),
            match_state=match_state,
            fielding_events=fielding_events,
        )
        response = calculate_ers(request)
    except ValidationError as exc:
        print("\nInput validation error:")
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"- {field}: {error['msg']}")
        return

    print("\nERS output")
    print(f"Expected runs saved: {response.expected_runs_saved}")
    print(f"Raw runs saved: {response.raw_runs_saved}")
    print(f"Grade: {response.grade}")
    print(f"Interpretation: {response.interpretation}")
    print("\nFull response:")
    print(json.dumps(response.model_dump(mode="json"), indent=2))


# ─────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if "--api" in sys.argv:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_raw_input_cli()
