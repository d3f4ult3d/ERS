"""
schemas.py — Pydantic v2 request/response models for the Cricket
Expected Runs Saved (ERS) API.

Cricket fielding events differ fundamentally from baseball:
  - Balls are NOT dead on contact; fielders can affect runs even on hits
  - Boundaries (4s and 6s) are the extreme events fielders can prevent
  - Run-scoring is continuous; every misfield potentially adds runs
  - Wickets (dismissals) via caught, run-out, stumping all have fielder agency
  - The "state" is: balls remaining in over, partnership, required run-rate
  - Pitch conditions, match format (T20/ODI/Test) and phase heavily affect baselines
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class MatchFormat(str, Enum):
    T20  = "T20"    # 20 overs; ~180-200 expected total
    ODI  = "ODI"    # 50 overs; ~280-320 expected total
    TEST = "Test"   # Unlimited overs; per-session context


class InningsPhase(str, Enum):
    """
    Over ranges that define the fielding-restriction and tactical context.
    Powerplay: mandatory ring-up, batters score freely.
    Middle:    settled partnership phase, dot-ball pressure matters.
    Death:     final overs, batters swing; fielders at boundary.
    """
    POWERPLAY = "powerplay"   # T20: 1-6, ODI: 1-10, Test: first session
    MIDDLE    = "middle"      # T20: 7-15, ODI: 11-40
    DEATH     = "death"       # T20: 16-20, ODI: 41-50, Test: declaration
    CHASE     = "chase"       # Explicitly a run chase (affects LI)


class FieldingEventType(str, Enum):
    """Every distinct action (or failure) a fielder can perform."""
    CATCH               = "catch"                # Caught out attempt
    CATCH_DROPPED       = "catch_dropped"        # Missed catch; batter survives
    RUN_OUT_DIRECT      = "run_out_direct"       # Direct-hit run-out
    RUN_OUT_RELAY       = "run_out_relay"        # Relay throw completes run-out
    RUN_OUT_MISS        = "run_out_miss"         # Attempted run-out, batter/runner safe
    STUMPING            = "stumping"             # Keeper stump off missed delivery
    BOUNDARY_STOP       = "boundary_stop"        # Prevented 4; ball retrieved at rope
    BOUNDARY_SAVED_6    = "boundary_saved_6"     # Exceptional save; would-be 6 becomes fewer
    BOUNDARY_MISSED     = "boundary_missed"      # Fielder failed to stop a 4
    MISFIELD_RUNS       = "misfield_runs"        # Ball misfielded; extra runs conceded
    DIVING_STOP         = "diving_stop"          # Stop in the field; no wicket
    RELAY_THROW_GOOD    = "relay_throw_good"     # Accurate relay restricts runs
    RELAY_THROW_POOR    = "relay_throw_poor"     # Poor relay; extra run taken
    GROUND_FIELDING     = "ground_fielding"      # Standard clean pickup-and-return
    OVERTHROW           = "overthrow"            # Errant throw goes to boundary
    OBSTRUCTING_FIELD   = "obstructing_field"    # Fielder illegally obstructs batter


class FieldingPosition(str, Enum):
    """Standard fielding positions in cricket."""
    WICKET_KEEPER   = "wicket_keeper"
    SLIP_1          = "first_slip"
    SLIP_2          = "second_slip"
    SLIP_3          = "third_slip"
    GULLY           = "gully"
    POINT           = "point"
    COVER_POINT     = "cover_point"
    COVER           = "cover"
    MID_OFF         = "mid_off"
    MID_ON          = "mid_on"
    MID_WICKET      = "mid_wicket"
    SQUARE_LEG      = "square_leg"
    FINE_LEG        = "fine_leg"
    THIRD_MAN       = "third_man"
    LONG_ON         = "long_on"
    LONG_OFF        = "long_off"
    DEEP_MID_WICKET = "deep_mid_wicket"
    DEEP_SQUARE_LEG = "deep_square_leg"
    DEEP_COVER      = "deep_cover"
    DEEP_POINT      = "deep_point"
    SHORT_LEG       = "short_leg"
    SILLY_MID_ON    = "silly_mid_on"
    FORWARD_SHORT   = "forward_short_leg"


class BallTrajectory(str, Enum):
    """How the ball reached the fielder."""
    EDGE_BEHIND      = "edge_behind"       # Edge to keeper/slips
    MISTIMED_AERIAL  = "mistimed_aerial"   # Aerial ball off mistimed shot
    HARD_DRIVE       = "hard_drive"        # Powerfully struck on the ground
    LOFTED_SHOT      = "lofted_shot"       # Intentional aerial (six/four attempt)
    GLANCE_DEFLECT   = "glance_deflect"    # Deflection off bat/pad
    DIRECT_HIT_CHANCE= "direct_hit_chance" # Run-out opportunity mid-pitch
    ROLLING_GROUND   = "rolling_ground"    # Ball rolling along the ground
    FULL_TOSS_PULL   = "full_toss_pull"    # Full toss, batting attack
    NOT_APPLICABLE   = "not_applicable"


class PitchCondition(str, Enum):
    """Affects both ball movement and fielding difficulty."""
    DRY_DUSTY  = "dry_dusty"    # Turning/stopping track; uneven bounce
    GREEN_TOP  = "green_top"    # Seaming; low bounce; slippery outfield
    FLAT       = "flat"         # High-scoring; easy outfield
    DAMP       = "damp"         # Wet outfield; ball skids
    WORN       = "worn"         # Late-Test worn surface


# ─────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────

class BallMetrics(BaseModel):
    """
    Ball-tracking / hawk-eye style data.
    All fields optional — richer data = more precise ERS.
    """
    speed_kmh: Optional[float] = Field(
        None, ge=0.0, le=200.0,
        description="Ball speed off bat or from bowler in km/h"
    )
    hang_time_seconds: Optional[float] = Field(
        None, ge=0.0, le=8.0,
        description="Air time for aerial balls (catches, boundaries)"
    )
    distance_to_fielder_metres: Optional[float] = Field(
        None, ge=0.0, le=90.0,
        description="Distance fielder had to cover to reach the ball"
    )
    height_off_ground_metres: Optional[float] = Field(
        None, ge=0.0, le=30.0,
        description="Ball height at point of catch/stop attempt"
    )
    estimated_carry_metres: Optional[float] = Field(
        None, ge=0.0, le=120.0,
        description="Estimated distance ball would travel to boundary"
    )
    impact_zone: Optional[str] = Field(
        None,
        description="Zone code: 'infield', 'circle', 'deep', 'boundary_rope'"
    )


class PartnershipState(BaseModel):
    """
    Current batting partnership context.
    Wicket value is central to cricket's run-expectancy model.
    """
    wickets_fallen: int = Field(..., ge=0, le=9,
        description="Wickets fallen before this ball")
    current_partnership_runs: int = Field(0, ge=0,
        description="Runs in the current partnership")
    striker_batting_average: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Striker's career/tournament batting average"
    )
    non_striker_batting_average: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Non-striker's career/tournament batting average"
    )
    striker_strike_rate: Optional[float] = Field(
        None, ge=0.0, le=300.0,
        description="Striker's current-innings or career strike rate"
    )


class MatchState(BaseModel):
    """
    The full match state at the moment of the fielding event.
    This is the run-expectancy anchor — equivalent to base-out state in baseball.
    """
    format: MatchFormat
    innings: int = Field(..., ge=1, le=4, description="1st, 2nd (ODI/T20) or 1st-4th (Test)")
    over: int = Field(..., ge=0, description="Current over number (0-indexed)")
    ball_in_over: int = Field(..., ge=1, le=6, description="Ball number within the over (1-6)")
    phase: InningsPhase
    runs_scored: int = Field(..., ge=0, description="Runs scored so far this innings")
    target: Optional[int] = Field(
        None, ge=0,
        description="Target (second innings only). None = first innings / Test first innings."
    )
    pitch_condition: PitchCondition = PitchCondition.FLAT
    partnership: PartnershipState
    batting_team_win_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Pre-ball win probability for batting team"
    )

    @model_validator(mode="after")
    def validate_target_logic(self) -> "MatchState":
        if self.target is not None and self.target < self.runs_scored:
            raise ValueError(
                "runs_scored cannot exceed target — the innings would be over."
            )
        return self

    @property
    def balls_remaining(self) -> int:
        """Balls remaining in the innings (approximate for Test)."""
        if self.format == MatchFormat.T20:
            total_balls = 120
        elif self.format == MatchFormat.ODI:
            total_balls = 300
        else:
            return 999  # Test: effectively unlimited

        balls_bowled = self.over * 6 + self.ball_in_over
        return max(0, total_balls - balls_bowled)

    @property
    def required_run_rate(self) -> Optional[float]:
        """Required run-rate for the batting team (second innings only)."""
        if self.target is None:
            return None
        runs_needed = self.target - self.runs_scored
        balls = self.balls_remaining
        if balls == 0:
            return float("inf")
        return round((runs_needed / balls) * 6, 2)


# ─────────────────────────────────────────────
# Fielding Event Detail
# ─────────────────────────────────────────────

class FieldingEventDetail(BaseModel):
    """
    One discrete fielding action on a single ball.
    Multiple events can chain within one ball (e.g., drop → overthrow).
    """
    event_type: FieldingEventType
    fielder_position: FieldingPosition
    ball_trajectory: BallTrajectory = BallTrajectory.NOT_APPLICABLE
    ball_metrics: Optional[BallMetrics] = None

    # Outcome
    wicket_taken: bool = Field(False, description="Did this event result in a dismissal?")
    actual_runs_conceded: int = Field(
        0, ge=0, le=7,
        description="Runs scored on this ball AFTER the fielding action (0-7 including extras)"
    )
    boundary_prevented: bool = Field(
        False, description="Did the fielder prevent what would have been a boundary (4)?"
    )
    six_prevented: bool = Field(
        False, description="Did the fielder prevent what would have been a six (6)?"
    )
    overthrow_runs: int = Field(
        0, ge=0, le=4,
        description="Extra runs conceded due to an overthrow on this ball"
    )

    # Difficulty
    catch_or_run_out_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Model-estimated probability an average fielder in this position "
                    "converts this chance (0-1). If None, inferred from ball metrics."
    )
    is_pressure_moment: bool = Field(
        False, description="Human tag: high-pressure context (e.g., last over, set batter)."
    )

    @model_validator(mode="after")
    def validate_boundary_flags(self) -> "FieldingEventDetail":
        if self.boundary_prevented and self.six_prevented:
            raise ValueError(
                "boundary_prevented (4) and six_prevented (6) cannot both be True "
                "on the same event."
            )
        if self.wicket_taken and self.actual_runs_conceded > 1:
            # On a wicket ball, batters may complete a run before dismissal
            # but more than 1 run is unusual; flag it
            pass  # valid but uncommon — do not block
        return self


# ─────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────

class ERSRequest(BaseModel):
    """Full payload for one ball's fielding action(s)."""
    ball_id: Optional[str] = Field(
        None, description="Caller-supplied identifier e.g. 'IND-AUS-ODI3-Inn1-Ov32-B4'"
    )
    fielder_id: Optional[str] = Field(None, description="Fielder name or ID")
    match_state: MatchState
    fielding_events: list[FieldingEventDetail] = Field(
        ..., min_length=1, max_length=5,
        description="All fielding events on this ball (e.g., drop then overthrow = 2 events)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ball_id": "IND-AUS-T20-Inn1-Ov18-B3",
                "fielder_id": "kohli_18",
                "match_state": {
                    "format": "T20",
                    "innings": 1,
                    "over": 17,
                    "ball_in_over": 3,
                    "phase": "death",
                    "runs_scored": 148,
                    "target": None,
                    "pitch_condition": "flat",
                    "partnership": {
                        "wickets_fallen": 3,
                        "current_partnership_runs": 62,
                        "striker_batting_average": 52.4,
                        "striker_strike_rate": 142.0
                    },
                    "batting_team_win_probability": 0.68
                },
                "fielding_events": [
                    {
                        "event_type": "boundary_stop",
                        "fielder_position": "long_on",
                        "ball_trajectory": "lofted_shot",
                        "ball_metrics": {
                            "speed_kmh": 118.0,
                            "hang_time_seconds": 3.2,
                            "distance_to_fielder_metres": 8.0,
                            "estimated_carry_metres": 67.0
                        },
                        "wicket_taken": False,
                        "actual_runs_conceded": 2,
                        "boundary_prevented": True,
                        "six_prevented": False,
                        "overthrow_runs": 0,
                        "catch_or_run_out_probability": None,
                        "is_pressure_moment": True
                    }
                ]
            }
        }


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────

class ExpectedVsActual(BaseModel):
    """
    The heart of the counterfactual.
    Shows exactly what an average fielder was expected to allow vs what happened.
    """
    expected_runs_avg_fielder: float = Field(
        description="Runs an average fielder in this position/format/phase would allow"
    )
    actual_runs_allowed: float = Field(
        description="Runs actually allowed after this fielder's action"
    )
    runs_differential: float = Field(
        description="expected_runs_avg_fielder − actual_runs_allowed. "
                    "Positive = fielder saved runs. Negative = fielder cost runs."
    )
    wicket_value_saved: float = Field(
        description="Run-value equivalent of the wicket (if applicable), else 0.0"
    )
    boundary_value_saved: float = Field(
        description="Run-value of boundary/six prevented, else 0.0"
    )
    overthrow_penalty: float = Field(
        description="Run-cost of any overthrows conceded (always <= 0)"
    )


class FormulaDerivation(BaseModel):
    """
    Step-by-step derivation of ERS for transparency and auditability.
    Every term is labelled and explained.
    """
    step_1_baseline_re: str
    step_2_catch_probability: str
    step_3_wicket_value: str
    step_4_boundary_component: str
    step_5_run_restriction_component: str
    step_6_leverage_index: str
    step_7_final_ers: str
    formula_summary: str


class ERSEventBreakdown(BaseModel):
    """Per-event contribution to total ERS."""
    event_type: str
    fielder_position: str
    probability_used: float
    leverage_index: float
    raw_ers: float
    leverage_adjusted_ers: float
    expected_vs_actual: ExpectedVsActual
    narrative: str


class ERSResponse(BaseModel):
    """Full API response."""
    ball_id: Optional[str]
    fielder_id: Optional[str]
    format: str
    phase: str

    # Core metric
    expected_runs_saved: float = Field(
        description="Total leverage-adjusted ERS. Positive = runs saved; negative = runs cost."
    )
    raw_runs_saved: float = Field(
        description="Sum of raw (pre-leverage) runs saved across all events."
    )

    # Supporting
    event_breakdown: list[ERSEventBreakdown]
    formula_derivation: FormulaDerivation
    grade: str
    interpretation: str
    percentile_estimate: Optional[float] = Field(None, ge=0.0, le=100.0)

    class Config:
        json_schema_extra = {
            "example": {
                "ball_id": "IND-AUS-T20-Inn1-Ov18-B3",
                "expected_runs_saved": 1.82,
                "raw_runs_saved": 1.52,
                "grade": "A+",
                "interpretation": "Elite boundary save during death overs. Prevented a near-certain 4 (2 runs saved) in a high-leverage T20 death phase, worth 1.82 leverage-adjusted runs."
            }
        }