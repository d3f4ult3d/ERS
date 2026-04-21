# Cricket Expected Runs Saved (ERS) API

A FastAPI service that estimates **how many runs a cricket fielder saved (or cost)** relative to what a league-average fielder would have allowed in the exact same match state.

---

## Why Counterfactual?

Raw cricket fielding stats — catches taken, run-outs, dropped catches — say nothing about *context*:

- Dropping a dolly at slip in the first over of a Test is not the same as dropping a sitter off the last ball of a T20 chase.
- A boundary save at long-on in the death overs is worth more than the same stop in the middle overs.
- A run-out from the deep that dismisses a set batter (avg 55) is worth twice what dismissing a tailender (avg 10) is.

ERS asks: *"What would a league-average fielder have allowed in this exact situation — and how does this fielder compare?"*

---

## Formula Derivation

See `/ers/formula` at runtime for the fully documented derivation. Summary:

```
ERS = { P × CRE(w+1, b) + (1−P) × [CRE(w, b) + R] − actual_outcome + B } × LI
```

| Symbol | Meaning |
|--------|---------|
| `P` | Probability average fielder converts this chance |
| `CRE(w, b)` | Cricket Run Expectancy: avg runs from (wickets, balls) state |
| `w, b` | Wickets fallen, balls remaining |
| `R` | Run-restriction baseline for this position/phase/format |
| `B` | Boundary component (4 or 6 prevented) |
| `LI` | Leverage Index |

### Step-by-step

**Step 1 — Cricket Run Expectancy (CRE)**

Uses a DLS-inspired model:
```
CRE(w, b) = Z0(w) × baseline_total × [1 − exp(−b / Z1(w))] × phase_multiplier
```
Z0 and Z1 are DLS resource parameters. Baselines: T20=160, ODI=290, Test=350 runs.

**Step 2 — Fielding Probability P**

Position zone (close/circle/deep) × trajectory (edge, lofted, hard drive…) baseline, adjusted for ball speed, hang time, distance, height, phase pressure, and run-out difficulty.

**Step 3 — Wicket Value V(w, b)**
```
V(w, b) = CRE(w, b) − CRE(w+1, b)  ×  BQM
```
BQM (Batter Quality Multiplier) = `clamp(striker_avg / league_avg, 0.5, 2.0)`. Dismissing a high-average batter is worth proportionally more.

**Step 4 — Boundary Component B**
```
B = 4 − actual_runs   (if boundary_prevented)
B = 6 − actual_runs   (if six_prevented)
B = 0                 (otherwise)
```

**Step 5 — Run Restriction R**

Phase-zone baselines from IPL/ICC 2018-2023 averages. Overthrows and obstructions apply direct penalties.

**Step 6 — Leverage Index (LI)**
```
LI = phase_weight × wicket_scarcity × rrr_pressure × format_weight
```

**Step 7 — Final ERS**
```
ERS = raw_ERS × LI
```

---

## Project Structure

```
cricket_ers/
├── main.py          # FastAPI routes and application
├── schemas.py       # Pydantic v2 request/response models
├── services.py      # ERS calculation engine (all event type logic)
├── utils.py         # CRE table, probability inference, LI, grading, formula derivation
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Docs: http://localhost:8000/docs  
Formula: http://localhost:8000/ers/formula  
CRE Table: http://localhost:8000/ers/cre-table

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ers/calculate` | ERS for a single ball |
| `POST` | `/ers/batch` | ERS for up to 50 balls |
| `GET`  | `/ers/cre-table` | Cricket Run Expectancy table |
| `GET`  | `/ers/formula` | Full formula documentation |
| `GET`  | `/health` | Health check |

---

## Example Request — Boundary Save in T20 Death Overs

```bash
curl -X POST http://localhost:8000/ers/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "ball_id": "IND-AUS-T20-Inn1-Ov18-B3",
    "fielder_id": "kohli_18",
    "match_state": {
      "format": "T20",
      "innings": 1,
      "over": 17,
      "ball_in_over": 3,
      "phase": "death",
      "runs_scored": 148,
      "pitch_condition": "flat",
      "partnership": {
        "wickets_fallen": 3,
        "current_partnership_runs": 62,
        "striker_batting_average": 52.4,
        "striker_strike_rate": 142.0
      }
    },
    "fielding_events": [{
      "event_type": "boundary_stop",
      "fielder_position": "long_on",
      "ball_trajectory": "lofted_shot",
      "ball_metrics": {
        "speed_kmh": 118.0,
        "hang_time_seconds": 3.2,
        "distance_to_fielder_metres": 8.0,
        "estimated_carry_metres": 67.0
      },
      "actual_runs_conceded": 2,
      "boundary_prevented": true,
      "overthrow_runs": 0
    }]
  }'
```

### Example Request — Dropped Catch on a Set Batter

```bash
-d '{
    "match_state": {
      "format": "ODI",
      "innings": 2,
      "over": 38,
      "ball_in_over": 2,
      "phase": "middle",
      "runs_scored": 220,
      "target": 310,
      "partnership": {
        "wickets_fallen": 3,
        "striker_batting_average": 58.0,
        "striker_strike_rate": 92.0
      }
    },
    "fielding_events": [{
      "event_type": "catch_dropped",
      "fielder_position": "cover",
      "ball_trajectory": "hard_drive",
      "ball_metrics": {"speed_kmh": 88.0, "height_off_ground_metres": 0.9},
      "wicket_taken": false,
      "actual_runs_conceded": 1,
      "catch_or_run_out_probability": 0.82
    }]
  }'
```

---

## Event Types

| Event | Description |
|-------|-------------|
| `catch` | Caught dismissal |
| `catch_dropped` | Dropped catch — negative ERS |
| `run_out_direct` | Direct-hit run-out |
| `run_out_relay` | Relay run-out |
| `run_out_miss` | Attempted run-out, batter safe |
| `stumping` | Keeper stumping |
| `boundary_stop` | Prevented 4; boundary_prevented=true |
| `boundary_saved_6` | Prevented 6; six_prevented=true |
| `boundary_missed` | Failed to stop a 4 — negative ERS |
| `misfield_runs` | Misfield conceded extra runs |
| `diving_stop` | Diving stop; no wicket |
| `relay_throw_good` | Accurate relay restricts runs |
| `relay_throw_poor` | Poor relay; extra run taken |
| `ground_fielding` | Standard clean pickup |
| `overthrow` | Errant throw gifted runs |
| `obstructing_field` | Fielder illegally blocked batter (5-run penalty) |

---

## Grading Scale

| Grade | ERS | Meaning |
|-------|-----|---------|
| A+ | ≥ 2.50 | Elite, game-changing |
| A  | 1.50–2.49 | Excellent |
| B+ | 0.80–1.49 | Above average |
| B  | 0.30–0.79 | Solid |
| C  | −0.10–0.29 | Average |
| D  | −0.60–−0.11 | Below average |
| F  | < −0.60 | Significantly hurt the team |

---

## Match Formats & Phases

**T20**: Powerplay = overs 1-6, Middle = 7-15, Death = 16-20  
**ODI**: Powerplay = overs 1-10, Middle = 11-40, Death = 41-50  
**Test**: Powerplay = first session, Middle = settled play, Death = declaration phase

---

## Design Decisions

**Why not just count drops and run-outs?**  
Binary outcomes ignore catch probability. Dropping a 95%-catchable slip catch is catastrophic. Missing a 5%-chance diving catch costs almost nothing in ERS.

**Why DLS-based CRE instead of a flat run-rate?**  
DLS correctly models the non-linear relationship between balls/wickets and run-scoring potential. A team 8 wickets down with 10 balls left scores far fewer than the simple run-rate implies.

**Why Batter Quality Multiplier?**  
Taking a wicket mid-over is worth more if the batter has 80 already on the board. The BQM ensures dismissals are valued by the actual threat removed, not a flat wicket rate.

**Why leverage × raw_ERS instead of win-probability added?**  
LI and WPA are related but LI is more stable and less noisy for individual fielding plays. WPA requires full match simulation; LI is directly interpretable from match state.

---

## References

- Duckworth, Lewis & Stern (2004) — DLS resource model
- Preston & Thomas (2000) — Batting strategy in limited overs cricket
- ESPNcricinfo / Hawk-Eye ball-tracking data 2018-2023
- ICC Playing Conditions 2023