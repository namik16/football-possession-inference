# Football Possession Inference

**More Than a Hack 2027 — Barcelona | Metrica Sports**

A reproducible, rule-based approach to inferring team possession from football tracking data and automated event annotations.

This project produces frame-level possession classifications for the **FC Barcelona vs Athletic Club** match provided as part of the More Than a Hack 2027 challenge.

The solution combines spatial proximity, temporal evidence accumulation, and restart detection to estimate possession without training a supervised machine learning model.

## 1. Project Overview

The objective is to assign one possession state to every frame of the tactical match recording.

| Label | Description |
|---|---|
| `H` | Home team possession |
| `A` | Away team possession |
| `D` | Dead ball / play stopped |
| `X` | Unknown or indeterminate possession |

The tactical recording contains **199,319 frames at 25 FPS**.

The approach prioritizes temporal consistency over assigning possession solely to whichever player is closest to the ball in an individual frame.

## 2. Methodology

The inference pipeline consists of four stages.

### 2.1 Tracking Data Extraction

The extraction process reads the provided tactical tracking data in ATD format and its accompanying metadata.

The script:

- Extracts ball coordinates, speed, and visibility.
- Extracts player positions and speeds.
- Associates player tracking slots with home and away teams.
- Preserves the original video-frame indexing.
- Converts the extracted observations into Parquet files for downstream processing.

The resulting datasets contain separate ball and player observations.

### 2.2 Tracking Data Filtering

The extracted tracking data is restricted to the two match periods used for possession inference.

| Period | First frame | Last frame |
|---|---:|---:|
| First half | 11,185 | 90,651 |
| Second half | 110,808 | 190,197 |

These periods contain **158,857 frames**.

The two halves are processed separately to prevent temporal possession evidence from carrying across halftime.

### 2.3 Spatial Possession Evidence

For each frame, the algorithm identifies the nearest tracked player from each team to the ball.

Player-to-ball distances are converted from normalized pitch coordinates into metres using the supplied pitch dimensions.

Each team's proximity score is calculated as:

\[
S_t = e^{-d_t/\lambda}
\]

Where:

- \(d_t\) is the nearest player-to-ball distance for a team.
- \(\lambda = 3.5\) metres is the distance-decay parameter.
- \(S_t\) is the resulting proximity score.

The difference between home and away proximity scores produces a net possession-evidence value:

\[
E_t = S_{H,t} - S_{A,t}
\]

When the ball is not visible, both teams receive zero proximity evidence for that frame.

### 2.4 Temporal Possession Inference

Possession is inferred using a stateful evidence-accumulation process rather than independent frame classifications.

The evidence state evolves according to:

\[
Z_t = \operatorname{clip}(0.95Z_{t-1}+E_t,-6,6)
\]

A team is assigned initial possession when the accumulated evidence crosses a threshold of **0.55** in its favor.

Once possession has been established, a stronger opposing threshold of **2.0** is required before ownership switches teams.

This hysteresis mechanism is intended to reduce rapid, potentially unrealistic possession changes caused by momentary player proximity, challenges, or tracking noise.

Possession remains with the previously inferred team until sufficient opposing evidence accumulates or a dead-ball event interrupts the sequence.

### 2.5 Dead-Ball and Restart Detection

Dead-ball inference uses the supplied Metrica Sports Smart Tagging XML.

The algorithm identifies annotations classified as `SET PIECES` and estimates a short dead-ball interval preceding the annotated restart boundary.

The implemented parameters are:

| Parameter | Value |
|---|---:|
| Pre-restart interval | 1.5 seconds |
| Restart boundary guard | 0.2 seconds |
| Minimum annotation duration | 0.5 seconds |

Frames identified as dead-ball are assigned `D`.

Dead-ball intervals also reset the accumulated possession evidence, requiring fresh evidence to establish ownership after a restart.

A stationary-ball detection mechanism is implemented but disabled in the final configuration.

Smart Tagging is treated as an imperfect source of event evidence rather than possession ground truth.

## 3. Repository Structure

The main components are:

```text
football-possession-inference/
├── games.py
├── requirements.txt
├── README.md
├── src/
│   ├── extract_tracking.py
│   ├── filter_tracking.py
│   ├── dead_ball.py
│   └── possession.py
└── starter/
    └── load_smart_tagging.py
```

The repository also includes supplementary starter utilities and tracking diagnostics.

## 4. Requirements

The project uses Python and three third-party dependencies:

- NumPy
- pandas
- PyArrow

Other imported modules are part of the Python standard library or included in this repository.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The final dependency versions should be pinned in `requirements.txt` to match the tested environment.

## 5. Input Data

The model requires the licensed tactical tracking data and Smart Tagging annotations provided by the competition organizers.

These assets are **not distributed with this repository**.

The required files should be placed in the following structure:

```text
data/
└── tactical/
    ├── BAR-ATH_tactical_atd.txt
    ├── BAR-ATH_tactical_atd.xml
    └── BAR-ATH_tactical_smart_tagging.xml
```

Alternatively, define the `MTAH_DATA_DIR` environment variable to point to the directory containing the `tactical` folder.

For Windows PowerShell:

```powershell
$env:MTAH_DATA_DIR = "D:\path\to\match-data"
```

For macOS or Linux:

```bash
export MTAH_DATA_DIR="/path/to/match-data"
```

The tactical video itself is not required to execute the final inference pipeline.

## 6. Reproducing the Submission

Run the following commands from the repository root, in order.

**Step 1 — Extract tracking data**

```bash
python src/extract_tracking.py
```

Generates:

- `output/tracking_ball.parquet`
- `output/tracking_players.parquet`

**Step 2 — Filter tracking data**

```bash
python src/filter_tracking.py
```

Generates:

- `output/tracking_ball_graded.parquet`
- `output/tracking_players_graded.parquet`

**Step 3 — Run possession inference**

```bash
python src/possession.py
```

This final command automatically invokes the dead-ball inference module and generates:

- `output/dead_ball_frames.parquet`
- `output/dead_ball_intervals.csv`
- `output/possession_frames.parquet`
- `output/possession_spells.csv`
- `output/Boston Press.csv`

The final submission is:

```text
output/Boston Press.csv
```

## 7. Submission Format

The final CSV contains exactly **199,319 rows**, with one prediction for every tactical video frame.

The required columns are:

```csv
frame,label
```

Frame indices begin at `0` and end at `199318`.

Each label must belong to the set `{H, A, D, X}`.

Frames outside the two processed match periods are assigned `X`.

The output should be validated using the official competition submission validator before submission.

## 8. Design Decisions

The solution makes several deliberate engineering choices.

**No supervised model training**

Possession is inferred through explicit spatial and temporal rules rather than a fitted classification model.

**Temporal consistency**

Accumulated evidence and asymmetric ownership thresholds reduce sensitivity to short-lived changes in player proximity.

**Independent processing by half**

Temporal evidence resets between halves, preventing possession states from crossing match-period boundaries.

**Conservative handling of missing ball observations**

Frames without a visible ball contribute no new spatial evidence, although the existing possession state can persist.

**Independent restart detection**

Dead-ball inference is performed separately and integrated into the possession state machine.

## 9. Limitations

The approach has several limitations:

- **Proximity is not equivalent to control.** The nearest player may not possess the ball, particularly during aerial play, tackles, and contested situations.
- **Missing ball observations create uncertainty.** Temporal persistence can maintain an incorrect possession state when direct spatial evidence is unavailable.
- **Loose balls may be attributed to a team.** The model does not explicitly identify every contested or uncontrolled ball.
- **Possession changes can be delayed.** Hysteresis reduces false switches but may postpone recognition of genuine turnovers.
- **Dead-ball detection is incomplete.** Restart annotations may not identify every stoppage, and their timing is approximate.
- **Stationary-ball inference is disabled.** Ball-speed reliability was not established sufficiently for its inclusion in the final configuration.
- **Parameters are heuristic.** The engineering thresholds are not presented as statistically optimal.

The model is intended as a reproducible possession-inference baseline rather than a complete semantic understanding of ball control.

## 10. Data Usage and Reproducibility

The repository contains the implementation required to reproduce the submission using the original competition assets.

Licensed tracking files, match videos, Smart Tagging annotations, and derived tracking datasets are excluded from version control.

The model does not require access to private competition ground-truth labels for execution.

## 11. Acknowledgments

This project was developed for **More Than a Hack 2027**, using the match tracking data and Smart Tagging assets supplied by **Metrica Sports**.

The repository retains selected organizer-provided data-loading utilities and match configuration code.

Competition resources and original starter materials are available from the [More Than a Hack 2027 repository](https://github.com/metrica-sports/more-than-a-hack-2027).