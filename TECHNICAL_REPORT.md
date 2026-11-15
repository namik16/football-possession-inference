# Technical Report: Football Possession Inference

**More Than a Hack 2027 — Barcelona | Metrica Sports**  
**Team:** Boston Press  
**Repository:** https://github.com/namik16/football-possession-inference  
**Match:** FC Barcelona (home) vs Athletic Club (away)  
**Submission file:** `Boston Press.csv`

## Abstract

This submission infers frame-level team possession from tactical tracking data using a deterministic, rule-based model. It combines player-to-ball proximity, temporal evidence accumulation, hysteresis-based possession switching, and restart annotations from Metrica Sports Smart Tagging. The method does not train a supervised classifier on the competition's possession ground truth. Its final output assigns each of 199,319 tactical-video frames one of four labels: home (`H`), away (`A`), dead ball (`D`), or unknown (`X`).

The core hypothesis is that possession should be modeled as a **persistent state supported by successive observations**, rather than assigned independently to the nearest player in every frame.

## 1. Data and Scope

The supplied tactical recording contains **199,319 frames at 25 FPS**. The implementation processes two configured match periods:

| Period | Inclusive frame range | Number of frames |
|---|---|---:|
| First half | 11,185–90,651 | 79,467 |
| Second half | 110,808–190,197 | 79,390 |
| **Total** | | **158,857** |

The remaining **40,462 frames** are assigned `X` when the full-length submission is constructed. This is an output convention for frames outside the configured periods, not a semantic claim about ball control.

### Source assets

- `BAR-ATH_tactical_atd.txt`: tactical ball and player tracking observations.
- `BAR-ATH_tactical_atd.xml`: tracking metadata used to associate player slots with teams.
- `BAR-ATH_tactical_smart_tagging.xml`: automatically annotated tactical intervals, including set pieces.
- `games.py`: pitch dimensions, frame count, frame rate, and alignment constants.

The pitch is configured as **105.3 × 68.0 metres**. Neither tactical video decoding nor broadcast footage is required to run the inference code. Licensed match assets and derived Parquet datasets are not included in the public repository.

## 2. Processing Pipeline

The final solution uses four source modules:

1. **`src/extract_tracking.py`** parses tactical ATD tracking and metadata, extracts ball observations and visible player observations, maps player slots to home (`H`) or away (`A`), and writes `tracking_ball.parquet` and `tracking_players.parquet`.
2. **`src/filter_tracking.py`** retains the two configured match periods, assigns half identifiers, verifies **158,857** ball frames, and writes graded tracking Parquet files.
3. **`src/dead_ball.py`** infers candidate dead-ball frames from set-piece annotations in Smart Tagging XML.
4. **`src/possession.py`** computes spatial evidence, applies the temporal possession state machine independently to each half, integrates dead-ball states, summarizes possession spells, and exports the submission.

`src/possession.py` automatically calls the dead-ball inference module; it is not necessary to execute `src/dead_ball.py` separately.

## 3. Spatial Possession Evidence

For each frame, the algorithm finds the minimum ball-to-player distance for each team. Normalized pitch coordinates are converted to metres using the pitch dimensions:

\[
d_{p,t}=\sqrt{((x_{p,t}-x_{b,t})L)^2+((y_{p,t}-y_{b,t})W)^2}
\]

where \(L=105.3\) m and \(W=68.0\) m.

Nearest-player distances are transformed into team evidence scores:

\[
S_{H,t}=e^{-d_{H,t}/3.5},\qquad S_{A,t}=e^{-d_{A,t}/3.5}
\]

The net observation is:

\[
e_t=S_{H,t}-S_{A,t}
\]

Positive values favor the home team; negative values favor the away team. Missing nearest-player distances yield zero evidence for the affected side. **If the ball is not visible, both team scores are explicitly set to zero.**

This is a proxy for likely control, not proof that the nearest player actually possesses the ball.

## 4. Temporal State and Switching

Rather than classify each frame independently, the model accumulates proximity evidence:

\[
z_t=\operatorname{clip}(0.95z_{t-1}+e_t,-6,6)
\]

The following constants are defined in the final source:

| Parameter | Value | Function |
|---|---:|---|
| Distance scale | 3.5 m | Converts distance into proximity evidence |
| Evidence decay | 0.95 | Discounts older evidence |
| Initial ownership threshold | 0.55 | Establishes ownership from `X` |
| Ownership switch threshold | 2.0 | Requires stronger opposing evidence to switch teams |
| Evidence limit | ±6.0 | Caps accumulated evidence |

Starting from `X`, accumulated evidence of at least **+0.55** establishes `H`, and evidence of at most **−0.55** establishes `A`. Once ownership is `H`, the model requires evidence of **−2.0** or lower to switch to `A`. Conversely, an `A`-to-`H` switch requires **+2.0** or higher.

This hysteresis reduces oscillation during close challenges and noisy tracking. It also introduces potential **turnover latency**: a genuine change of possession may be recognized after it occurs. Ownership and evidence are reset at the start of each half.

## 5. Dead-Ball and Restart Inference

The dead-ball module reads Smart Tagging intervals and selects annotations whose code is `SET PIECES`. For sufficiently long intervals, it marks a short window before the approximate restart boundary as a dead-ball candidate.

| Parameter | Value |
|---|---:|
| Pre-restart lookback | 1.5 seconds |
| End-of-tag guard | 0.2 seconds |
| Minimum tag duration | 0.5 seconds |
| Stationary-ball detector | Disabled |

Marked frames receive label `D`. Each dead-ball frame resets the current owner to `X` and clears accumulated evidence. Following a restart, ownership must be re-established from new spatial observations.

The code includes an optional stationary-ball heuristic, but **`ENABLE_STATIONARY_BALL = False`** in the final configuration. Smart Tagging is treated as automated, imperfect event evidence—not possession ground truth. This restart-window heuristic does not claim to detect all fouls, stoppages, balls out of play, or other dead-ball situations.

## 6. Loose Balls, Challenges, and Missing Tracking

The method does not explicitly classify loose balls or contested challenges using a separate model.

- **Challenges:** when players from both sides are near the ball, their evidence can partly cancel. The previous owner may persist until the opposing threshold is reached.
- **Loose balls:** the previous owner can remain assigned even when neither team clearly controls the ball. This is a limitation of temporal persistence.
- **Missing ball positions:** the frame adds zero new proximity evidence, while the accumulated state decays and the previous owner may persist.
- **Unknown state:** `X` is used before initial ownership is established and immediately after a dead-ball interruption until sufficient evidence returns.
- **Restart candidates:** annotated dead-ball frames receive `D` and reset ownership.

These rules improve continuity but should not be mistaken for semantic understanding of every duel, aerial ball, deflection, or uncontrolled phase.

## 7. Local Evaluation

The following measurements were recorded during development on the available public evaluation segment. They are **historical project results**, not independently rerun for this report and not hidden-test scores.

| Configuration | Recorded spell F1 |
|---|---:|
| Organizer baseline | 0.4242 |
| **Selected final heuristic (formerly V5)** | **0.3733** |
| Earlier variant (V4) | 0.3608 |
| Later variant (V6) | 0.3556 |

Additional recorded metrics for the selected configuration:

| Metric | Recorded value |
|---|---:|
| Macro F1 | 0.4929 |
| Transition F1 | 0.2041 |
| Dead-ball F1 | 0.0312 |

The selected heuristic did **not** outperform the organizer baseline on the recorded spell F1 metric. Its low dead-ball F1 highlights the limitations of relying primarily on annotated restart windows. The final configuration was retained after comparison with the project's other heuristic variants because it was the selected stable implementation, not because it demonstrated superior performance to the baseline.

No private competition ground-truth labels were used to train the model.

## 8. Limitations

1. **Proximity is not control.** The nearest player may be challenging, marking, or chasing rather than possessing the ball.
2. **Switching delay.** Hysteresis can recognize genuine turnovers late.
3. **Stale ownership during occlusion.** Missing ball observations do not automatically force the state to `X`.
4. **Incomplete dead-ball detection.** Set-piece annotations cover only a subset of stoppages and may be temporally imprecise.
5. **No dedicated loose-ball detector.** Contested or uncontrolled phases may be attributed to the prior owner.
6. **Heuristic thresholds.** The parameters are fixed engineering choices rather than statistically estimated optimal values.
7. **Match-specific configuration.** Half boundaries, pitch dimensions, and total frame count are configured for this challenge match.

## 9. Reproduction

Obtain the licensed tactical assets separately and place them under `data/tactical/`, or point `MTAH_DATA_DIR` at a directory containing the `tactical` folder:

```text
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

From the repository root, execute:

```bash
python -m pip install -r requirements.txt
python src/extract_tracking.py
python src/filter_tracking.py
python src/possession.py
```

The resulting files include:

```text
output/dead_ball_frames.parquet
output/dead_ball_intervals.csv
output/possession_frames.parquet
output/possession_spells.csv
output/Boston Press.csv
```

The submission artifact is **`output/Boston Press.csv`**, with columns `frame,label`, one row for every frame **0–199318**, and labels restricted to `H`, `A`, `D`, and `X`.

The repository intentionally excludes licensed data, videos, and generated tracking datasets. Reproduction requires access to the original competition assets. Exact dependency versions should be pinned in `requirements.txt` to the tested environment.

## 10. Conclusion

This solution provides a transparent, reproducible frame-level possession estimator using a simple physical signal—nearest-player proximity—combined with temporal state persistence. Its strength is deterministic behavior and a clear interpretation of every state transition. Its main weaknesses are semantic ambiguity in challenges and loose-ball situations, delayed turnovers, and incomplete dead-ball inference. These limitations are acknowledged rather than concealed by the reported local metrics.

## References

- [Metrica Sports — More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
- [Boston Press — football-possession-inference](https://github.com/namik16/football-possession-inference)

---

*This report documents the published final heuristic implementation. The evaluation figures are previously recorded development results and have not been independently reproduced for this document.*
