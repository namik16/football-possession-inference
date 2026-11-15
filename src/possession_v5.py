
"""
possession_v5.py

Independent V5 possession inference.

Inputs:
    output/tracking_ball_graded.parquet
    output/tracking_players_graded.parquet
    Smart Tagging XML

Outputs:
    output/possession_v5_frames.parquet
    output/possession_v5_spells.csv
    output/possession_v5_submission.csv

No V4 files are read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks
from dead_ball_v5 import infer_dead_ball

OUT = ROOT / "output"

FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
BLOCKS = half_video_blocks(GAME)

# Independent V5 engineering parameters.
DISTANCE_SCALE_M = 3.5
EVIDENCE_DECAY = 0.95
SWITCH_THRESHOLD = 2.0
INITIAL_THRESHOLD = 0.55
MAX_EVIDENCE = 6.0


def load_data():
    ball = pd.read_parquet(
        OUT / "tracking_ball_graded.parquet"
    )

    players = pd.read_parquet(
        OUT / "tracking_players_graded.parquet"
    )

    ball = ball.sort_values(
        "frame"
    ).reset_index(drop=True)

    players = players.sort_values(
        ["frame", "slot"]
    )

    expected = np.concatenate([
        np.arange(first, last + 1)
        for _, first, last in BLOCKS
    ])

    np.testing.assert_array_equal(
        ball["frame"].to_numpy(dtype=int),
        expected
    )

    assert ball["frame"].is_unique

    assert set(
        players["team"].dropna().unique()
    ) <= {"H", "A"}

    return ball, players


def build_physical_evidence(ball, players):
    p = players.merge(
        ball[[
            "frame",
            "ball_x",
            "ball_y",
            "ball_visible"
        ]],
        on="frame",
        how="left",
        validate="many_to_one"
    )

    dx = (
        p["x"] - p["ball_x"]
    ) * PITCH_LENGTH

    dy = (
        p["y"] - p["ball_y"]
    ) * PITCH_WIDTH

    p["distance_m"] = np.hypot(dx, dy)

    nearest = (
        p.groupby(["frame", "team"])["distance_m"]
        .min()
        .unstack("team")
        .reindex(columns=["H", "A"])
    )

    result = ball[[
        "frame",
        "half",
        "ball_visible"
    ]].copy()

    result = result.join(
        nearest,
        on="frame"
    )

    result = result.rename(columns={
        "H": "home_distance_m",
        "A": "away_distance_m"
    })

    home_dist = result[
        "home_distance_m"
    ].to_numpy(dtype=float)

    away_dist = result[
        "away_distance_m"
    ].to_numpy(dtype=float)

    home_score = np.exp(
        -home_dist / DISTANCE_SCALE_M
    )

    away_score = np.exp(
        -away_dist / DISTANCE_SCALE_M
    )

    home_score = np.nan_to_num(
        home_score, nan=0.0
    )

    away_score = np.nan_to_num(
        away_score, nan=0.0
    )

    visible = result[
        "ball_visible"
    ].fillna(False).to_numpy(dtype=bool)

    home_score[~visible] = 0.0
    away_score[~visible] = 0.0

    result["home_score"] = home_score
    result["away_score"] = away_score

    result["net_evidence"] = (
        home_score - away_score
    )

    return result


def infer_half(group):
    group = group.copy()

    owner = "X"
    evidence_state = 0.0
    previous_dead = False

    labels = []
    strengths = []

    for net, dead in zip(
        group["net_evidence"].to_numpy(dtype=float),
        group["dead_ball"].to_numpy(dtype=bool)
    ):

        if dead:
            # Dead ball interrupts possession ownership.
            owner = "X"
            evidence_state = 0.0

            labels.append("D")
            strengths.append(evidence_state)

            previous_dead = True
            continue

        if previous_dead:
            # Restart requires fresh ownership evidence.
            owner = "X"
            evidence_state = 0.0
            previous_dead = False

        evidence_state = (
            EVIDENCE_DECAY * evidence_state + net
        )

        evidence_state = float(
            np.clip(
                evidence_state,
                -MAX_EVIDENCE,
                MAX_EVIDENCE
            )
        )

        if owner == "X":
            if evidence_state >= INITIAL_THRESHOLD:
                owner = "H"

            elif evidence_state <= -INITIAL_THRESHOLD:
                owner = "A"

        elif owner == "H":
            if evidence_state <= -SWITCH_THRESHOLD:
                owner = "A"

        elif owner == "A":
            if evidence_state >= SWITCH_THRESHOLD:
                owner = "H"

        labels.append(owner)
        strengths.append(evidence_state)

    group["label"] = labels
    group["evidence_state"] = strengths

    return group


def infer_possession(evidence):
    parts = []

    for _, group in evidence.groupby(
        "half", sort=True
    ):
        parts.append(infer_half(group))

    return pd.concat(
        parts, ignore_index=True
    )


def build_spells(predictions):
    parts = []

    for half, group in predictions.groupby(
        "half", sort=True
    ):
        group = group.copy()

        changes = group["label"].ne(
            group["label"].shift()
        )

        group["spell_id"] = changes.cumsum()

        spells = (
            group.groupby("spell_id")
            .agg(
                label=("label", "first"),
                start_frame=("frame", "first"),
                end_frame=("frame", "last"),
                frames=("frame", "size"),
                visible_frames=(
                    "ball_visible", "sum"
                ),
                mean_home_score=(
                    "home_score", "mean"
                ),
                mean_away_score=(
                    "away_score", "mean"
                )
            )
            .reset_index(drop=True)
        )

        spells.insert(0, "half", half)

        spells["duration_s"] = (
            spells["frames"] / FPS
        )

        spells["ball_visibility_pct"] = (
            100 *
            spells["visible_frames"] /
            spells["frames"]
        )

        parts.append(spells)

    return pd.concat(
        parts, ignore_index=True
    )


def print_diagnostics(predictions, spells):
    print("\n--- V5 POSSESSION DIAGNOSTICS ---")

    print("Frames:", len(predictions))

    print("\nFrame labels:")
    print(
        predictions["label"]
        .value_counts()
        .to_string()
    )

    possession = spells[
        spells["label"].isin(["H", "A"])
    ]

    print(
        "\nPossession spells:",
        len(possession)
    )

    if not possession.empty:
        durations = possession["duration_s"]

        print(
            "Mean duration:",
            round(durations.mean(), 2)
        )

        print(
            "Median duration:",
            round(durations.median(), 2)
        )

        print(
            "90th percentile:",
            round(durations.quantile(.9), 2)
        )

        print(
            "Maximum duration:",
            round(durations.max(), 2)
        )

        print(
            "Spells over 60 seconds:",
            int((durations > 60).sum())
        )

        print("\nLongest spells:")

        print(
            possession.nlargest(
                10, "duration_s"
            )[
                [
                    "half",
                    "label",
                    "start_frame",
                    "end_frame",
                    "duration_s",
                    "ball_visibility_pct"
                ]
            ].to_string(index=False)
        )


def main():
    ball, players = load_data()

    print("Ball frames:", len(ball))
    print("Player observations:", len(players))

    # Independent game-state inference.
    dead_ball = infer_dead_ball()

    evidence = build_physical_evidence(
        ball, players
    )

    evidence = evidence.merge(
        dead_ball[[
            "frame",
            "dead_ball",
            "restart_candidate",
            "stationary_candidate"
        ]],
        on="frame",
        how="left",
        validate="one_to_one"
    )

    if evidence["dead_ball"].isna().any():
        raise ValueError(
            "Missing dead-ball predictions."
        )

    predictions = infer_possession(evidence)

    spells = build_spells(predictions)

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    predictions.to_parquet(
        OUT / "possession_v5_frames.parquet",
        index=False
    )

    spells.to_csv(
        OUT / "possession_v5_spells.csv",
        index=False
    )

    # Full-length competition submission.
    total_frames = GAMES[GAME][
        "n_video_frames"
    ]

    labels = np.full(
        total_frames,
        "X",
        dtype="<U1"
    )

    labels[
        predictions["frame"].to_numpy(dtype=int)
    ] = predictions["label"].to_numpy()

    submission = pd.DataFrame({
        "frame": np.arange(total_frames),
        "label": labels
    })

    submission.to_csv(
        OUT / "possession_v5_submission.csv",
        index=False
    )

    print_diagnostics(
        predictions, spells
    )

    print(
        "\nSaved V5 outputs to:",
        OUT
    )


if __name__ == "__main__":
    main()
