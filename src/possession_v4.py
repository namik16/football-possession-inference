
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks

OUT = ROOT / "output"
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]

# Initial engineering parameters, not optimized against GT.
DISTANCE_SCALE_M = 3.0
EVIDENCE_DECAY = 0.92
SWITCH_THRESHOLD = 1.5
INITIAL_THRESHOLD = 0.4
MAX_EVIDENCE = 5.0

# Graded frames remain in original video coordinates.
BLOCKS = half_video_blocks(GAME)


def load_data():
    ball = pd.read_parquet(
        OUT / "tracking_ball_graded.parquet"
    )
    players = pd.read_parquet(
        OUT / "tracking_players_graded.parquet"
    )

    ball = ball.sort_values("frame").reset_index(drop=True)
    players = players.sort_values(["frame", "slot"])

    assert ball["frame"].is_unique
    assert set(players["team"].dropna().unique()) <= {"H", "A"}

    expected = np.concatenate([
        np.arange(first, last + 1)
        for _, first, last in BLOCKS
    ])
    np.testing.assert_array_equal(
        ball["frame"].to_numpy(), expected
    )

    return ball, players


def build_evidence(ball, players):
    # Join ball coordinates onto each observed player.
    p = players.merge(
        ball[["frame", "ball_x", "ball_y", "ball_visible"]],
        on="frame",
        how="left",
        validate="many_to_one",
    )

    dx = (p["x"] - p["ball_x"]) * PITCH_LENGTH
    dy = (p["y"] - p["ball_y"]) * PITCH_WIDTH
    p["distance_m"] = np.hypot(dx, dy)

    # Minimum distance for each team, for each frame.
    nearest = (
        p.groupby(["frame", "team"])["distance_m"]
        .min()
        .unstack("team")
        .reindex(columns=["H", "A"])
    )

    result = ball[[
        "frame", "half", "ball_visible"
    ]].copy()

    result = result.join(nearest, on="frame")
    result = result.rename(columns={
        "H": "home_distance_m",
        "A": "away_distance_m",
    })

    h = result["home_distance_m"].to_numpy(dtype=float)
    a = result["away_distance_m"].to_numpy(dtype=float)

    # Larger score means stronger evidence of control.
    # Exponential distance scoring reduces the influence
    # of distant players.
    h_score = np.exp(-h / DISTANCE_SCALE_M)
    a_score = np.exp(-a / DISTANCE_SCALE_M)

    # Missing team observations contribute no evidence.
    h_score = np.nan_to_num(h_score, nan=0.0)
    a_score = np.nan_to_num(a_score, nan=0.0)

    visible = result["ball_visible"].fillna(False).to_numpy()

    h_score[~visible] = 0.0
    a_score[~visible] = 0.0

    result["home_score"] = h_score
    result["away_score"] = a_score
    result["net_evidence"] = h_score - a_score

    return result


def infer_half(df):
    labels = []
    strengths = []

    owner = "X"
    evidence = 0.0

    for net in df["net_evidence"].to_numpy(dtype=float):
        # Decay old evidence, accumulate new evidence.
        evidence = (
            EVIDENCE_DECAY * evidence + net
        )
        evidence = float(np.clip(
            evidence, -MAX_EVIDENCE, MAX_EVIDENCE
        ))

        if owner == "X":
            if evidence >= INITIAL_THRESHOLD:
                owner = "H"
            elif evidence <= -INITIAL_THRESHOLD:
                owner = "A"

        elif owner == "H":
            if evidence <= -SWITCH_THRESHOLD:
                owner = "A"

        elif owner == "A":
            if evidence >= SWITCH_THRESHOLD:
                owner = "H"

        labels.append(owner)
        strengths.append(evidence)

    result = df.copy()
    result["label"] = labels
    result["evidence_state"] = strengths

    return result


def infer(evidence):
    # Never propagate ownership across halftime.
    parts = [
        infer_half(group)
        for _, group in evidence.groupby("half", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def build_spells(predictions):
    parts = []

    for half, group in predictions.groupby("half", sort=True):
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
                mean_home_score=("home_score", "mean"),
                mean_away_score=("away_score", "mean"),
                visible_frames=("ball_visible", "sum"),
            )
            .reset_index(drop=True)
        )

        spells.insert(0, "half", half)
        spells["duration_s"] = spells["frames"] / FPS
        spells["ball_visibility_pct"] = (
            100 * spells["visible_frames"] / spells["frames"]
        )

        parts.append(spells)

    return pd.concat(parts, ignore_index=True)


def print_diagnostics(predictions, spells):
    possession = spells[spells["label"].isin(["H", "A"])]

    print("\n--- V4 DIAGNOSTICS ---")
    print("Frames:", len(predictions))
    print("Possession spells:", len(possession))

    if possession.empty:
        print("No possession spells detected.")
        return

    durations = possession["duration_s"]

    print(f"Mean duration:   {durations.mean():.2f}s")
    print(f"Median duration: {durations.median():.2f}s")
    print(f"90th percentile: {durations.quantile(.90):.2f}s")
    print(f"Maximum:         {durations.max():.2f}s")
    print(f"Spells over 60s: {(durations > 60).sum()}")

    print("\nFrame labels:")
    print(predictions["label"].value_counts())

    print("\nLongest spells:")
    print(
        possession.nlargest(10, "duration_s")[
            ["half", "label", "start_frame",
             "end_frame", "duration_s",
             "ball_visibility_pct"]
        ].to_string(index=False)
    )


def main():
    ball, players = load_data()

    print("Ball frames:", len(ball))
    print("Player observations:", len(players))

    evidence = build_evidence(ball, players)
    predictions = infer(evidence)
    spells = build_spells(predictions)

    OUT.mkdir(parents=True, exist_ok=True)

    predictions.to_parquet(
        OUT / "possession_v4_frames.parquet",
        index=False,
    )
    spells.to_csv(
        OUT / "possession_v4_spells.csv",
        index=False,
    )

    # Full-length submission, with ungraded frames X.
    total_frames = GAMES[GAME]["n_video_frames"]
    labels = np.full(total_frames, "X", dtype="<U1")
    labels[predictions["frame"].to_numpy(dtype=int)] = (
        predictions["label"].to_numpy()
    )

    submission = pd.DataFrame({
        "frame": np.arange(total_frames),
        "label": labels,
    })
    submission.to_csv(
        OUT / "possession_v4_submission.csv",
        index=False,
    )

    print_diagnostics(predictions, spells)
    print("\nSaved V4 outputs to:", OUT)


if __name__ == "__main__":
    main()
