
"""
dead_ball.py

Independent dead-ball inference.

Inputs:
    output/tracking_ball_graded.parquet
    Smart Tagging XML via starter.load_smart_tagging

Outputs:
    output/dead_ball__frames.parquet
    output/dead_ball_intervals.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks
from starter import load_smart_tagging as st

OUT = ROOT / "output"
FPS = GAMES[GAME]["fps"]

BLOCKS = half_video_blocks(GAME)

# Engineering assumptions, not GT-optimized.
PRE_RESTART_SECONDS = 1.5
END_GUARD_SECONDS = 0.2
MIN_TAG_SECONDS = 0.5

# Disabled until ball speed reliability is established.
ENABLE_STATIONARY_BALL = False
STATIONARY_SPEED_THRESHOLD = 0.5
STATIONARY_MIN_SECONDS = 2.0


def load_ball():
    ball = pd.read_parquet(
        OUT / "tracking_ball_graded.parquet"
    )

    ball = ball.sort_values("frame").reset_index(drop=True)

    expected = np.concatenate([
        np.arange(first, last + 1)
        for _, first, last in BLOCKS
    ])

    np.testing.assert_array_equal(
        ball["frame"].to_numpy(dtype=int),
        expected
    )

    return ball


def build_restart_candidates(ball):
    events = st.load()

    result = ball[["frame", "half"]].copy()
    result["restart_candidate"] = False

    intervals = []

    pre_frames = int(round(PRE_RESTART_SECONDS * FPS))
    guard_frames = int(round(END_GUARD_SECONDS * FPS))
    minimum_frames = int(round(MIN_TAG_SECONDS * FPS))

    frames = result["frame"].to_numpy()

    for event in events:
        code = (event.get("code") or "").strip().upper()

        if code != "SET PIECES":
            continue

        start = int(event["start_frame"])
        end = int(event["end_frame"])

        if end - start < minimum_frames:
            continue

        # Approximate restart at the end of the tag.
        # Exclude a small guard interval around that boundary.
        candidate_end = end - guard_frames
        candidate_start = max(
            start,
            candidate_end - pre_frames
        )

        if candidate_end <= candidate_start:
            continue

        mask = (
            (frames >= candidate_start) &
            (frames < candidate_end)
        )

        if not mask.any():
            continue

        result.loc[mask, "restart_candidate"] = True

        intervals.append({
            "start_frame": candidate_start,
            "end_frame_exclusive": candidate_end,
            "duration_s": (
                candidate_end - candidate_start
            ) / FPS,
            "restart_type": (
                event.get("tags", {}).get("Type") or ""
            ),
            "source": "smart_tagging"
        })

    return result, pd.DataFrame(intervals)


def add_stationary_ball_evidence(result, ball):
    result = result.copy()
    result["stationary_candidate"] = False

    if not ENABLE_STATIONARY_BALL:
        return result

    speed = pd.to_numeric(
        ball["ball_speed"],
        errors="coerce"
    ).to_numpy(dtype=float)

    visible = (
        ball["ball_visible"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    stationary = (
        visible &
        np.isfinite(speed) &
        (speed <= STATIONARY_SPEED_THRESHOLD)
    )

    minimum_frames = int(round(
        STATIONARY_MIN_SECONDS * FPS
    ))

    # Process each half independently.
    for _, group in ball.groupby("half", sort=True):
        indices = group.index.to_numpy()
        flags = stationary[indices]

        boundaries = np.r_[
            0,
            np.flatnonzero(flags[1:] != flags[:-1]) + 1,
            len(flags)
        ]

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:]
        ):
            if (
                flags[start] and
                end - start >= minimum_frames
            ):
                result.loc[
                    indices[start:end],
                    "stationary_candidate"
                ] = True

    return result


def infer_dead_ball():
    ball = load_ball()

    result, intervals = build_restart_candidates(ball)

    result = add_stationary_ball_evidence(
        result, ball
    )

    result["dead_ball"] = (
        result["restart_candidate"] |
        result["stationary_candidate"]
    )

    result["game_state"] = np.where(
        result["dead_ball"], "DEAD", "LIVE"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        OUT / "dead_ball_frames.parquet",
        index=False
    )

    intervals.to_csv(
        OUT / "dead_ball_intervals.csv",
        index=False
    )

    print("\n--- DEAD-BALL DIAGNOSTICS ---")
    print("Graded frames:", len(result))
    print("Restart candidate intervals:", len(intervals))
    print("Dead-ball frames:", int(result["dead_ball"].sum()))
    print(
        "Dead-ball percentage:",
        round(100 * result["dead_ball"].mean(), 2)
    )

    return result


if __name__ == "__main__":
    infer_dead_ball()
