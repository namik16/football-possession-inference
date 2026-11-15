# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 21:28:23 2026

@author: Namik
"""


"""
render_tracking_2d.py

Render every tactical-video frame as a 2D football pitch.

Input:
    output/tracking_ball.parquet
    output/tracking_players.parquet

Output:
    output/tracking_2d/frames/frame_000000.png ...
    output/tracking_2d/render_manifest.csv

Independent of all possession models.
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks

OUT = ROOT / "output"
FRAME_DIR = OUT / "tracking_2d" / "frames"
MANIFEST = OUT / "tracking_2d" / "render_manifest.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
BLOCKS = half_video_blocks(GAME)

# Rendering configuration
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 800
MARGIN_X = 75
MARGIN_Y = 65

# Render all frames by default.
START_FRAME = 0
END_FRAME = N_FRAMES

# Change to a small value (e.g. 250) for a quick test.
# None renders every frame in the selected range.
MAX_FRAMES = None

# JPEG-style compression is not used: these are PNGs.
PNG_COMPRESSION = 3

# Set to True to replace existing frames.
OVERWRITE = False

# Tracking orientation controls.
# These affect only the drawing, not source data.
FLIP_X = False
FLIP_Y = False

# OpenCV colors are BGR, not RGB.
BARCELONA_COLOR = (210, 87, 21)  # Blue
ATHLETIC_COLOR = (60, 75, 233)   # Red
BALL_COLOR = (255, 255, 255)
WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
PITCH_COLOR = (65, 125, 40)

PLAYER_RADIUS = 9
BALL_RADIUS = 6

SHOW_SLOT_NUMBERS = True


def pitch_point(x, y):
    """Convert normalized tracking coordinates to pixels."""
    if FLIP_X:
        x = 1.0 - x
    if FLIP_Y:
        y = 1.0 - y

    px = MARGIN_X + x * (
        IMAGE_WIDTH - 2 * MARGIN_X
    )
    py = MARGIN_Y + y * (
        IMAGE_HEIGHT - 2 * MARGIN_Y
    )

    return int(round(px)), int(round(py))


def draw_pitch():
    """Build the static pitch background once."""
    img = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        PITCH_COLOR,
        dtype=np.uint8
    )

    left = MARGIN_X
    right = IMAGE_WIDTH - MARGIN_X
    top = MARGIN_Y
    bottom = IMAGE_HEIGHT - MARGIN_Y

    cx = (left + right) // 2
    cy = (top + bottom) // 2

    scale_x = (right - left) / PITCH_LENGTH
    scale_y = (bottom - top) / PITCH_WIDTH

    # Outer boundaries.
    cv2.rectangle(
        img, (left, top), (right, bottom),
        WHITE, 2
    )

    # Halfway line.
    cv2.line(
        img, (cx, top), (cx, bottom),
        WHITE, 2
    )

    # Centre circle and centre spot.
    radius = int(round(9.15 * scale_x))

    cv2.circle(
        img, (cx, cy), radius,
        WHITE, 2
    )

    cv2.circle(
        img, (cx, cy), 3,
        WHITE, -1
    )

    # Penalty areas: 16.5 m deep, 40.32 m wide.
    penalty_half_height = int(
        round(40.32 * scale_y / 2)
    )

    penalty_depth = int(
        round(16.5 * scale_x)
    )

    for x0, x1 in [
        (left, left + penalty_depth),
        (right - penalty_depth, right)
    ]:
        cv2.rectangle(
            img,
            (x0, cy - penalty_half_height),
            (x1, cy + penalty_half_height),
            WHITE, 2
        )

    # Goal areas: 5.5 m deep, 18.32 m wide.
    goal_half_height = int(
        round(18.32 * scale_y / 2)
    )

    goal_depth = int(
        round(5.5 * scale_x)
    )

    for x0, x1 in [
        (left, left + goal_depth),
        (right - goal_depth, right)
    ]:
        cv2.rectangle(
            img,
            (x0, cy - goal_half_height),
            (x1, cy + goal_half_height),
            WHITE, 2
        )

    # Penalty spots.
    spot_distance = int(
        round(11.0 * scale_x)
    )

    for px in [
        left + spot_distance,
        right - spot_distance
    ]:
        cv2.circle(
            img, (px, cy), 3,
            WHITE, -1
        )

    # Penalty arcs (outside penalty areas).
    arc_radius = int(round(9.15 * scale_x))
    arc_radius_y = int(round(9.15 * scale_y))

    for px, start_angle, end_angle in [
        (left + spot_distance, -53, 53),
        (right - spot_distance, 127, 233)
    ]:
        cv2.ellipse(
            img,
            (px, cy),
            (arc_radius, arc_radius_y),
            0,
            start_angle,
            end_angle,
            WHITE,
            2
        )

    # Goals outside pitch boundary.
    goal_half = int(round(7.32 * scale_y / 2))
    goal_depth_px = 12

    cv2.rectangle(
        img,
        (left - goal_depth_px, cy - goal_half),
        (left, cy + goal_half),
        WHITE, 2
    )

    cv2.rectangle(
        img,
        (right, cy - goal_half),
        (right + goal_depth_px, cy + goal_half),
        WHITE, 2
    )

    return img


def load_tracking():
    ball = pd.read_parquet(
        OUT / "tracking_ball.parquet"
    )

    players = pd.read_parquet(
        OUT / "tracking_players.parquet"
    )

    ball = ball.sort_values("frame").reset_index(drop=True)
    players = players.sort_values(
        ["frame", "slot"]
    ).reset_index(drop=True)

    expected = np.arange(N_FRAMES)

    np.testing.assert_array_equal(
        ball["frame"].to_numpy(dtype=int),
        expected
    )

    if not players["frame"].between(
        0, N_FRAMES - 1
    ).all():
        raise ValueError("Player frame out of range.")

    if not set(
        players["team"].dropna().unique()
    ) <= {"H", "A"}:
        raise ValueError("Unexpected team code.")

    print("Ball frames:", len(ball))
    print("Player observations:", len(players))

    return ball, players


def draw_player(img, x, y, team, slot):
    if not (
        np.isfinite(x) and
        np.isfinite(y)
    ):
        return

    px, py = pitch_point(x, y)

    color = (
        BARCELONA_COLOR
        if team == "H"
        else ATHLETIC_COLOR
    )

    cv2.circle(
        img, (px, py),
        PLAYER_RADIUS + 2,
        BLACK, -1
    )

    cv2.circle(
        img, (px, py),
        PLAYER_RADIUS,
        color, -1
    )

    if SHOW_SLOT_NUMBERS:
        text = str(slot)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.28
        thickness = 1

        (tw, th), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        cv2.putText(
            img,
            text,
            (px - tw // 2, py + th // 2),
            font,
            font_scale,
            WHITE,
            thickness,
            cv2.LINE_AA
        )


def draw_ball(img, x, y):
    if not (
        np.isfinite(x) and
        np.isfinite(y)
    ):
        return False

    px, py = pitch_point(x, y)

    cv2.circle(
        img,
        (px, py),
        BALL_RADIUS + 3,
        BLACK,
        -1
    )

    cv2.circle(
        img,
        (px, py),
        BALL_RADIUS,
        BALL_COLOR,
        -1
    )

    return True


def get_half(frame):
    for half, start, end in BLOCKS:
        if start <= frame <= end:
            return half
    return None


def add_header(img, frame, half, visible, n_players):
    time_s = frame / FPS
    minutes = int(time_s // 60)
    seconds = int(time_s % 60)

    half_text = (
        f"Half {half}"
        if half is not None
        else "Outside tracked halves"
    )

    text = (
        f"Frame {frame:06d}  |  "
        f"Video {minutes:02d}:{seconds:02d}  |  "
        f"{half_text}"
    )

    cv2.putText(
        img,
        text,
        (MARGIN_X, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
        cv2.LINE_AA
    )

    status = (
        "BALL VISIBLE"
        if visible
        else "BALL NOT TRACKED"
    )

    footer = (
        f"Barcelona H (Blue)  |  "
        f"Athletic A (Red)  |  "
        f"{status}  |  "
        f"Players: {n_players}"
    )

    cv2.putText(
        img,
        footer,
        (MARGIN_X, IMAGE_HEIGHT - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        WHITE,
        1,
        cv2.LINE_AA
    )


def main():
    ball, players = load_tracking()

    # Convert to NumPy for fast repeated access.
    ball_xy = ball[[
        "ball_x", "ball_y"
    ]].to_numpy(dtype=float)

    player_frames = players[
        "frame"
    ].to_numpy(dtype=np.int32)

    player_x = players[
        "x"
    ].to_numpy(dtype=float)

    player_y = players[
        "y"
    ].to_numpy(dtype=float)

    player_teams = players[
        "team"
    ].to_numpy()

    player_slots = players[
        "slot"
    ].to_numpy(dtype=int)

    # Offsets into the sorted player observations.
    counts = np.bincount(
        player_frames,
        minlength=N_FRAMES
    )

    offsets = np.zeros(
        N_FRAMES + 1,
        dtype=np.int64
    )

    offsets[1:] = np.cumsum(counts)

    start = max(0, START_FRAME)
    end = min(END_FRAME, N_FRAMES)

    if MAX_FRAMES is not None:
        end = min(end, start + MAX_FRAMES)

    if start >= end:
        raise ValueError("Invalid rendering frame range.")

    background = draw_pitch()
    manifest_rows = []

    print(f"\nRendering frames {start} to {end - 1}")
    print("Output:", FRAME_DIR)

    start_time = time.time()

    for frame in range(start, end):
        filename = f"frame_{frame:06d}.png"
        destination = FRAME_DIR / filename

        if destination.exists() and not OVERWRITE:
            continue

        img = background.copy()

        first = offsets[frame]
        last = offsets[frame + 1]

        for i in range(first, last):
            draw_player(
                img,
                player_x[i],
                player_y[i],
                player_teams[i],
                player_slots[i]
            )

        bx, by = ball_xy[frame]

        ball_visible = draw_ball(
            img, bx, by
        )

        add_header(
            img,
            frame,
            get_half(frame),
            ball_visible,
            last - first
        )

        success = cv2.imwrite(
            str(destination),
            img,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                PNG_COMPRESSION
            ]
        )

        if not success:
            raise IOError(
                f"Failed to write {destination}"
            )

        manifest_rows.append({
            "frame": frame,
            "filename": filename,
            "half": get_half(frame),
            "ball_visible": ball_visible,
            "player_count": int(last - first)
        })

        if (frame - start + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {frame - start + 1:,} "
                f"/ {end - start:,} frames "
                f"({elapsed:.1f}s)"
            )

    # Manifest describes frames written by this run.
    pd.DataFrame(
        manifest_rows,
        columns=[
            "frame",
            "filename",
            "half",
            "ball_visible",
            "player_count"
        ]
    ).to_csv(MANIFEST, index=False)

    elapsed = time.time() - start_time

    print("\nRender complete.")
    print("New images:", len(manifest_rows))
    print("Elapsed seconds:", round(elapsed, 2))
    print("Folder:", FRAME_DIR)


if __name__ == "__main__":
    main()
