
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games import asset_paths, GAMES, GAME

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

paths = asset_paths()
raw_path = Path(paths["atd_raw"])
meta_path = Path(paths["atd_meta"])

if not raw_path.is_file() or not meta_path.is_file():
    raise FileNotFoundError("Missing tactical ATD files in data/tactical/")

pitch_length, pitch_width = GAMES[GAME]["pitch"]
expected_frames = GAMES[GAME]["n_video_frames"]

# Map each tracking slot to its team.
metadata = ET.parse(meta_path).getroot()
teams = list(metadata.iter("Team"))
players = list(metadata.iter("Player"))

assert len(teams) == 2

home_id = teams[0].get("id")
team_by_slot = [
    "H" if player.get("teamId") == home_id else "A"
    for player in players
]


def parse_xyz(group):
    values = group.strip().split(",")
    if len(values) != 3:
        raise ValueError(f"Invalid tracking observation: {group!r}")
    return tuple(float(v) for v in values)


ball_rows = []
player_rows = []
seen_frames = set()

with raw_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        parts = line.strip().split(":")

        if len(parts) != 3:
            raise ValueError("Unexpected ATD line format")

        frame_id = int(parts[0])
        frame = frame_id - 1

        if frame in seen_frames:
            raise ValueError(f"Duplicate frame: {frame}")
        seen_frames.add(frame)

        bx, by, bs = parse_xyz(parts[2])

        ball_rows.append({
            "frame": frame,
            "ball_x": bx,
            "ball_y": by,
            "ball_speed": bs,
            "ball_visible": np.isfinite(bx) and np.isfinite(by),
        })

        observations = parts[1].split(";")
        if len(observations) != len(team_by_slot):
            raise ValueError(
                f"Frame {frame}: expected {len(team_by_slot)} "
                f"slots, found {len(observations)}"
            )

        for slot, group in enumerate(observations):
            x, y, speed = parse_xyz(group)

            # Preserve observed players, not missing slot records.
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            player_rows.append({
                "frame": frame,
                "slot": slot + 1,
                "team": team_by_slot[slot],
                "x": x,
                "y": y,
                "speed": speed,
            })

ball = pd.DataFrame(ball_rows).sort_values("frame")
players_df = pd.DataFrame(player_rows).sort_values(
    ["frame", "slot"]
)

assert len(ball) == expected_frames
assert np.array_equal(
    ball["frame"].to_numpy(),
    np.arange(expected_frames)
)

ball.to_parquet(OUTPUT / "tracking_ball.parquet", index=False)
players_df.to_parquet(
    OUTPUT / "tracking_players.parquet", index=False
)

print("Extraction complete")
print("Frames:", len(ball))
print("Player observations:", len(players_df))
print("Ball visibility:", round(ball["ball_visible"].mean() * 100, 2), "%")
print("Home observations:", (players_df["team"] == "H").sum())
print("Away observations:", (players_df["team"] == "A").sum())
print("Saved to:", OUTPUT)
