
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

PERIODS = [
    (11185, 90652),     # First half
    (110808, 190198),   # Second half
]

def in_play_frames(df):
    mask = pd.Series(False, index=df.index)

    for start, end in PERIODS:
        mask |= df["frame"].between(start, end - 1)

    return df.loc[mask].copy()


# Load original extraction
ball = pd.read_parquet(OUTPUT / "tracking_ball.parquet")
players = pd.read_parquet(OUTPUT / "tracking_players.parquet")

# Restrict to graded periods
ball_filtered = in_play_frames(ball)
players_filtered = in_play_frames(players)

# Identify half for temporal processing
ball_filtered["half"] = (
    (ball_filtered["frame"] >= PERIODS[1][0]).astype("int8") + 1
)

players_filtered["half"] = (
    (players_filtered["frame"] >= PERIODS[1][0]).astype("int8") + 1
)

# Verify frame coverage
assert len(ball_filtered) == 158857
assert ball_filtered["frame"].is_unique

# Save working datasets
ball_filtered.to_parquet(
    OUTPUT / "tracking_ball_graded.parquet",
    index=False
)

players_filtered.to_parquet(
    OUTPUT / "tracking_players_graded.parquet",
    index=False
)

print("Ball frames:", len(ball_filtered))
print("Player observations:", len(players_filtered))
print("Ball visibility:",
      round(ball_filtered["ball_visible"].mean() * 100, 2), "%")

print("\nFrames per half:")
print(ball_filtered.groupby("half").size())
