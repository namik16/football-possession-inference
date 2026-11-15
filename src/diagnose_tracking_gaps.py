
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

ball = pd.read_parquet(
    OUT / "tracking_ball_graded.parquet"
).sort_values("frame")

# Calculate consecutive runs of visible/missing ball tracking.
results = []

for half, group in ball.groupby("half"):
    group = group.copy()
    visible = group["ball_visible"].fillna(False)

    run_id = visible.ne(visible.shift()).cumsum()

    runs = (
        group.assign(visible=visible, run_id=run_id)
        .groupby("run_id")
        .agg(
            visible=("visible", "first"),
            start_frame=("frame", "first"),
            end_frame=("frame", "last"),
            frames=("frame", "size")
        )
        .reset_index(drop=True)
    )

    runs["half"] = half
    runs["duration_s"] = runs["frames"] / 25

    results.append(runs)

runs = pd.concat(results, ignore_index=True)
missing = runs[~runs["visible"]]

print("\n--- BALL TRACKING GAPS ---")
print("Total missing runs:", len(missing))
print("Median gap:", round(missing["duration_s"].median(), 2), "s")
print("90th percentile:", round(missing["duration_s"].quantile(.9), 2), "s")
print("Maximum gap:", round(missing["duration_s"].max(), 2), "s")

print("\nLongest missing-ball intervals:")
print(
    missing.nlargest(15, "duration_s")[
        ["half", "start_frame", "end_frame", "duration_s"]
    ].to_string(index=False)
)

runs.to_csv(
    OUT / "tracking_visibility_runs.csv",
    index=False
)
