"""Detect rising/falling photodiode edges on the task window.

Emits a parquet with one row per edge. `t_peak` and `idx` are relative to the
start of the task window (i.e. `expt_start_time` is subtracted off), matching
how the downstream alignment expects them.
"""
import json

import pandas as pd

from photodiode import detect_events_env_direction

df = pd.read_parquet(snakemake.input.parquet)
with open(snakemake.input.meta) as f:
    meta = json.load(f)
fs = meta["fs"]

t0 = snakemake.params.expt_start_time
t1 = snakemake.params.expt_end_time
mask = (df["t"] >= t0) & (df["t"] <= t1)
sub = df.loc[mask].reset_index(drop=True)
# Edge times from detect_events_env_direction are samples-since-start-of-sub,
# i.e. already relative to the task window.

edges = detect_events_env_direction(
    sub["x"].to_numpy(),
    fs=fs,
    **dict(snakemake.params.detector),
)

rows = []
for kind in ("rise", "fall"):
    for t_peak, t_half, idx in zip(
        edges[kind]["t_peaks"],
        edges[kind]["t_half"],
        edges[kind]["idx_peaks"],
    ):
        rows.append(
            {"kind": kind, "t_peak": float(t_peak), "t_half": float(t_half), "idx": int(idx)}
        )
pd.DataFrame(rows).sort_values("t_peak").reset_index(drop=True).to_parquet(
    snakemake.output.edges, index=False
)
