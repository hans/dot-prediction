"""Load the BNC photodiode channel for one subject/block from Neuralynx."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from neuralynx_loader import extract_data, load_channel_subset, validate_sampling

raw_dir = Path(snakemake.params.raw_dir)
session_subdir = snakemake.params.session_subdir
if session_subdir is None:
    children = [p for p in raw_dir.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise RuntimeError(
            f"Expected exactly one session subdir under {raw_dir}, found: {children}"
        )
    session_subdir = children[0].name
data_path = raw_dir / session_subdir

channel = int(snakemake.params.channel)
ncs_suffix = snakemake.params.ncs_suffix or ""

blocks = load_channel_subset(data_path, [channel], ncs_suffix, channel_type="BNC")

all_data, all_times = [], []
fs = None
for block in blocks:
    data, times, rates, _units = extract_data(block)
    _, fs_block = validate_sampling(rates)
    if fs is None:
        fs = fs_block
    elif fs != fs_block:
        raise RuntimeError(f"Sampling rate mismatch across blocks: {fs} vs {fs_block}")
    all_data.append(data)
    all_times.append(times)

data = np.concatenate(all_data, axis=0)
times = np.concatenate(all_times, axis=0)

os.makedirs(os.path.dirname(snakemake.output.parquet), exist_ok=True)
pd.DataFrame(
    {"t": times.astype(np.float64), "x": data[:, 0].astype(np.float32)}
).to_parquet(snakemake.output.parquet, index=False)

with open(snakemake.output.meta, "w") as f:
    json.dump(
        {"fs": float(fs), "channel": channel, "session_subdir": session_subdir},
        f,
        indent=2,
    )
