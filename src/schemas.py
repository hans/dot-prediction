"""Pandera schemas for the tabular inputs to the pipeline.

Two CSVs feed the analyses:

- ``data/{subject}/behavior/data.csv`` — the raw task log, one row per
  *subtrial* (a single reveal or click within a trial). Validated by
  :class:`BehaviorSchema`.

- ``data/{subject}/model_outputs/model_outputs.csv`` — per-particle posterior
  predictions from cognitive models (LoT, Linear), joined against the behavior
  log. One row per ``(seq_id, tpt, model, model_particle)`` — i.e. 20 particles
  × 2 models rows per subtrial. Validated by :class:`ModelOutputsSchema`.

All timestamps are epoch-milliseconds from the task machine unless noted. All
spatial coordinates are normalized to the display, roughly x∈[0, 1] and
y∈[0, 0.75] (the ``min_x``/``max_x``/``min_y``/``max_y`` columns carry the
exact bounds used per trial).

Usage
-----
>>> import pandas as pd
>>> from schemas import BehaviorSchema, ModelOutputsSchema
>>> df = BehaviorSchema.validate(pd.read_csv("data/EC347/behavior/data.csv"))
"""

from __future__ import annotations

from typing import Optional

import pandera.pandas as pa
from pandera.typing import Series


class BehaviorSchema(pa.DataFrameModel):
    """Raw task log.

    Grain: one row per ``(trial_idx, tpt)`` subtrial. A trial consists of an
    intro auto-reveal phase (``tpt`` ∈ {0, 1, 2}, no click) followed by
    prediction subtrials where the subject clicks before each reveal. Rows
    with ``tpt < 3`` therefore have NaN ``response_*`` / ``correct`` fields.

    Trial 0 is a warm-up (no ECoG recorded) and is filtered out in
    ``notebooks/align_behavior.py``.
    """

    response_x: Series[float] = pa.Field(
        nullable=True,
        description="Subject click x, normalized display coords. NaN on intro reveals.",
    )
    response_y: Series[float] = pa.Field(
        nullable=True,
        description="Subject click y, normalized display coords. NaN on intro reveals.",
    )
    response_time: Series[float] = pa.Field(
        nullable=True,
        description="Epoch-ms when the subject clicked. NaN on intro reveals.",
    )
    reveal_time: Series[int] = pa.Field(
        description="Epoch-ms when the true dot for this tpt was revealed.",
    )
    trial_onset: Series[int] = pa.Field(
        description="Epoch-ms of trial start (repeated across rows of a trial).",
    )
    trial_offset: Series[int] = pa.Field(
        description="Epoch-ms of trial end (repeated across rows of a trial).",
    )
    end_of_auto_reveal: Series[int] = pa.Field(
        description="Epoch-ms marking the end of the three intro reveals.",
    )
    true_x: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description="Ground-truth dot x for this tpt (normalized).",
    )
    true_y: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description="Ground-truth dot y for this tpt (normalized).",
    )
    tpt: Series[int] = pa.Field(
        ge=0,
        description="Subtrial index within the trial (0-indexed). tpt<3 are intro reveals.",
    )
    trial_idx: Series[int] = pa.Field(
        ge=0,
        description="Trial index. Trial 0 is a warm-up (no ECoG); downstream code drops it.",
    )
    correct: Series[float] = pa.Field(
        nullable=True,
        description="1.0/0.0 whether the subject's prediction was correct; NaN on intro reveals.",
    )
    seq_id: Series[str] = pa.Field(
        description="Name of the latent pattern driving the trial (e.g. 'zigzag_widening').",
    )
    subject_id: Series[str] = pa.Field(
        description="Subject code (e.g. 'EC347', 'NP168').",
    )
    min_x: Series[float] = pa.Field(description="Display x lower bound (usually 0).")
    max_x: Series[float] = pa.Field(description="Display x upper bound (usually 1).")
    min_y: Series[float] = pa.Field(description="Display y lower bound (usually 0).")
    max_y: Series[float] = pa.Field(description="Display y upper bound (usually 0.75).")
    seq_attempt: Series[float] = pa.Field(
        nullable=True,
        description=(
            "Which attempt at this sequence this trial represents (1-indexed). "
            "Logger-version dependent: EC347 emits 1 on every row; NP168 emits "
            "NaN on every row. Stored as float to accommodate the all-NaN case."
        ),
    )
    expt_start_time: Series[int] = pa.Field(
        description="Epoch-ms when the experiment process started (constant within a session).",
    )
    last_reveal_time: Optional[Series[float]] = pa.Field(
        nullable=True,
        description="Epoch-ms of the preceding reveal. Only present in newer logs (e.g. NP168).",
    )
    time_since_reveal: Optional[Series[float]] = pa.Field(
        nullable=True,
        description="Ms between last reveal and this response. Only in newer logs.",
    )

    class Config:
        strict = False  # allow extra columns added downstream (e.g. l2_error, rt_duration)
        coerce = True


class ModelOutputsSchema(pa.DataFrameModel):
    """Per-particle model predictions joined against behavior.

    Grain: one row per ``(seq_id, tpt, model, model_particle)``. Each
    ``(seq_id, tpt, model)`` group holds ``n_particles`` rows (20 in the
    reference export) whose ``model_posterior`` values sum to 1 once the
    model has made at least one observation (posterior is NaN for ``tpt=0``
    before any reveal).

    The marginal columns (``model_marg_*``) are the posterior-weighted mean
    across particles within the same ``(seq_id, tpt, model)`` group, so they
    are constant within that group.

    Note: the source README (``data/{subject}/model_outputs/README.md``) also
    lists an ``is_top_particle`` flag. It is not present in the reference
    export shipped with this repo; treat it as optional.
    """

    # Identifier / behavioral columns mirrored from the task log.
    tpt: Series[int] = pa.Field(ge=0, description="Subtrial index within trial.")
    seq_id: Series[str] = pa.Field(description="Latent-pattern name for the trial.")
    response_x: Series[float] = pa.Field(nullable=True, description="Subject click x.")
    response_y: Series[float] = pa.Field(nullable=True, description="Subject click y.")
    response_time: Series[float] = pa.Field(nullable=True, description="Epoch-ms of click.")
    reveal_time: Series[int] = pa.Field(description="Epoch-ms when truth revealed.")
    trial_onset: Series[int] = pa.Field(description="Epoch-ms trial start.")
    trial_offset: Series[int] = pa.Field(description="Epoch-ms trial end.")
    end_of_auto_reveal: Series[int] = pa.Field(description="Epoch-ms end of intro reveals.")
    true_x: Series[float] = pa.Field(ge=0.0, le=1.0, description="Ground-truth x.")
    true_y: Series[float] = pa.Field(ge=0.0, le=1.0, description="Ground-truth y.")
    trial_idx: Series[int] = pa.Field(ge=0, description="Trial index.")
    correct: Series[float] = pa.Field(nullable=True, description="1.0/0.0/NaN correctness.")
    subject_id: Series[str] = pa.Field(description="Subject code.")
    min_x: Series[float] = pa.Field(description="Display x lower bound.")
    max_x: Series[float] = pa.Field(description="Display x upper bound.")
    min_y: Series[float] = pa.Field(description="Display y lower bound.")
    max_y: Series[float] = pa.Field(description="Display y upper bound.")
    prev_x: Series[float] = pa.Field(
        nullable=True,
        description="Ground-truth x of the previous tpt (NaN at tpt=0).",
    )
    prev_y: Series[float] = pa.Field(
        nullable=True,
        description="Ground-truth y of the previous tpt (NaN at tpt=0).",
    )
    seq_attempt: Series[int] = pa.Field(ge=1, description="Sequence attempt (1-indexed).")
    expt_start_time: Series[int] = pa.Field(description="Epoch-ms experiment start.")
    last_reveal_time: Series[float] = pa.Field(
        nullable=True, description="Epoch-ms of preceding reveal."
    )
    time_since_reveal: Series[float] = pa.Field(
        nullable=True, description="Ms between last reveal and this response."
    )

    # Subject-behavior-derived error metrics (constant across particles in a group).
    prediction_error_x: Series[float] = pa.Field(
        nullable=True, description="Signed subject error, response_x - true_x."
    )
    prediction_error_y: Series[float] = pa.Field(
        nullable=True, description="Signed subject error, response_y - true_y."
    )
    prediction_error: Series[float] = pa.Field(
        ge=0.0, nullable=True, description="L2 distance ||response - truth|| (subject)."
    )
    relative_prediction_error: Series[float] = pa.Field(
        ge=0.0,
        nullable=True,
        description="Subject L2 error divided by step distance ||truth - prev_truth||.",
    )

    # Model-specific columns — one row per particle.
    model: Series[str] = pa.Field(
        isin=["LoT", "Linear"],
        description="Cognitive model family generating the particle set.",
    )
    model_posterior: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        nullable=True,
        description=(
            "Posterior probability of this particle under ``model`` at this "
            "(seq_id, tpt). Sums to 1 across particles once the model has seen "
            "≥1 reveal; NaN at tpt=0."
        ),
    )
    model_particle: Series[int] = pa.Field(
        ge=1,
        description="Particle / hypothesis id within the model's fixed set (1..n_particles).",
    )
    model_response_x: Series[float] = pa.Field(
        nullable=True, description="Predicted click x for this particle."
    )
    model_response_y: Series[float] = pa.Field(
        nullable=True, description="Predicted click y for this particle."
    )
    model_prediction_error_x: Series[float] = pa.Field(
        nullable=True, description="Signed particle error, model_response_x - true_x."
    )
    model_prediction_error_y: Series[float] = pa.Field(
        nullable=True, description="Signed particle error, model_response_y - true_y."
    )
    model_prediction_error: Series[float] = pa.Field(
        ge=0.0, nullable=True, description="L2 distance ||model_response - truth|| (particle)."
    )
    model_relative_prediction_error: Series[float] = pa.Field(
        ge=0.0,
        nullable=True,
        description="Particle L2 error divided by step distance.",
    )
    model_marg_prediction_error: Series[float] = pa.Field(
        ge=0.0,
        nullable=True,
        description=(
            "Posterior-weighted mean of ``model_prediction_error`` across "
            "particles within (seq_id, tpt, model). Constant within the group."
        ),
    )
    model_marg_relative_prediction_error: Series[float] = pa.Field(
        ge=0.0,
        nullable=True,
        description="Posterior-weighted mean of relative error, as above.",
    )

    class Config:
        strict = False
        coerce = True
