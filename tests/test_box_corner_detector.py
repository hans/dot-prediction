"""Fixture-only tests for box_corner_detector (no video I/O)."""

import cv2
import numpy as np
import pytest

from src.box_corner_detector import detect_box_corner


def _make_corner_frame(size: int = 200, corner_xy: tuple[int, int] = (100, 100)) -> np.ndarray:
    """200×200 BGR image with a sharp dark upper-left / bright lower-right corner.

    Above and left of corner_xy: dark (0). Below and right: bright (200).
    Produces a strong Harris response at corner_xy.
    """
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = corner_xy
    frame[cy:, cx:] = 200
    return frame


def test_centred_prediction_detected_within_2px():
    """Predict at the true corner location; expect detection within 2 px."""
    frame = _make_corner_frame()
    result = detect_box_corner(frame, predicted_xy=(100.0, 100.0), corner="bl")
    assert result is not None
    x, y = result
    assert abs(x - 100) <= 2
    assert abs(y - 100) <= 2


def test_x_left_offset_prediction_detected():
    """Predict 15 px left of corner (x−15), 15 px above (y−15). Corner still in window."""
    frame = _make_corner_frame()
    # prediction (85, 85): window x=[65,105], y=[80,105] — truth (100,100) inside both.
    result = detect_box_corner(frame, predicted_xy=(85.0, 85.0), corner="bl")
    assert result is not None
    x, y = result
    assert abs(x - 100) <= 2
    assert abs(y - 100) <= 2


def test_x_right_offset_prediction_detected():
    """Predict 15 px right of corner (x+15), 15 px above (y−15). Corner still in window."""
    frame = _make_corner_frame()
    # prediction (115, 85): window x=[95,135], y=[80,105] — truth (100,100) inside both.
    result = detect_box_corner(frame, predicted_xy=(115.0, 85.0), corner="bl")
    assert result is not None
    x, y = result
    assert abs(x - 100) <= 2
    assert abs(y - 100) <= 2


def test_uniform_patch_returns_none():
    """Uniform patch has no corner; expect None."""
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = detect_box_corner(frame, predicted_xy=(100.0, 100.0), corner="bl")
    assert result is None


def test_nan_prediction_returns_none():
    """NaN prediction coordinates return None without raising."""
    frame = _make_corner_frame()
    assert detect_box_corner(frame, (float("nan"), float("nan")), corner="bl") is None
    assert detect_box_corner(frame, (100.0, float("nan")), corner="bl") is None


def test_prediction_outside_frame_returns_none():
    """Prediction far outside the frame; window is empty → None."""
    frame = _make_corner_frame()
    result = detect_box_corner(frame, predicted_xy=(5000.0, 5000.0), corner="bl")
    assert result is None


def test_corner_bl_and_br_both_accept_valid_corner():
    """corner='bl' and corner='br' both accept the same synthetic corner without error."""
    frame = _make_corner_frame()
    for corner in ("bl", "br"):
        result = detect_box_corner(frame, predicted_xy=(100.0, 100.0), corner=corner)
        assert result is not None, f"corner={corner!r} returned None unexpectedly"
