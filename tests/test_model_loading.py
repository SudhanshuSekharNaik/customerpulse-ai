"""Tests for model serialization, loading, and artifact persistence."""

import os
import pytest
from ml.segmentation.predict import CustomerSegmentPredictor
from ml.uplift.predict import UpliftPredictor


def test_segment_predictor_loading():
    pred = CustomerSegmentPredictor()
    assert pred is not None


def test_uplift_predictor_loading():
    pred = UpliftPredictor()
    assert pred is not None
