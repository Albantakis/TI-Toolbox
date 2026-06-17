"""Tests for multipolar exhaustive-search helpers."""

import numpy as np
import pytest

from tit.calc import (
    MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM_AVG,
    MTI_METRIC_BOTZANOWSKI_MAGNITUDE_AM,
    MTI_METRIC_RECURSIVE_TI,
    compute_mti_metric_field,
)
from tit.opt.config import MExConfig
from tit.opt.mex.logic import count_multipolar_combinations, generate_multipolar_combinations


def _buckets():
    return {
        "e1_plus": ["LA1"],
        "e1_minus": ["RA1"],
        "e2_plus": ["LA2"],
        "e2_minus": ["RA2"],
        "e3_plus": ["LP1"],
        "e3_minus": ["RP1"],
        "e4_plus": ["LP2"],
        "e4_minus": ["RP2"],
    }


def test_bucket_mode_generates_one_eight_electrode_candidate():
    combos = list(generate_multipolar_combinations(_buckets()))

    assert combos == [
        ("LA1", "RA1", "LA2", "RA2", "LP1", "RP1", "LP2", "RP2")
    ]


def test_bucket_mode_requires_all_eight_electrodes_unique():
    buckets = _buckets()
    buckets["e4_minus"] = ["LA1"]

    assert count_multipolar_combinations(buckets) == 0


def test_symmetric_bucket_mode_pairs_each_plus_minus_bucket():
    mirror_map = {
        "LA1": "RA1",
        "LA2": "RA2",
        "LP1": "RP1",
        "LP2": "RP2",
    }

    assert count_multipolar_combinations(_buckets(), symmetry_mirror_map=mirror_map) == 1


def test_cross_pair_symmetric_bucket_mode_pairs_e1_e3_and_e2_e4():
    buckets = {
        "e1_plus": ["LA1"],
        "e1_minus": ["LP1"],
        "e2_plus": ["LA2"],
        "e2_minus": ["LP2"],
        "e3_plus": ["RA1"],
        "e3_minus": ["RP1"],
        "e4_plus": ["RA2"],
        "e4_minus": ["RP2"],
    }
    mirror_map = {
        "LA1": "RA1",
        "LP1": "RP1",
        "LA2": "RA2",
        "LP2": "RP2",
    }

    combos = list(
        generate_multipolar_combinations(
            buckets,
            symmetry_mirror_map=mirror_map,
            symmetry_pairing="cross_pairs",
        )
    )

    assert combos == [("LA1", "LP1", "LA2", "LP2", "RA1", "RP1", "RA2", "RP2")]


def test_mex_config_coerces_metric_and_electrodes():
    cfg = MExConfig(
        subject_id="001",
        leadfield_hdf="net_leadfield.hdf5",
        roi_name="target",
        electrodes=_buckets(),
        mti_metric="botzanowski_magnitude_am",
    )

    assert isinstance(cfg.electrodes, MExConfig.BucketElectrodes)
    assert cfg.roi_name == "target.csv"
    assert cfg.mti_metric is MExConfig.MTIMetric.BOTZANOWSKI_MAGNITUDE_AM


def test_compute_mti_metric_field_supported_metrics():
    fields = [
        np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.3, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.4]]),
        np.array([[1.0, 1.0, 0.0], [0.1, 0.1, 0.0]]),
    ]

    for metric in (
        MTI_METRIC_RECURSIVE_TI,
        MTI_METRIC_BOTZANOWSKI_MAGNITUDE_AM,
        MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM_AVG,
    ):
        values = compute_mti_metric_field(fields, metric)
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))


def test_compute_mti_metric_field_rejects_unknown_metric():
    fields = [np.zeros((2, 3)) for _ in range(4)]

    with pytest.raises(ValueError, match="Unsupported mTI metric"):
        compute_mti_metric_field(fields, "not_a_metric")
