"""Layered delay + attenuation utility tests."""

import numpy as np
import pytest

from soundersim.physics import (C, attenuation_loss_db, db_to_linear_power,
                                in_medium_speed, optical_path_length,
                                two_way_delay)


def test_in_medium_speed():
    assert in_medium_speed(1.0) == pytest.approx(C)
    assert in_medium_speed(3.17) == pytest.approx(C / np.sqrt(3.17))


def test_slab_nadir_two_way_delay_exact():
    """Flat-slab nadir two-way delay is 2h/c + 2d*sqrt(eps)/c exactly."""
    h, d, eps = 1000.0, 300.0, 3.17
    twtt = two_way_delay([h, d], [1.0, eps])
    expected = 2.0 * h / C + 2.0 * d * np.sqrt(eps) / C
    assert twtt == pytest.approx(expected, rel=0, abs=1e-15)


def test_optical_path_length_vectorized():
    lengths = np.array([[1000.0, 300.0], [500.0, 100.0]])
    eps = np.array([1.0, 3.17])
    opl = optical_path_length(lengths, eps)
    assert opl.shape == (2,)
    assert opl[0] == pytest.approx(1000.0 + 300.0 * np.sqrt(3.17))


def test_attenuation_accumulation_round_numbers():
    # 1 km at 10 dB/km + 2 km at 5 dB/km = 20 dB one-way loss.
    loss = attenuation_loss_db([1000.0, 2000.0], [10.0, 5.0])
    assert loss == pytest.approx(20.0)
    assert db_to_linear_power(loss) == pytest.approx(0.01)
    assert db_to_linear_power(0.0) == pytest.approx(1.0)
    assert db_to_linear_power(30.0) == pytest.approx(1e-3)
