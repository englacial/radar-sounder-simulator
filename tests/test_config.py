import pytest

from soundersim.config import FacetConfig, Medium, RadarConfig, SimConfig
from soundersim.physics import fresnel_normal


def test_json_round_trip():
    cfg = SimConfig(
        mode="coherent",
        split_sides=True,
        radar=RadarConfig(dt=1e-9, n_samples=2048, t0=6e-6, f0=195e6),
        facets=FacetConfig(spacing=25.0),
        media=[Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17)],
    )
    assert SimConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_defaults():
    cfg = SimConfig(
        mode="coherent",
        radar=RadarConfig(dt=2e-9, n_samples=512, t0=0.0),
        facets=FacetConfig(),
    )
    assert cfg.split_sides is False
    assert cfg.facets.spacing is None
    assert cfg.radar.c == 299792458.0
    assert cfg.radar.f0 is None
    # Default media: air then ice, ordered top-down.
    assert [m.name for m in cfg.media] == ["air", "ice"]
    assert cfg.media[1].eps_r == pytest.approx(3.17)


def test_wavelength():
    rc = RadarConfig(dt=1e-9, n_samples=512, t0=0.0, f0=195e6)
    assert rc.wavelength == pytest.approx(299792458.0 / 195e6)
    with pytest.raises(ValueError):
        RadarConfig(dt=1e-9, n_samples=512, t0=0.0).wavelength


def test_fresnel_normal_air_ice():
    """Normal-incidence scalar Fresnel coefficient, sign preserved."""
    assert fresnel_normal(1.0, 3.17) == pytest.approx(-0.2807, abs=1e-4)
    assert fresnel_normal(3.17, 1.0) == pytest.approx(0.2807, abs=1e-4)
