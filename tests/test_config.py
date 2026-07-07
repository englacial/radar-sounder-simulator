from soundersim.config import FacetConfig, RadarConfig, SimConfig


def test_json_round_trip():
    cfg = SimConfig(
        mode="incoherent",
        split_sides=True,
        radar=RadarConfig(dt=1e-9, n_samples=2048, t0=6e-6),
        facets=FacetConfig(spacing=None),
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
