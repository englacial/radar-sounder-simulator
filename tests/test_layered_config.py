"""Config tests for the multilayer stack (media/interface plumbing)."""

import pytest

from soundersim.config import (DemInterface, FacetConfig, FlatInterface,
                               Medium, OffsetInterface, RadarConfig, SimConfig)


def _radar():
    return RadarConfig(dt=1e-9, n_samples=512, t0=0.0, f0=195e6)


def test_medium_attenuation_default_and_set():
    assert Medium(name="ice", eps_r=3.17).attenuation_db_per_km == 0.0
    m = Medium(name="ice", eps_r=3.17, attenuation_db_per_km=12.5)
    assert m.attenuation_db_per_km == 12.5


def test_default_single_interface_backward_compatible():
    cfg = SimConfig(mode="coherent", radar=_radar(), facets=FacetConfig())
    assert len(cfg.media) == 2
    assert len(cfg.interfaces) == 1
    assert isinstance(cfg.interfaces[0], DemInterface)


def test_media_interface_count_validation():
    with pytest.raises(ValueError):
        SimConfig(mode="coherent", radar=_radar(), facets=FacetConfig(),
                  media=[Medium(name="air", eps_r=1.0),
                         Medium(name="ice", eps_r=3.17),
                         Medium(name="bed", eps_r=6.0)],
                  interfaces=[FlatInterface(elevation=500.0)])  # 3 media, 1 iface


def test_three_media_two_interfaces_ok():
    cfg = SimConfig(
        mode="coherent", radar=_radar(), facets=FacetConfig(),
        media=[Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17),
               Medium(name="bed", eps_r=6.0)],
        interfaces=[FlatInterface(name="surface", elevation=500.0),
                    FlatInterface(name="bed", elevation=200.0)])
    assert len(cfg.interfaces) == 2


def test_dem_interface_exactly_one_source():
    DemInterface()               # scene default, ok
    DemInterface(path="/x.tif")  # ok
    DemInterface(ref="bed")      # ok
    with pytest.raises(ValueError):
        DemInterface(path="/x.tif", ref="bed")


def test_offset_reference_validation():
    common = dict(mode="coherent", radar=_radar(), facets=FacetConfig(),
                  media=[Medium(name="air", eps_r=1.0),
                         Medium(name="firn", eps_r=2.5),
                         Medium(name="ice", eps_r=3.17)])
    SimConfig(interfaces=[FlatInterface(name="surface", elevation=500.0),
                          OffsetInterface(reference="surface", offset=-2.0)],
              **common)
    SimConfig(interfaces=[FlatInterface(name="surface", elevation=500.0),
                          OffsetInterface(reference=0, offset=-2.0)], **common)
    with pytest.raises(ValueError):  # unknown name
        SimConfig(interfaces=[FlatInterface(name="surface", elevation=500.0),
                              OffsetInterface(reference="bed", offset=-2.0)],
                  **common)
    with pytest.raises(ValueError):  # out-of-range index
        SimConfig(interfaces=[FlatInterface(name="surface", elevation=500.0),
                              OffsetInterface(reference=5, offset=-2.0)],
                  **common)


@pytest.mark.parametrize("iface", [
    DemInterface(name="bed", path="/data/bed.tif"),
    FlatInterface(name="bed", elevation=123.5),
    OffsetInterface(name="firn1", reference="surface", offset=-2.0),
])
def test_interface_json_round_trip(iface):
    cfg = SimConfig(
        mode="coherent", radar=_radar(), facets=FacetConfig(),
        media=[Medium(name="air", eps_r=1.0),
               Medium(name="ice", eps_r=3.17, attenuation_db_per_km=8.0),
               Medium(name="sub", eps_r=6.0)],
        interfaces=[DemInterface(name="surface"), iface])
    back = SimConfig.model_validate_json(cfg.model_dump_json())
    assert back == cfg
    assert type(back.interfaces[1]) is type(iface)
