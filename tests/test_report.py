"""Fast CI test for the HTML report builder (tools/make_report.build_report)."""

import base64
import importlib.util
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
_spec = importlib.util.spec_from_file_location("make_report", TOOLS / "make_report.py")
make_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_report)

# 1x1 PNG so the base64 embed exercises real image bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _case(root, name, metrics, *, notes=None, png=True, raw=None):
    d = root / name
    d.mkdir(parents=True)
    if raw is not None:
        (d / "metrics.json").write_text(raw)
    else:
        doc = {"case": name, "created": "2026-07-07T00:00:00+00:00", "metrics": metrics}
        if notes:
            doc["notes"] = notes
        (d / "metrics.json").write_text(json.dumps(doc))
    if png:
        (d / "fig.png").write_bytes(PNG)


def test_build_report(tmp_path):
    _case(tmp_path, "goodcase",
          {"m1": {"value": 0.5, "threshold": 1.0, "pass": True, "extra": 3}},
          notes="a passing case")
    _case(tmp_path, "BadCase",  # arbitrary case (mixed-case dir name)
          {"m1": {"value": 5.0, "threshold": 1.0, "pass": False}})
    _case(tmp_path, "brokencase", {}, raw="{not valid json", png=True)

    html = make_report.build_report(tmp_path)

    assert "goodcase" in html and "BadCase" in html
    assert 'class="fail"' in html          # failing metric -> red cell
    assert "data:image/png;base64," in html  # embedded image
    assert "a passing case" in html          # notes rendered
    assert "brokencase" in html and "malformed" in html.lower()  # tolerated warning
    assert "<!doctype html>" in html.lower()
