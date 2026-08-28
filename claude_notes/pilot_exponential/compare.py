"""Fixture (Gaussian) vs ATM exponential-ACF surface roughness: pilot comparison.

Reads two finished pilot runs per line -- the fixture control
(outputs/<case>/pilot, 2026-08-27) and the exponential arm
(outputs/<case>/pilot_exponential, this study) -- and writes

  outputs/pilot_exponential_comparison/<line>_{radargrams,decomposition,
      decomposition_trace}.png     stacked PNGs (no re-rendering)
  outputs/pilot_exponential_comparison/metrics_comparison.{md,csv}

Nothing here simulates: the figures are the run's own PNGs stacked with a
label bar, the table is metrics.json arithmetic.

Run: uv run python claude_notes/pilot_exponential/compare.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "pilot_exponential_comparison"
FIXTURE, EXPONENTIAL = "pilot", "pilot_exponential"
# line -> outputs/<case prefix>
CASES = {"antarctica_david": "antarctica_david",
         "antarctica_getz": "antarctica_getz",
         "greenland_geikie01_transit": "greenland_geikie",
         "greenland_westcoast": "greenland_westcoast"}
FIGS = ("radargrams", "decomposition", "decomposition_trace")
BAR, PAD = 46, 10


# ------------------------------------------------------------------ figures
def _font(size=30):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def stack(top_path, bot_path, out_path, top_label, bot_label):
    """Two PNGs stacked vertically, each under a labelled bar, widths matched."""
    ims = [Image.open(p).convert("RGB") for p in (top_path, bot_path)]
    w = max(im.width for im in ims)
    ims = [im if im.width == w else
           im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
           for im in ims]
    h = sum(im.height for im in ims) + 2 * BAR + 3 * PAD
    out = Image.new("RGB", (w, h), "white")
    d, f = ImageDraw.Draw(out), _font()
    y = PAD
    for im, lab in zip(ims, (top_label, bot_label)):
        d.rectangle([0, y, w, y + BAR], fill=(30, 30, 30))
        d.text((12, y + BAR // 2), lab, fill="white", font=f, anchor="lm")
        y += BAR
        out.paste(im, (0, y))
        y += im.height + PAD
    out.save(out_path)
    return out_path


# ------------------------------------------------------------------- metrics
def load(case, run):
    p = ROOT / "outputs" / case / run / "metrics.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def _n(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def rows_for(line, fix, exp):
    """One row per comparable number: (metric, field, measured, fix, exp)."""
    out = []

    def add(metric, field, meas, a, b, flag=""):
        if _n(a) is None and _n(b) is None:
            return
        out.append({"line": line, "metric": metric, "field": field,
                    "measured": _n(meas), "fixture": _n(a),
                    "exponential": _n(b), "flag": flag})

    keys = list(dict.fromkeys(list(fix) + list(exp)))
    for k in keys:
        a, b = fix.get(k, {}), exp.get(k, {})
        va, vb = _n(a.get("value")), _n(b.get("value"))
        pa, pb = a.get("pass"), b.get("pass")
        flag = "PASS-FLIP" if (pa is not None and pb is not None
                               and bool(pa) != bool(pb)) else ""
        if k.startswith("clutter_"):
            for sub, tag in (("midcol_rel_surf_db", "mid-column"),
                             ("bed_rel_surf_db", "bed rel surf")):
                add(k, tag,
                    (a.get("measured") or b.get("measured") or {}).get(sub),
                    (a.get("sim") or {}).get(sub), (b.get("sim") or {}).get(sub),
                    flag)
            continue
        if k == "altitude_trend":
            pairs = {**(a.get("pairs") or {}), **(b.get("pairs") or {})}
            for pk in pairs:
                add(k, pk,
                    (a.get("pairs", {}).get(pk) or
                     b.get("pairs", {}).get(pk) or {}).get("measured_db"),
                    a.get("pairs", {}).get(pk, {}).get("sim_db"),
                    b.get("pairs", {}).get(pk, {}).get("sim_db"), flag)
            continue
        if k.startswith("bed_return_tail_"):
            sa = a.get("bed_return_tail_slope_db_per_us") or {}
            sb = b.get("bed_return_tail_slope_db_per_us") or {}
            add(k, "tail slope dB/us", sa.get("measured") or sb.get("measured"),
                sa.get("picked_bed"), sb.get("picked_bed"), flag)
            ea = (a.get("bed_return_tail_excess_db") or {}).get("picked_bed", {})
            eb = (b.get("bed_return_tail_excess_db") or {}).get("picked_bed", {})
            for d in ("+1us", "+2us", "+3us"):
                add(k, f"excess {d}", None, ea.get(d), eb.get(d), flag)
            # the guard: sim bed returns minus sim surface returns over the fit
            # window. A FAIL means the tail is surface clutter and the slope /
            # excess above are upper bounds -- so a flip is the headline.
            ga = ((a.get("sim") or {}).get("picked_bed", {}) or {}).get("guard", {})
            gb = ((b.get("sim") or {}).get("picked_bed", {}) or {}).get("guard", {})
            gflag = ("GUARD-FLIP" if (ga.get("pass") is not None
                                      and gb.get("pass") is not None
                                      and bool(ga["pass"]) != bool(gb["pass"]))
                     else "")
            add(k, "guard bed-surf min dB",
                None, ga.get("min_bed_minus_surface_returns_db"),
                gb.get("min_bed_minus_surface_returns_db"), gflag)
            continue
        if k == "simulation_wall_s":
            add(k, "total s", None, va, vb)
            pa_, pb_ = a.get("per_pass_s") or {}, b.get("per_pass_s") or {}
            for p in dict.fromkeys(list(pa_) + list(pb_)):
                add(k, f"{p} s", None, pa_.get(p), pb_.get(p))
            continue
        if k.startswith("surface_alignment_"):
            add(k, "rms bins", None, va, vb, flag)
            add(k, "offset bins", None, a.get("offset_bins"),
                b.get("offset_bins"), flag)
            continue
        if k == "rssnr_level_residuals":
            add(k, "median residual dB", None, va, vb, flag)
            ra, rb = (a.get("per_pass_residual_db") or {},
                      b.get("per_pass_residual_db") or {})
            for p in dict.fromkeys(list(ra) + list(rb)):
                add(k, f"{p} residual dB", None, ra.get(p), rb.get(p))
            continue
        add(k, "value", None, va, vb, flag)
    return out


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, figs = [], []
    for line, case in CASES.items():
        for name in FIGS:
            a = ROOT / "outputs" / case / FIXTURE / f"{name}.png"
            b = ROOT / "outputs" / case / EXPONENTIAL / f"{name}.png"
            if a.exists() and b.exists():
                figs.append(stack(
                    a, b, OUT / f"{line}_{name}.png",
                    f"{line}  {name}  |  FIXTURE Gaussian sigma 4.9 cm, l 2.98 m"
                    f"  (outputs/{case}/{FIXTURE})",
                    f"{line}  {name}  |  ATM EXPONENTIAL ACF"
                    f"  (outputs/{case}/{EXPONENTIAL})"))
        fix, exp = load(case, FIXTURE), load(case, EXPONENTIAL)
        if fix and exp:
            rows += rows_for(line, fix, exp)

    for r in rows:
        f_, e_, m_ = r["fixture"], r["exponential"], r["measured"]
        r["delta"] = None if None in (f_, e_) else e_ - f_
        r["err_fixture"] = None if None in (f_, m_) else f_ - m_
        r["err_exponential"] = None if None in (e_, m_) else e_ - m_

    cols = ["line", "metric", "field", "measured", "fixture", "exponential",
            "delta", "err_fixture", "err_exponential", "flag"]
    with (OUT / "metrics_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows({c: r.get(c) for c in cols} for r in rows)

    md = ["# Fixture (Gaussian) vs ATM exponential-ACF surface roughness",
          "",
          "delta = exponential - fixture; err = sim - measured (dB unless the "
          "field says otherwise).", ""]
    for line in CASES:
        lr = [r for r in rows if r["line"] == line]
        if not lr:
            continue
        md += [f"## {line}", "",
               "| metric | field | measured | fixture | exponential | delta | "
               "err fix | err exp |", "|---|---|---|---|---|---|---|---|"]
        for r in lr:
            md.append("| {metric} | {field} | {meas} | {fix} | {exp} | {d} | "
                      "{ef} | {ee} |{flag}".format(
                          metric=r["metric"], field=r["field"],
                          meas=fmt(r["measured"]), fix=fmt(r["fixture"]),
                          exp=fmt(r["exponential"]), d=fmt(r["delta"]),
                          ef=fmt(r["err_fixture"]), ee=fmt(r["err_exponential"]),
                          flag=(" " + r["flag"]) if r["flag"] else "").rstrip())
        md.append("")
    (OUT / "metrics_comparison.md").write_text("\n".join(md))
    print(f"{len(rows)} rows -> {OUT/'metrics_comparison.md'}")
    for p in figs:
        print(f"  fig {p}")


if __name__ == "__main__":
    main()
