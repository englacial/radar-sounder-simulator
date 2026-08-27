"""HAPS 14 km design study driver: write instrument + experiment YAMLs for a
round of designs, run the pilot, and tabulate the synthetic-pass metrics.

    uv run python claude_notes/haps_design_study/gen.py r1 designs_r1.json [--posting-div N] [--dry]

Design dict keys: name, f0_MHz, bw_MHz, T_us, window (hann|hamming|none),
n_el, taper (uniform|hann|hamming|taylor), span_m (default 10).
Element spacing = span/(n_el-1) converted to carrier wavelengths.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]
C = 299792458.0


def weights(kind, n):
    if kind == "uniform":
        return [1.0] * n
    if kind == "hann":
        return [round(float(v), 4) for v in np.hanning(n + 2)[1:-1]]
    if kind == "hamming":
        return [round(float(v), 4) for v in np.hamming(n)]
    if kind.startswith("taylor"):
        from scipy.signal.windows import taylor
        sll = float(kind[6:] or 30)
        return [round(float(v), 4) for v in taylor(n, nbar=4, sll=sll, norm=False)]
    raise ValueError(kind)


def instrument_yaml(d):
    lam = C / (d["f0_MHz"] * 1e6)
    n = d["n_el"]
    span = d.get("span_m", 10.0)
    sp_lam = span / ((n - 1) * lam) if n > 1 else 0.0
    if n == 1:
        ant = {"kind": "isotropic", "roll_source": "none"}
    elif d.get("taper", "uniform") == "uniform":
        ant = {"kind": "array", "n_elements": n, "spacing_lam": round(sp_lam, 5),
               "roll_source": "none"}
    else:
        w_tx = d["tx_w"] if "tx_w" in d else weights(d.get("taper_tx", d["taper"]), n)
        w_rx = d["rx_w"] if "rx_w" in d else weights(d.get("taper_rx", d["taper"]), n)
        ant = {"kind": "array_tapered", "n_elements": n,
               "spacing_lam": round(sp_lam, 5), "tx_weights": w_tx,
               "rx_weights": w_rx, "roll_source": "none"}
    return {
        "schema_version": 1, "name": f"hd_{d['name']}",
        "description": f"HAPS design study {d['name']}: {d['f0_MHz']} MHz / "
                       f"{d['bw_MHz']} MHz, {d['T_us']} us, {n} el over {span} m "
                       f"({sp_lam:.2f} lam), taper {d.get('taper','uniform')}",
        "source": {"kind": "stated"},
        "simulated": {"frequency_Hz": d["f0_MHz"] * 1e6,
                      "bandwidth_Hz": d["bw_MHz"] * 1e6,
                      "pulse_length_s": d["T_us"] * 1e-6,
                      "window": d.get("window", "hann"),
                      "construction": "chirp", "antenna": ant},
        "provenance": "design study 2026-08-26, see claude_notes/haps_design_study/",
    }


def experiment_yaml(rnd, designs, posting_div, srough=True):
    name = f"wc_hd_{rnd}"
    extra = {f"hd_{d['name']}": {"carrier": "reference", "altitude_m": 14000.0,
                                 "instrument": f"hd_{d['name']}",
                                 "facet_spacing_scale": 0.7} for d in designs}
    return name, {
        "schema_version": 1,
        "meta": {"name": name, "status": "exploratory", "role": "study",
                 "backs": f"HAPS 14 km design study round {rnd}"},
        "run": {"line": "greenland_westcoast", "segment": "pilot",
                "passes": ["p3_2017"] + list(extra),
                "bed": {"source": "picked"}, "extra_passes": extra,
                "reflectivity": {"gamma_from_rssnr": True,
                                 "specular_diffuse": {"specular_fraction": 0.5,
                                                      "tilt_s0_deg": 3.0}},
                "physics": {"att_db_per_km": "solve", "surface_roughness": srough,
                            "bed_roughness": {"sigma_m": 0.10, "corr_length_m": 0.886}},
                "processing": {"chain": "standard", "proc_cache": True,
                               "companion": False, "posting_div": posting_div},
                "out_name": name}}


def tabulate(name, designs):
    m = json.load(open(ROOT / "outputs/greenland_westcoast" / name / "metrics.json"))["metrics"]
    rows = []
    for d in designs:
        k = f"hd_{d['name']}"
        v = m.get(f"{k}_bed_visibility")
        if not v:
            rows.append((d["name"], None)); continue
        dec = v["decomposition_db"]
        rows.append((d["name"], v["value"], v["bed_rel_surf_db"], v["midcol_rel_surf_db"],
                     dec["bed"]["bed_rel_surf_db"], dec["surface"]["bed_rel_surf_db"],
                     m["simulation_wall_s"]["per_pass_s"].get(k),
                     v["geometry"].get("aperture_m"), v["geometry"].get("facet_spacing_m")))
    print(f"{'design':18s} {'bedvis':>7s} {'bed/surf':>8s} {'midcol':>7s} {'bedarm':>7s} {'surfarm':>7s} {'wall_s':>7s} {'aper_m':>7s} {'facet':>6s}")
    for r in rows:
        if r[1] is None:
            print(f"{r[0]:18s}  (missing)"); continue
        print(f"{r[0]:18s} {r[1]:7.2f} {r[2]:8.2f} {r[3]:7.2f} {r[4]:7.2f} {r[5]:7.2f} {r[6]:7.1f} {r[7]:7.0f} {r[8]:6.1f}")
    return rows


if __name__ == "__main__":
    rnd, dfile = sys.argv[1], sys.argv[2]
    pdiv = int(sys.argv[sys.argv.index("--posting-div") + 1]) if "--posting-div" in sys.argv else 1
    designs = json.load(open(Path(__file__).parent / dfile))
    for d in designs:
        (ROOT / "config/instruments" / f"hd_{d['name']}.yaml").write_text(
            yaml.safe_dump(instrument_yaml(d), sort_keys=False))
    name, exp = experiment_yaml(rnd, designs, pdiv, srough="--no-srough" not in sys.argv)
    (ROOT / "config/experiments" / f"{name}.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    if "--dry" in sys.argv:
        sys.exit(0)
    if "--tabulate-only" not in sys.argv:
        log = ROOT / "claude_notes/logs" / f"{name}.log"
        with open(log, "w") as fh:
            rc = subprocess.call(["uv", "run", "python", "tools/run_basal_clutter.py",
                                  "--config", f"config/experiments/{name}.yaml"],
                                 cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print("exit", rc, "log", log)
    tabulate(name, designs)
