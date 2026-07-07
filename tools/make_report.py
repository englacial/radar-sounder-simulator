"""Build a single self-contained HTML verification report (stdlib only).

Globs ``outputs/verification/*/metrics.json`` and assembles
``outputs/verification/report.html``: a summary table (case x metric, value vs
threshold, green/red pass cells) followed by a section per case with its figures
(PNGs embedded as base64 data URIs) and notes. Cases with a missing or malformed
metrics.json become warnings and never crash the build.

Run: uv run python tools/make_report.py
"""

import base64
import datetime
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "outputs" / "verification"


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


def _version():
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import soundersim
        return soundersim.__version__
    except Exception:
        return "unknown"


def _load_cases(root):
    """Return (cases, warnings). cases: list of dicts with dir/name/doc/figs."""
    cases, warnings = [], []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        figs = sorted(d.glob("*.png"))
        mpath = d / "metrics.json"
        doc, note = None, None
        if not mpath.exists():
            note = f"{d.name}: no metrics.json"
        else:
            try:
                doc = json.loads(mpath.read_text())
                if not isinstance(doc.get("metrics"), dict):
                    raise ValueError("missing 'metrics' object")
            except Exception as e:  # malformed -> warn, still show figures
                doc, note = None, f"{d.name}: malformed metrics.json ({e})"
        if note:
            warnings.append(note)
        if doc is None and not figs:
            continue
        name = (doc or {}).get("case", d.name)
        cases.append({"dir": d, "name": name, "doc": doc, "figs": figs})
    return cases, warnings


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))


# Report sections, in display order. Cases declare their section via the
# doc-level "group" key; _group_of falls back to a name heuristic for older
# artifacts. Unknown groups are appended after these.
GROUP_ORDER = ("simc comparison", "xOPR clutter", "Radar equation comparison")


def _group_of(c):
    g = (c["doc"] or {}).get("group")
    if g:
        return g
    if c["name"].startswith("opr_"):
        return "xOPR clutter"
    if c["name"] == "haynes":
        return "Radar equation comparison"
    return "simc comparison"


def _criterion(e):
    """Human-readable pass criterion: '≤ 1', '≥ 0.99', or '−4 ± 0.05'."""
    thr = _fmt(e.get("threshold"))
    if "target" in e:
        return f'{_fmt(e["target"])} &plusmn; {thr}'
    sym = {"<=": "&le;", ">=": "&ge;", "<": "&lt;", ">": "&gt;"}
    return f'{sym.get(e.get("op", "<="), html.escape(str(e.get("op"))))} {thr}'


def _summary_table(cases):
    metric_names = []
    for c in cases:
        if c["doc"]:
            for k in c["doc"]["metrics"]:
                if k not in metric_names:
                    metric_names.append(k)
    if not metric_names:
        return "<p>No metrics to summarize.</p>"
    head = "".join(f"<th>{html.escape(m)}</th>" for m in metric_names)
    rows = []
    for c in cases:
        cells = [f'<th class="case">{html.escape(c["name"])}</th>']
        metrics = (c["doc"] or {}).get("metrics", {})
        for m in metric_names:
            e = metrics.get(m)
            if e is None:
                cells.append('<td class="na">-</td>')
                continue
            cls = "pass" if e.get("pass") else "fail"
            txt = f'{_fmt(e.get("value"))} ({_criterion(e)})'
            cells.append(f'<td class="{cls}">{txt}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table class="summary"><thead><tr><th>case</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _img_tag(path):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img alt="{html.escape(path.name)}" '
            f'src="data:image/png;base64,{b64}"><figcaption>'
            f'{html.escape(path.name)}</figcaption></figure>')


def _case_section(c):
    parts = [f'<h3 id="{html.escape(c["name"])}">{html.escape(c["name"])}</h3>']
    doc = c["doc"]
    if doc is None:
        parts.append('<p class="warn">No valid metrics.json for this case.</p>')
    else:
        if doc.get("notes"):
            parts.append(f'<p class="notes">{html.escape(doc["notes"])}</p>')
        rows = []
        for name, e in doc["metrics"].items():
            cls = "pass" if e.get("pass") else "fail"
            extras = {k: v for k, v in e.items()
                      if k not in ("value", "threshold", "pass", "op", "target",
                                   "tolerance")}
            extra = ", ".join(f"{k}={_fmt(v)}" for k, v in extras.items())
            rows.append(
                f'<tr><th>{html.escape(name)}</th>'
                f'<td class="{cls}">{_fmt(e.get("value"))}</td>'
                f'<td>{_criterion(e)}</td>'
                f'<td class="{cls}">{"PASS" if e.get("pass") else "FAIL"}</td>'
                f'<td class="extra">{html.escape(extra)}</td></tr>')
        parts.append('<table class="detail"><thead><tr><th>metric</th><th>value</th>'
                     '<th>criterion</th><th>result</th><th>details</th></tr></thead>'
                     f'<tbody>{"".join(rows)}</tbody></table>')
    for f in c["figs"]:
        parts.append(_img_tag(f))
    return "<section>" + "".join(parts) + "</section>"


_CSS = """
body{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#1a1a1a;max-width:1100px}
h1{margin-bottom:.2rem}.meta{color:#555;font-size:.9rem;margin-bottom:1.5rem}
table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}
th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}
td.pass,th.pass{background:#c8f7c5}td.fail,th.fail{background:#f7c5c5}
td.na{background:#eee;color:#888}td.extra{color:#555;font-size:.8rem}
th.case{background:#f0f0f0}
figure{margin:1rem 0}img{max-width:100%;height:auto;border:1px solid #ddd}
figcaption{color:#666;font-size:.8rem}
.warn{background:#fff3cd;border:1px solid #ffe08a;padding:.5rem;border-radius:4px}
.notes{color:#333;font-style:italic}
section{border-top:2px solid #eee;padding-top:.5rem;margin-top:2rem}
"""


def build_report(root):
    """Build the report HTML from ``root`` (outputs/verification); returns str."""
    cases, warnings = _load_cases(root)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    sha, ver = _git_sha(), _version()
    n_fail = sum(1 for c in cases if c["doc"]
                 and any(not e.get("pass") for e in c["doc"]["metrics"].values()))
    meta = (f'generated {now} &middot; soundersim {html.escape(ver)}'
            + (f' &middot; git {html.escape(sha)}' if sha else ''))
    warn_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        warn_html = f'<div class="warn"><b>Warnings</b><ul>{items}</ul></div>'
    status = ("all cases green" if n_fail == 0
              else f"{n_fail} case(s) with failing metrics")
    groups = {}
    for c in cases:
        groups.setdefault(_group_of(c), []).append(c)
    order = [g for g in GROUP_ORDER if g in groups] + sorted(
        g for g in groups if g not in GROUP_ORDER)
    sections = []
    for g in order:
        sections.append(f'<section class="group"><h2>{html.escape(g)}</h2>'
                        + _summary_table(groups[g])
                        + "".join(_case_section(c) for c in groups[g])
                        + "</section>")
    body = (f"<h1>soundersim verification report</h1>"
            f'<p class="meta">{meta}<br>{len(cases)} case(s); {status}.</p>'
            f"{warn_html}" + "".join(sections))
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>soundersim verification</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")


def main():
    if not VERIFY.exists():
        print(f"no verification outputs at {VERIFY}", file=sys.stderr)
        return 1
    html_str = build_report(VERIFY)
    out = VERIFY / "report.html"
    out.write_text(html_str)
    print(f"wrote {out} ({len(html_str) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
