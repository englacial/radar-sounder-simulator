"""Per-chunk wall_s: new-kernel pilot (outputs/_runtime_bench) vs pilot_fixed."""
import json, glob, re, collections, sys
ref_dirs = {"antarctica_getz": "outputs/antarctica_getz/pilot_fixed/runs",
            "antarctica_david": "outputs/antarctica_david/pilot_fixed/runs"}
def load(d):
    out = {}
    for f in glob.glob(d + "/*.json"):
        j = json.load(open(f)); m = j["meta"] if isinstance(j["meta"], dict) else eval(j["meta"])
        if "brough" not in j["rid"] and "rssnr" not in j["rid"]: continue   # main physics runs only
        out[(m["pass"], m["chunk"])] = (j["wall_s"], j["n_facets_per_interface"], j["n_samples"], m.get("n_traces_total"))
    return out
tot_old = tot_new = 0.0
for line, rd in ref_dirs.items():
    new = load("outputs/_runtime_bench/pilot_smoke/runs"); old = load(rd)
    per = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for k in sorted(new):
        if k in old:
            per[k[0]][0] += old[k][0]; per[k[0]][1] += new[k][0]; per[k[0]][2] += 1
    print(f"== {line}")
    for p, (o, n, c) in per.items():
        print(f"  {p:16s} chunks={c:2d}  old {o:7.1f} s  new {n:7.1f} s  speedup {o/max(n,1e-9):.2f}x"); tot_old += o; tot_new += n
print(f"TOTAL old {tot_old/60:.1f} min  new {tot_new/60:.1f} min  speedup {tot_old/max(tot_new,1e-9):.2f}x")
