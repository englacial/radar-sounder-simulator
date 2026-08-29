"""Parse Bed SCR + tail guard per pass from a run log."""
import re, sys, numpy as np
def parse(path):
    cur=None; scr={}; guard={}; mid={}
    for ln in open(path):
        m=re.match(r"== (\S+) ",ln)
        if m: cur=m.group(1); scr.setdefault(cur,[])
        m=re.search(r"bed-window bed - surface returns ([+-][\d.]+) dB",ln)
        if m and cur: scr[cur].append(float(m.group(1)))
        m=re.search(r"bed-return tail (\S+):.*guard picked_bed (ok|FAIL) \(([+-][\d.]+) dB\)",ln)
        if m: guard[m.group(1)]=(m.group(2),float(m.group(3)))
        if ln.startswith("clutter (midcol"):
            for k,v in re.findall(r"(\w+): [-\d.]*/([-\d.]+) \[",ln): mid[k]=float(v)
    return scr,guard,mid
for path in sys.argv[1:]:
    scr,guard,mid=parse(path)
    print(f"--- {path}")
    print(f"{'pass':>8}{'n':>3}{'BedSCR med':>12}{'min':>8}{'max':>8}{'guard':>9}{'midcol':>9}")
    for k,v in scr.items():
        if not v: continue
        g=guard.get(k,('?',float('nan')))
        print(f"{k:>8}{len(v):3d}{np.median(v):12.1f}{min(v):8.1f}{max(v):8.1f}{g[1]:9.1f}{mid.get(k,float('nan')):9.1f}")
