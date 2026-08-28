"""Search ILATM2 (all versions) 2009-2019 by year and hemisphere; log counts and sizes."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import earthaccess
earthaccess.login(strategy="netrc")
out = {}
for yr in range(2009, 2020):
    for hemi, bbox in (("gl", (-75, 58, -10, 84)), ("aa", (-180, -90, 180, -60))):
        res = earthaccess.search_data(short_name="ILATM2", bounding_box=bbox, temporal=(f"{yr}-01-01", f"{yr}-12-31"))
        vers = {}
        for g in res:
            v = g["umm"].get("CollectionReference", {}).get("Version", "?")
            vers[v] = vers.get(v, 0) + 1
        mb = sum(float(g.size()) for g in res)
        names = sorted({g["umm"]["GranuleUR"] for g in res})
        out[f"{yr}_{hemi}"] = {"n": len(res), "mb": mb, "versions": vers, "first": names[:2], "last": names[-1:]}
        print(yr, hemi, len(res), f"{mb:.0f} MB", vers, names[:1], flush=True)
json.dump(out, open("outputs/cache/atm2/search_log.json", "w"), indent=1)
