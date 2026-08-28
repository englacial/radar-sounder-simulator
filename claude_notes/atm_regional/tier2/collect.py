"""Concatenate site-visit rows -> outputs/atm_regional/tier2/rows.parquet + summary_interim.md"""
import pandas as pd, numpy as np, time
from common import OUT

ROWS = OUT / "rows"


def main():
    ps = sorted(ROWS.glob("*.parquet"))
    if not ps: return None
    df = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    df.to_parquet(OUT / "rows.parquet", index=False)
    ok = df[df.status == "ok"] if "status" in df else df.iloc[0:0]
    lines = [f"# Tier 2 interim ({time.ctime()})", "", f"rows {len(df)}, ok {len(ok)}, sites {df.site.nunique()}, "
             f"status counts {df.status.value_counts().to_dict()}", ""]
    if len(ok):
        lines.append(f"MB pulled {df.mb.sum() / 1e3:.1f} GB; best family: {ok.best.value_counts(normalize=True).round(2).to_dict()}")
        lines.append(f"adequate (white + |Bragg 1.5,1.0 m| < 3 dB): {ok.adequate.mean():.2f}; bragg-only {ok.adequate_bragg_only.mean():.2f}")
        g = ok.groupby(["hemi", "stratum"]).agg(n=("site", "size"), e_sigma_cm=("e_sigma", lambda x: np.nanmedian(x) * 100), e_l=("e_l", "median"),
                                                 nu=("m_nu", "median"), H=("pl_H", "median"), adequate=("adequate", "mean"),
                                                 mis15=("bragg_195MHz_vs_best", "median"), mis10=("bragg_300MHz_vs_best", "median"),
                                                 exp_frac=("best", lambda x: (x == "exponential").mean()))
        lines.append(""); lines.append(g.round(2).to_string())
    (OUT / "summary_interim.md").write_text("\n".join(lines))
    print("\n".join(lines[:4]))
    return df


if __name__ == "__main__":
    main()
