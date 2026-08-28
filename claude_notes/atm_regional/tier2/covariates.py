"""Sample climate covariates (SMB, melt, 10 m wind) at tier-2 sites and add a proxy facies class.

Inputs (outputs/cache/covariates/, all nearest-grid-cell sampling via KD-tree on lat/lon -> unit vectors):
  Greenland: RACMO2.3p3 FGRN11 (11 km) monthly smb & snowmelt Sep2000-Dec2018 (Zenodo 10.5281/zenodo.4013856);
             RACMO2.3p2 11 km ff10m 1961-1990 monthly climatology (Zenodo 10.5281/zenodo.3368405);
             NSIDC-0533 MEaSUREs daily surface melt 25 km EASE2 (melt-season files, 2010-2012) -> melt days [optional].
  Antarctica: RACMO2.3p2 ANT27 (27 km) monthly smb, snowmelt, ff10m 1979-2022 (Zenodo 10.5281/zenodo.7845736).
Climatology period: 2009-2018 (GL, limited by dataset end) / 2009-2019 (AA). Antarctic melt days are not available; a
proxy from monthly snowmelt is used (months with melt > 1 mm w.e. counted, scaled) -- see melt_days_source column.
Output: outputs/atm_regional/tier2/covariates.csv
"""
import glob, os
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__)).split('/claude_notes')[0]
CACHE = f'{ROOT}/outputs/cache/covariates'
SITES = f'{ROOT}/outputs/atm_regional/tier2/sites.csv'
OUT = f'{ROOT}/outputs/atm_regional/tier2/covariates.csv'
YEARS = dict(gl=(2009, 2018), aa=(2009, 2019))


def xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def sampler(lat2d, lon2d):
    la, lo = np.ravel(lat2d), np.ravel(lon2d); ok = np.isfinite(la) & np.isfinite(lo); ids = np.flatnonzero(ok)
    tree = cKDTree(xyz(la[ok], lo[ok]))
    def f(lat, lon):
        d, i = tree.query(xyz(lat, lon))
        return np.unravel_index(ids[i], lat2d.shape), 2 * np.arcsin(d / 2) * 6371.0  # idx, distance km
    return f


def racmo_time(ds):
    return pd.Timestamp('1950-01-01') + pd.to_timedelta(ds['time'].values, 'D')


def annual(ds, var, y0, y1):
    """Monthly sums (kg m-2 = mm w.e.) -> mean annual total over [y0, y1]; also mean #months with melt>1 mm."""
    da = ds[var].squeeze('height', drop=True)
    da = da.assign_coords(time=racmo_time(ds)).sel(time=slice(f'{y0}-01', f'{y1}-12'))
    yr = da.groupby('time.year')
    return yr.sum('time').mean('year').values, (da > 1).groupby('time.year').sum('time').mean('year').values


def melt_days_nsidc(files):
    """NSIDC-0533: daily melt flag on 25 km EASE2 grid -> mean melt days/yr (season files only)."""
    if not files: return None
    ds0 = xr.open_dataset(files[0])
    var = [v for v in ds0.data_vars if 'melt' in v.lower()][0]
    lat, lon = ds0['latitude'].values, ds0['longitude'].values
    acc, years = {}, set()
    for f in files:
        d = xr.open_dataset(f)
        y = int(os.path.basename(f).split('_')[1][:4]); years.add(y)
        m = (d[var].squeeze().values == 51)  # flags: 50 no melt, 51 melt, 90 missing, 91 masked
        acc[y] = acc.get(y, 0) + m.astype(int)
    md = np.mean([acc[y] for y in years], axis=0)
    return md, lat, lon, f'NSIDC-0533 v1 (MEaSUREs Greenland Surface Melt Daily 25km EASE2; May-Sep {min(years)}-{max(years)})'


def facies_proxy(hemi, h, lat, dist_km, melt_days, smb):
    """Proxy facies. Elevation/latitude rule first, then override by melt days if available:
       Greenland: ablation below ELA = 1600 m at 65N falling linearly to 1000 m at 80N;
                  percolation up to dry-snow line = 2500 m at 66N falling to 2000 m at 78N; dry snow above.
       Antarctica: shelf if dist_km==0 or h<100 m; coastal if h<1500 m; interior above.
       Melt override (if melt_days finite): <1 d/yr dry snow, 1-30 percolation, >30 wet snow, or ablation if SMB<0."""
    if hemi == 'gl':
        ela = np.interp(lat, [65, 80], [1600, 1000]); dsl = np.interp(lat, [66, 78], [2500, 2000])
        f = 'ablation' if h < ela else ('percolation' if h < dsl else 'dry_snow')
    else:
        f = 'shelf' if (dist_km == 0 or h < 100) else ('coastal' if h < 1500 else 'interior')
    if np.isfinite(melt_days):
        if hemi == 'gl' or f != 'shelf':
            m = 'dry_snow' if melt_days < 1 else ('percolation' if melt_days <= 30 else 'wet_snow')
            if melt_days > 30 and np.isfinite(smb) and smb < 0: m = 'ablation'
            f = m if hemi == 'gl' else f'{f}_{m}'
        return f, 'melt_days+elevation_rule'
    return f, 'elevation_rule'


def main():
    sites = pd.read_csv(SITES)
    out = sites[['site', 'hemi', 'lat', 'lon', 'h', 'dist_km']].copy()
    for c in ['smb_mmwe', 'melt_mmwe', 'melt_months', 'melt_days', 'wind10_ms', 'cell_dist_km']: out[c] = np.nan
    out['melt_days_source'] = ''; out['source'] = ''

    # ---- Greenland ----
    g = sites.hemi == 'gl'
    if g.any():
        smb = xr.open_dataset(f'{CACHE}/smb_monthlyS_Rp3_GRL_200009_201812.nc', decode_times=False)
        mlt = xr.open_dataset(f'{CACHE}/snowmelt_monthlyS_Rp3_GRL_200009_201812.nc', decode_times=False)
        samp = sampler(smb.lat.values, smb.lon.values)
        idx, dkm = samp(sites.lat[g].values, sites.lon[g].values)
        a_smb, _ = annual(smb, 'smb', *YEARS['gl']); a_mlt, n_mm = annual(mlt, 'snowmelt', *YEARS['gl'])
        out.loc[g, 'smb_mmwe'] = a_smb[idx]; out.loc[g, 'melt_mmwe'] = a_mlt[idx]; out.loc[g, 'melt_months'] = n_mm[idx]
        out.loc[g, 'cell_dist_km'] = dkm
        w = xr.open_dataset(f'{CACHE}/ff10m_1961-1990_ymonmean.nc', decode_times=False)
        wg = xr.open_dataset(f'{CACHE}/aux11km/RACMO23_masks_ZGRN11.nc', decode_times=False)  # LAT/LON in ff10m file are junk
        widx, _ = sampler(wg.lat.values, wg.lon.values)(sites.lat[g].values, sites.lon[g].values)
        out.loc[g, 'wind10_ms'] = w.ff10m.mean('time').values[widx]
        src = ('SMB/melt: RACMO2.3p3 FGRN11 11km monthly 2009-2018, doi:10.5281/zenodo.4013856; '
               'wind: RACMO2.3p2 11km ff10m 1961-1990 clim, doi:10.5281/zenodo.3368405')
        md = melt_days_nsidc(sorted(glob.glob(f'{CACHE}/nsidc0533/*.nc')))
        if md is not None:
            mdays, la, lo, mds = md
            midx, _ = sampler(la, lo)(sites.lat[g].values, sites.lon[g].values)
            out.loc[g, 'melt_days'] = mdays[midx]; out.loc[g, 'melt_days_source'] = mds; src += '; melt days: ' + mds
        else:
            out.loc[g, 'melt_days'] = np.clip(n_mm[idx] * 20, 0, None) * (a_mlt[idx] > 1)
            out.loc[g, 'melt_days_source'] = 'proxy: 20 d per RACMO month with melt>1 mm w.e.'
        out.loc[g, 'source'] = src

    # ---- Antarctica ----
    a = sites.hemi == 'aa'
    if a.any():
        smb = xr.open_dataset(f'{CACHE}/smb_monthlyS_ANT27_ERA5-3H_RACMO2.3p2_197901_202212.nc', decode_times=False)
        mlt = xr.open_dataset(f'{CACHE}/snowmelt_monthlyS_ANT27_ERA5-3H_RACMO2.3p2_197901_202212.nc', decode_times=False)
        w = xr.open_dataset(f'{CACHE}/ff10m_monthlyA_ANT27_ERA5-3H_RACMO2.3p2_197901_202212.nc', decode_times=False)
        samp = sampler(smb.lat.values, smb.lon.values)
        idx, dkm = samp(sites.lat[a].values, sites.lon[a].values)
        a_smb, _ = annual(smb, 'smb', *YEARS['aa']); a_mlt, n_mm = annual(mlt, 'snowmelt', *YEARS['aa'])
        wd = w.ff10m.squeeze('height', drop=True)
        wm = wd.assign_coords(time=racmo_time(w)).sel(time=slice('2009-01', '2019-12')).mean('time').values
        out.loc[a, 'smb_mmwe'] = a_smb[idx]; out.loc[a, 'melt_mmwe'] = a_mlt[idx]; out.loc[a, 'melt_months'] = n_mm[idx]
        out.loc[a, 'wind10_ms'] = wm[idx]; out.loc[a, 'cell_dist_km'] = dkm
        out.loc[a, 'melt_days'] = np.clip(n_mm[idx] * 20, 0, None) * (a_mlt[idx] > 1)
        out.loc[a, 'melt_days_source'] = 'proxy: 20 d per RACMO month with melt>1 mm w.e.'
        out.loc[a, 'source'] = 'RACMO2.3p2 ANT27 27km monthly smb/snowmelt/ff10m 2009-2019, doi:10.5281/zenodo.7845736'

    fp = [facies_proxy(r.hemi, r.h, r.lat, r.dist_km, r.melt_days, r.smb_mmwe) for r in out.itertuples()]
    out['facies_proxy'] = [x[0] for x in fp]; out['facies_source'] = [x[1] for x in fp]
    cols = ['site', 'hemi', 'facies_proxy', 'facies_source', 'melt_days', 'melt_days_source', 'melt_mmwe', 'melt_months',
            'smb_mmwe', 'wind10_ms', 'cell_dist_km', 'source']
    out[cols].to_csv(OUT, index=False, float_format='%.3f')
    print(out.groupby(['hemi', 'facies_proxy']).size()); print(out[cols[4:11]].describe().T)


if __name__ == '__main__':
    main()
