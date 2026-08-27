"""Firn density-core -> permittivity pipeline (Kovacs + segment transfer
matrix), promoted from tools/run_b26_comparison.py after the B26 validation
arc (claude_notes/b26_comparison_findings.md).

Standard firn methodology: model layers carry the SEGMENT-AGGREGATE
transfer-matrix reflectivity of the raw full-resolution core density profile
(effective contrasts). Point-sampled permittivities discard the 0.1-0.5 m
Bragg-scale density strata and are ~12 dB weak in the 20-70 m band --
DEPRECATED for simulation; they remain only as the trend reference the
effective-contrast construction is anchored to.
"""

import numpy as np

from soundersim.config import (DemInterface, Medium, OffsetInterface,
                               RoughnessConfig)


def eps_kovacs(rho_kgm3):
    """Kovacs et al. (1993) / C&S 2020 Eq. (4): eps = (1 + 0.845*rho[g/cc])^2."""
    return (1.0 + 0.845 * np.asarray(rho_kgm3) / 1000.0) ** 2


def load_density_tab(path):
    """(depth_m, rho_kgm3) from a PANGAEA density .tab (comment block, then a
    tab-separated header line starting with 'Depth ice/snow' + data)."""
    lines = path.read_text().splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if ln.startswith("Depth ice/snow"))
    d = np.loadtxt(path, delimiter="\t", skiprows=hdr + 1)
    return d[:, 0], d[:, 1]


def tmm_reflection(n_stack, dz, lam):
    """Normal-incidence transfer-matrix reflection coefficient (Yeh; C&S 2020
    Sec. IV-C) of ``len(n_stack) - 2`` slabs of thickness ``dz`` and indices
    ``n_stack[1:-1]``, between half-spaces n_stack[0] / n_stack[-1]."""
    kx = 2.0 * np.pi / lam * np.asarray(n_stack, np.float64)
    phi = kx[1:-1] * dz

    def interface(m):  # index-matching matrix from medium m to m+1
        q = kx[m + 1] / kx[m]
        return 0.5 * np.array([[1 + q, 1 - q], [1 - q, 1 + q]], complex)

    M = np.eye(2, dtype=complex)
    for m in range(len(phi)):
        M = M @ interface(m) @ np.diag([np.exp(-1j * phi[m]),
                                        np.exp(1j * phi[m])])
    M = M @ interface(len(phi))
    return M[1, 0] / M[0, 0]


class FirnCore:
    """One density core: raw (full-resolution) profile for the effective
    contrasts, plus the lightly smoothed profile for the point-sampled trend.

    The smoothing is an EDGE-NORMALIZED boxcar (the moving average is divided
    by the local window overlap): a plain ``mode='same'`` convolution
    zero-pads beyond the core and halves the deepest samples, which reads as
    a spurious bright deep reflector (run_firn_investigation.load_b26).
    """

    def __init__(self, path, smooth_m=0.1, z_top=1.0):
        self.path = path
        self.z_top = float(z_top)
        z, rho = load_density_tab(path)
        self.z_raw, self.rho_raw = z, rho
        k = int(round(smooth_m / np.median(np.diff(z)))) | 1
        box = np.ones(k)
        self.z = z
        self.rho = (np.convolve(rho, box, "same")
                    / np.convolve(np.ones_like(rho), box, "same"))
        self.eps = eps_kovacs(self.rho)
        self.zmax = float(z.max())

    def point_eps(self, depth):
        """Closest smoothed-sample permittivity at ``depth`` (point sample;
        trend reference only -- deprecated as a simulation contrast)."""
        return float(self.eps[np.argmin(np.abs(self.z - depth))])

    def equal_depths(self, n):
        """n equally spaced layer depths spanning [z_top, zmax]."""
        return np.linspace(self.z_top, self.zmax, n)

    def raw_index(self):
        """(depth_m, refractive index) of the RAW full-resolution core -- the
        0.1 m pre-smoothing alone costs ~1.4 dB of band level, so the
        effective-contrast construction reads the raw profile."""
        return self.z_raw, np.sqrt(eps_kovacs(self.rho_raw))

    def contrast_density(self, lam, window_m):
        """(depth, density): the COHERENT reflectivity envelope of the raw
        profile -- |sum of the Born reflection phasors| in a sliding window of
        ``window_m`` (use the in-firn range resolution, so the peaks are the
        bright horizons the radar can actually resolve).

        This is the same coherent quantity segment_reflectivity() aggregates,
        so its peaks are where a model interface realizes the largest |r|. A
        phase-blind proxy such as |d eps/dz| is NOT equivalent: it peaks
        wherever the density trend is steep, including thick smooth-gradient
        zones whose strata cancel at the Bragg wavenumber (lam/2n ~ 0.47 m)
        and which therefore reflect almost nothing at 195 MHz."""
        z, n = self.raw_index()
        dz = float(np.median(np.diff(z)))
        phi = np.concatenate([[0.0],
                              np.cumsum(2 * np.pi / lam * n[:-1] * np.diff(z))])
        r = (n[:-1] - n[1:]) / (n[:-1] + n[1:])
        w = int(round(window_m / dz)) | 1
        return z[:-1], np.abs(np.convolve(r * np.exp(2j * phi[:-1]),
                                          np.ones(w), "same"))

    def peak_depths(self, n, lam, window_m, min_sep=None, gap_mult=1.5):
        """(depths, prominences, min_sep_m): the ``n`` most prominent peaks of
        contrast_density() inside [z_top, zmax] -- peak-centered placement.

        Two guards keep the set usable as a layer stack. ``min_sep`` (default
        min(window_m, 0.6*span/n) -- the radar cannot resolve interfaces closer
        than a range cell, and the tighter bound keeps n peaks feasible)
        prevents clustering; then, while any gap in the covered extent exceeds
        ``gap_mult`` x the uniform spacing, the least prominent selected peak
        is swapped for the most prominent unselected one inside that gap
        (swapped-in peaks are pinned so the repair cannot oscillate). Without
        the gap repair a purely prominence-ranked set abandons the weakly
        contrasted deep firn: at n=10 it leaves a 30 m hole below 63 m and
        collapses 41 m of profile onto one interface, which drives the
        synthetic eps above solid ice."""
        from scipy.signal import find_peaks

        z, dens = self.contrast_density(lam, window_m)
        dz = float(np.median(np.diff(z)))
        m = (z >= self.z_top) & (z <= self.zmax)
        z, dens = z[m], dens[m]
        span = self.zmax - self.z_top
        sep = float(min(window_m, 0.6 * span / n) if min_sep is None
                    else min_sep)
        while True:
            idx, props = find_peaks(dens, distance=max(1, round(sep / dz)),
                                    prominence=0)
            if len(idx) >= n or sep < 0.2:
                break
            sep /= 2.0
        zp, pr = z[idx], props["prominences"]     # peak depths + prominences
        order = list(np.argsort(pr)[::-1])
        sel, pool, pinned = order[:n], order[n:], set()
        gmax = gap_mult * span / n
        for _ in range(3 * n):
            edges = np.concatenate([[self.z_top], np.sort(zp[sel]),
                                    [self.zmax]])
            gaps = np.diff(edges)
            j = int(np.argmax(gaps))
            if gaps[j] <= gmax:
                break
            cand = [k for k in pool if edges[j] < zp[k] < edges[j + 1]
                    and all(abs(zp[k] - zp[s]) >= sep for s in sel)]
            drop = [k for k in sel if k not in pinned]
            if not cand or not drop:
                break
            best, weakest = cand[0], min(drop, key=lambda k: pr[k])
            sel.remove(weakest), pool.append(weakest)
            sel.append(best), pool.remove(best), pinned.add(best)
            pool.sort(key=lambda k: -pr[k])
        o = np.argsort(zp[sel])
        return np.asarray(zp[sel])[o], np.asarray(pr[sel])[o], sep

    def segment_reflectivity(self, depths, lam, complex_r=False, top=None):
        """|r| (or complex r, referenced to the SEGMENT TOP) of the raw
        full-resolution profile aggregated by transfer matrix over each
        layer's SEGMENT -- segment j spans the midpoints either side of
        depths[j] (the first starts at ``top``, default depths[0]; the last
        ends at the core end). The profile ABOVE that first edge is what the
        air-firn surface interface already represents and is not covered.

        ``top`` matters only for non-uniform placements, where depths[0] is not
        z_top: passing top=z_top keeps the covered extent (and hence the total
        aggregated reflectivity) identical to the uniform stack's, which is
        what makes the two placements comparable."""
        z, n = self.raw_index()
        dz = float(np.median(np.diff(z)))
        d = np.asarray(depths, np.float64)
        bnd = np.concatenate([[d[0] if top is None else float(top)],
                              (d[:-1] + d[1:]) / 2.0, [z[-1]]])
        out = []
        for a, b in zip(bnd[:-1], bnd[1:]):
            s, e = int(np.searchsorted(z, a)), int(np.searchsorted(z, b))
            stack = np.concatenate(([n[s - 1] if s else 1.0], n[s:e],
                                    [n[min(e, len(n) - 1)]]))
            rj = tmm_reflection(stack, dz, lam)
            out.append(rj if complex_r else abs(rj))
        return np.array(out)

    def effective_contrast_eps(self, depths, lam, top=None):
        """(eps[len(depths)+1], |r|[len(depths)]): synthetic permittivities
        for firn0..firn_{N-1} + substrate whose PLAIN Fresnel interface
        contrasts equal segment_reflectivity().

        firn0 keeps its point-sampled value (the air-firn surface interface
        and the surface-peak normalization ride on it); thereafter
        n_{j+1} = n_j (1 +- r_j)/(1 -+ r_j) with the sign taken to land
        closest to the point-sampled Kovacs trend, which keeps the sequence
        bounded around it instead of drifting."""
        r = self.segment_reflectivity(depths, lam, top=top)
        trend = np.array([self.point_eps(d) for d in depths]
                         + [self.point_eps(float(depths[-1]) + 1.0)])
        n_tr = np.sqrt(trend)
        n = np.empty(len(trend))
        n[0] = n_tr[0]
        for j, rj in enumerate(r):
            cand = [n[j] * (1.0 + s * rj) / (1.0 - s * rj) for s in (1.0, -1.0)]
            n[j + 1] = min(cand, key=lambda v: abs(v - n_tr[j + 1]))
        return n ** 2, r


def firn_stack(depths, eps, att_db_per_km, roughness=None):
    """(media, interfaces) for a conformal firn stack under a DEM surface:
    media air + firn0..firn_{N-1} + substrate (firn + substrate attenuating at
    ``att_db_per_km`` one-way), interfaces surface + OffsetInterface L{i} at
    surface - depths[i]. ``eps`` must have len(depths)+1 entries
    (firn0..firn_{N-1}, substrate); ``roughness`` = (sigma_m[], corr_len_m[]
    [, acf]) per layer attaches sub-facet roughness to every INTERNAL
    interface (the air-firn surface stays smooth); ``acf`` "gaussian"
    (default) or "exponential" (RoughnessConfig.acf)."""
    e = np.asarray(eps, np.float64)
    if e.shape != (len(depths) + 1,):
        raise ValueError(f"eps must have {len(depths) + 1} entries")
    sig, cl, *acf = (None, None) if roughness is None else roughness
    acf = acf[0] if acf else "gaussian"
    media = [Medium(name="air", eps_r=1.0)]
    ifaces = [DemInterface(name="surface")]
    for i, d in enumerate(depths):
        media.append(Medium(name=f"firn{i}", eps_r=float(e[i]),
                            attenuation_db_per_km=att_db_per_km))
        rc = None if roughness is None else RoughnessConfig(
            sigma_m=float(sig[i]), corr_length_m=float(cl[i]), acf=acf)
        ifaces.append(OffsetInterface(name=f"L{i}", reference="surface",
                                      offset=-float(d), roughness=rc))
    media.append(Medium(name="substrate", eps_r=float(e[-1]),
                        attenuation_db_per_km=att_db_per_km))
    return media, ifaces
