"""M21 dev probe: flat-surface pedestal numbers (delta vs chirp vs referee).

Measures the numbers the report-case gates in tests/test_waveform_pedestal.py
are set from. Session artifact, not part of the main flows.
"""

import time

import numpy as np

from soundersim.compare.multifreq import multifreq_profile
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.nav import nav_to_frame
from soundersim.physics import C, fresnel_normal
from soundersim.scene import LocalFrame, build_facets
from soundersim import synthetic as syn
from soundersim.waveform import compressed_pulse, convolve_fast_time

H, ELEV, EXTENT, POST = 500.0, 500.0, 600.0, 4.0
F0, BW, PL = 195e6, 30e6, 10e-6
DT, NSAMP = 5e-9, 512
T0 = 2.0 * (H - 10.0) / C
GAMMA = fresnel_normal(1.0, 3.17)
EPS_ICE = 3.17
TWTT = T0 + np.arange(NSAMP) * DT


def scene_arrays(scene):
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame,
                          spacing=POST)
    track = nav_to_frame(scene.nav_llh, frame)
    return facets, track


def run_kernel(facets, track, interp):
    field, dropped = coherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, facets.e1, facets.e2, k=2 * np.pi * F0 / C, gamma=GAMMA,
        t0=T0, dt=DT, n_samples=NSAMP, c=C, interp_bins=interp)
    return field, dropped


def db(x, ref):
    return 10.0 * np.log10(np.maximum(x, 1e-300) / ref)


def main():
    scene = syn.flat_scene(elevation=ELEV, altitude=H, extent=EXTENT,
                           posting=POST, n_traces=3)
    facets, track = scene_arrays(scene)
    print(f"{len(facets.centers)} facets")

    t = time.time()
    f_delta, _ = run_kernel(facets, track, False)
    f_interp, _ = run_kernel(facets, track, True)
    print(f"kernel runs {time.time() - t:.1f} s")

    p, m = compressed_pulse(BW, PL, DT, "hann")
    y_chirp = convolve_fast_time(f_delta.astype(np.complex128), p, m)
    y_chirp_i = convolve_fast_time(f_interp.astype(np.complex128), p, m)

    mid = 1
    pos = track.positions[mid]
    t = time.time()
    kw = dict(gamma=GAMMA, f0=F0, bandwidth=BW, c=C, twtt=TWTT, n_freq=128)
    y_ref = multifreq_profile(pos, facets.centers, facets.normals,
                              facets.areas, facets.e1, facets.e2, **kw)
    y_frz = multifreq_profile(pos, facets.centers, facets.normals,
                              facets.areas, facets.e1, facets.e2,
                              freeze_amplitudes=True, **kw)
    print(f"referee {time.time() - t:.1f} s")

    # apparent depth axis (below the surface return, in-ice speed)
    r_min = np.linalg.norm(pos - facets.centers, axis=1).min()
    tau0 = 2.0 * r_min / C
    depth = (TWTT - tau0) * C / (2.0 * np.sqrt(EPS_ICE))
    region = (depth >= 5.0) & (depth <= 80.0)
    print(f"surface return at bin {(tau0 - T0)/DT:.2f}; region bins "
          f"{region.sum()}")

    pd = (np.abs(f_delta[mid]) ** 2)
    pc = (np.abs(y_chirp[mid]) ** 2)
    pci = (np.abs(y_chirp_i[mid]) ** 2)
    pr = np.abs(y_ref) ** 2
    pz = np.abs(y_frz) ** 2

    for name, prof in (("delta", pd), ("chirp", pc), ("chirp_interp", pci),
                       ("referee", pr), ("frozen", pz)):
        d = db(prof, prof.max())
        print(f"{name:14s} peak {prof.max():.3e}  region median "
              f"{np.median(d[region]):7.2f} dB  p90 "
              f"{np.percentile(d[region], 90):7.2f} dB  max "
              f"{d[region].max():7.2f} dB")

    # residuals in region (conv vs referee), where referee above -X dB of its peak
    for name, prof in (("chirp-vs-ref", pc), ("chirpI-vs-ref", pci)):
        dd = db(prof, prof.max())[region] - db(pr, pr.max())[region]
        print(f"{name:14s} median {np.median(np.abs(dd)):.3f} dB  p90 "
              f"{np.percentile(np.abs(dd), 90):.3f} dB  max "
              f"{np.abs(dd).max():.3f} dB")

    # directivity variation: full vs frozen
    good = region & (pr > pr.max() * 1e-10)
    dd = db(pr, pr.max())[good] - db(pz, pz.max())[good]
    print(f"directivity (full-frozen) median {np.median(np.abs(dd)):.3f} "
          f"p90 {np.percentile(np.abs(dd), 90):.3f} max {np.abs(dd).max():.3f} dB")
    resid_energy = 10 * np.log10(np.abs(y_ref - y_frz)[region].sum() ** 2
                                 / np.abs(y_ref)[region].sum() ** 2)
    e2 = 10 * np.log10((np.abs(y_ref - y_frz)[region] ** 2).sum()
                       / (np.abs(y_ref)[region] ** 2).sum())
    print(f"directivity residual energy in region: {e2:.2f} dB")

    # rough surface
    rough = syn.sinusoid_scene(amplitude=0.3, wavelength=150.0, elevation=ELEV,
                               altitude=H, extent=EXTENT, posting=POST,
                               n_traces=3)
    fr, trr = scene_arrays(rough)
    f_r, _ = run_kernel(fr, trr, True)
    y_r = convolve_fast_time(f_r.astype(np.complex128), p, m)
    posr = trr.positions[mid]
    y_rr = multifreq_profile(posr, fr.centers, fr.normals, fr.areas, fr.e1,
                             fr.e2, **kw)
    prr = np.abs(y_rr) ** 2
    pcr = np.abs(y_r[mid]) ** 2
    dd = db(pcr, pcr.max())[region] - db(prr, prr.max())[region]
    print(f"rough chirpI-vs-ref median {np.median(np.abs(dd)):.3f} p90 "
          f"{np.percentile(np.abs(dd), 90):.3f} max {np.abs(dd).max():.3f} dB")
    d = db(prr, prr.max())
    print(f"rough referee region median {np.median(d[region]):.2f} dB")


if __name__ == "__main__":
    main()
