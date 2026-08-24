# Antenna realism: mkb60_2023 pilot (pilot_ant_mkb) vs baselines

Branch antenna-realism. Instrument YAMLs now carry the REAL antenna models
(basler195_2017: array_tapered 8-el 0.304-lam, TX taper x hanning(8) RX;
mkb60_basler: finite_dipole cross_track 0.40 lam = the MARFA wing plate);
roll_source nav for both (Roll verified usable in all three seasons: 0% NaN,
2017 -20..+14 deg, 2022 +/-2.6, 2023 +/-8).

## Two-way pattern values (power dB, cross-track plane, rel nadir)

theta:                 30      60      72      75      78     82     84
basler195 tapered:  -28.6   -57.8   -58.4   -59.0   -59.7  -60.6  -60.9
mkb60 fd(0.40):      -3.1   -13.9   -22.7   -25.8   -29.7  -36.7  -41.7
(72-84 band power-avg: -59.7 and -27.8; the fd sits inside the s3 scout's
-16/-36 credible bracket; the 195 model has NO element pattern -> lower
bound on the real rejection.)

## Pilot comparison (mkb60_2023, pilot segment, all rel own surface peak, dB)

                        midcol  bedwin_tot  scout   surf_arm(bw)  bed(bw)
pilot_smoke (iso)       -40.19    -63.01    -2.40     -63.05      -82.57
pilot_ant_mkb (fd)      -42.40    -66.38    -5.51     -66.51      -80.65
pilot_s3_array8         -47.59    -76.82   -10.20     -78.20      -83.21
measured                -44.73    -75.86   -23.52

## Why the element pattern buys only 3.5 dB in the bed window (IMPORTANT)

Verified NOT an implementation problem. Control experiment on the same real
chunk (c00): an AXISYMMETRIC tabulated pattern equal to the fd cross-track
cut gives -25/-29/-32 dB at dt 10-14/14-18/18-22 us (raw surface layer,
matching the closed form), while the committed cross-track-axis fd gives
~0 dB there and array8 gives -14.4 dB. So the sim's deep-delay surface
energy sits OFF the cross-track plane, at diagonal/along-track azimuths
with moderate |u| = |sin(theta) cos(az_ct)| where a cross-track dipole has
g ~ 1 but an 8-element array still suppresses hard. In the PROCESSED sim
that energy survives because of the documented g5 along-track Doppler
aliasing artifact (surface_alias_ratio 4.9): real 0.46 m raw spacing lets
the real SAR reject those arms; the sim's 27 m sim-posting aliases them
into the passband. Consequence:

- The fd model's cross-track annulus rejection IS working (the annulus arm
  share it can touch drops as designed).
- The remaining sim-vs-measured gap in the bed window (-66.4 vs -75.9) is
  the aliasing artifact's floor, NOT antenna physics; --posting-div is the
  existing antidote (raises the alias-limited aperture). array8 matched the
  measured total only by unphysically suppressing the artifact.
- A real wing plate also rolls off fore/aft at grazing (aircraft body,
  finite plate) -- unmodeled here; it would act on real along-track arms
  but cannot fix an aliasing artifact in the sim.

Kernel wiring proof: raw chunk ratio fd/iso and unit tests (g^2 field,
g^4 power at 0/20/30/45/60/75/80/82 deg through coherent + incoherent
kernels; multilayer flat-slab end-to-end with roughness+roll).
