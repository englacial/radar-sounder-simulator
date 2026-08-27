# Exponential-ACF option for the sub-facet roughness model (2026-08-27)

Branch `roughness-exponential` (off `roughness-b1`), worktree
`.claude/worktrees/agent-a80f3fac14b3b6f96`. Not merged. Docs:
`docs/roughness.md` ("Exponential correlation function"). Code:
`src/soundersim/roughness.py` (`acf_spectrum`, `d_phi(..., acf=)`),
`kernels/coherent.py`, `kernels/multilayer.py`, `simulate.py`, `config.py`
(`RoughnessConfig.acf`, validator), `firn.py` (`firn_stack` acf),
`tools/run_basal_clutter.py` (`{source: atm_exponential}`, `(sigma, l, acf)`
triples, rid/meta fork), `tools/clutter_spec.py`, `tools/surface_roughness_b1.py`
(`resolve_exponential`), `tools/run_b26_comparison.py` (`--rough-runs
n:src:acf[:gfx]`), `config/roughness/atm_b1.yaml` (westcoast `*_exp` entries +
`exponential` alternate mapping), `config/experiments/pilot_smoke_exp.yaml`,
`tests/test_roughness_exponential.py`. Scripts for this note:
`claude_notes/roughness_exponential/{mc_exponential.py, layer_asf.py,
tabulate.py, run_queue.sh}`; outputs `outputs/roughness_exponential/`,
`outputs/<line>/pilot_smoke_exp/`, `outputs/b26_comparison/`.

## 1. What was added

D_Phi (Gerekos 2023 Eq. 21) is `e^{-x} sum_m x^m/m! I_m`, x = sigma^2 K^2; the
ACF enters only through I_m (the Fourier transform of rho^m over the facet).
In the area-only (grazing-fix) form, I_m = Lx Ly W_m(k_B), k_B^2 = A0^2 + B0^2:

    gaussian     W_m = pi (l^2/m) exp(-k_B^2 l^2 / (4 m))          (unchanged program)
    exponential  W_m = 2 pi (l/m)^2 [1 + (k_B l/m)^2]^(-3/2)       (C&S 2020 Eq. 6, n = m)

The coherent term exp(-sigma^2 K^2 / 2) and the Poisson weights are
ACF-independent. `acf` is a static flag on `d_phi`, the kernels' jit statics
and the roughness tuple (5th element, only present when non-Gaussian, so every
existing call site traces the same program). The exponential ACF has no closed
form for the finite-facet edge terms (Eqs. 22-24), so `acf: exponential`
requires the area-only D_Phi: `SimConfig` validator (needs `grazing_fix`),
`d_phi` and both kernels raise otherwise. Gaussian bit-identity is gated
(`test_gaussian_path_bit_identical`, 5-tuple `"gaussian"` == 4-tuple kernel
output). `n_terms_for` is unchanged: vs a 1000-term float64 sum the fixed count
is within 9e-10 dB over f = 60-400 MHz, sigma 1-20 cm, l 0.5-5 m, theta
0-89.9 deg (x up to 11, k_B l up to 84 = 2 k l at 400 MHz / l = 5 m).

Plumbing: `physics.surface_roughness` accepts `{sigma_m, corr_length_m, acf}`
and `{source: atm_exponential}` (the table's exponential entry used directly
as (sigma, l, exponential); power-law entries are refused with a message
pointing at `atm_b1`; westcoast gets an `exponential` alternate mapping, see
Sec. 5). Chunk rid appends `_exp` and chunk_meta adds `surf_rough_acf` ONLY for
the exponential ACF; the fixture, booleans and Gaussian pairs keep byte-identical
rid/meta (`test_runner_keys_fork_only_for_exponential`). `run_config.json`
`surface_roughness.passes.<pass>` records `acf`, `sigma_m`, `corr_length_m`,
spectrum id and provenance. B26: `ROUGH_RUNS` entries `(n, src[, acf[, gfx]])`,
keys `firn_N{n}_rough_{src}[_exp][_gfx]`, exponential implies the grazing fix
(recorded in the run meta); legacy `(n, src)` keys/meta unchanged.

## 2. (a) Facet-in-isolation Monte Carlo with exponential surfaces

Script `mc_exponential.py` (300 realizations per point, surfaces by spectral
filtering with sqrt(S), S = (1 + k^2 l^2)^-3/2 for the exponential and
exp(-k^2 l^2/4) for the Gaussian control; on-grid unit variance exactly, the
sub-Nyquist part of the exponential spectrum (1 + (k_N l)^2)^-1/2 <= 1.6 % at
lambda/40 dropped). Facet 4x7 lambda at lambda/40 as in the paper, plus a
16x28 lambda control (the O(1) facet-edge remainder that area-only drops is
12 dB smaller relative to the area term). Geometries: nadir, 30 deg in-plane,
50 deg in-plane. Error = 10 log10 of MC <|Phi|^2> over |<Phi>|^2 +
D_Phi(area_only, acf); columns sigma/lambda = 0.02 / 0.05 / 0.1 / 0.2.
Figures `outputs/roughness_exponential/fig_mc_error_map_{4x7,16x28}.png`,
numbers `mc_metrics.json`.

**Exponential surfaces, 16x28 lambda facet (the ACF physics):**

| geometry | l = 0.5 lam | l = 1 | l = 2 | l = 4 |
|---|---|---|---|---|
| nadir | -0.00 -0.01 -0.01 +0.14 | -0.00 -0.01 -0.05 -0.27 | -0.01 -0.04 -0.16 -0.39 | -0.03 -0.19 -0.66 -0.91 |
| oblique 30 | +0.78 +0.75 +0.56 +0.38 | +0.96 +0.95 +0.78 +0.11 | +2.54 +2.11 +1.03 +0.08 | +6.17 +5.61 +4.14 +1.59 |
| wide 50 | -0.04 -0.18 -0.44 -0.07 | +0.05 +0.09 +0.20 +0.62 | +0.03 +0.07 +0.18 +0.23 | +0.05 +0.14 +0.46 +1.49 |

**Exponential surfaces, 4x7 lambda facet (the paper's setup):**

| geometry | l = 0.5 lam | l = 1 | l = 2 | l = 4 |
|---|---|---|---|---|
| nadir | -0.01 -0.04 -0.15 -0.15 | -0.03 -0.15 -0.51 -0.47 | -0.15 -0.82 -2.02 -1.23 | -0.75 -3.06 -5.37 -3.43 |
| oblique 30 | +1.43 +1.45 +1.22 +0.82 | +2.46 +2.25 +1.51 +0.37 | +5.50 +5.03 +3.59 +1.04 | +6.79 +6.41 +5.20 +2.08 |
| wide 50 | -0.06 +0.15 +0.96 +1.15 | -0.00 +0.31 +1.20 +1.13 | +0.16 +0.79 +2.33 +2.64 | +0.14 +0.81 +2.80 +4.64 |

Gaussian-surface control, same MC machinery: vs the EXACT finite-facet series
(Eqs. 21-24) max |err| 0.54 dB (16x28) / 0.56 dB (4x7) over the whole map --
the generator and the estimator are sound. Vs the area-only Gaussian the
control is within 0.5 dB at nadir / 50 deg for l <= 2 lam but +12..+180 dB at
30 deg for l >= lam: there the Gaussian W_m is exp(-(k_B l)^2/4 m) ~ 1e-9 and
smaller, and the finite facet's edge diffraction (the term the grazing fix
drops on purpose, facet-size dependent) is the whole MC signal. This is a
property of area-only + Gaussian, not of the exponential option; the
exponential W_m never collapses, so the edge remainder stays a few dB.

Reading the exponential map honestly: (i) nadir and wide angle at l <= 2 lam
are within 0.5 dB on the large facet; (ii) at 30 deg the area-only law is LOW
by +0.8 (l = 0.5) to +2.5 dB (l = 2 lam) at small sigma, and by +6 dB at
l = 4 lam -- the largest error of the map. It shrinks with sigma (at
sigma = 0.2 lam: +0.4 / +0.1 / +0.1 / +1.6 dB), so it is not the Kirchhoff
tangent-plane failure of a divergent-slope surface (that would grow with sigma
and be worst at l = 0.5 lam); it is the finite-facet remainder: at 30 deg the
Bragg sample k_B l/m sits on the steep part of (1 + (k_B l/m)^2)^-3/2 and the
facet's sinc-shaped spectral window (width ~ 2 pi/L) integrates the spectrum
over a band, which the point value W_m(k_B) under-represents when the spectrum
curves within the window -- the same effect on the 4x7 facet is 2-3 dB
larger. (iii) The expected Kirchhoff degradation at small l/lambda and wide
angle shows up as the sigma-dependent drift at 50 deg on the small facet
(+1..+1.2 dB at sigma = 0.1-0.2 lam for l <= lam) and as the sigma = 0.2 lam
column generally; on the large facet it is <= 0.6 dB for l <= 2 lam. For the
campaign: sigma/lambda = 0.03 (geikie at 195 MHz) with l/lambda = 3.4 on
27-70 m facets (18-45 lambda), and the firn layers at sigma/lambda_firn ~ 0.05,
l/lambda_firn ~ 3 on ~12-lambda facets, so the relevant cells are the
small-sigma, l = 2-4 lam ones: 0.0-0.2 dB at nadir, +1..+3 dB (large facet) at
30 deg, < 0.5 dB at 50 deg.

**W_m tail, Gaussian vs exponential at the same (sigma, l)** (D_Phi per unit
area, dB re m^2, theta = 30 deg so the Bragg wavelength is 5 / 1.5 / 1 / 0.75 m
at 60 / 195 / 300 / 400 MHz; `fig_wm_tail.png`):

| (sigma, l) | 5 m | 1.5 m | 1.0 m | 0.75 m |
|---|---|---|---|---|
| geikie 5.15 cm / 5.28 m: gauss / exp | -44.9 / -21.4 | -154.4 / -26.6 | -221.0 / -28.2 | -289.0 / -29.5 |
| C&S layer 4 cm / 3 m: gauss / exp | -22.1 / -21.4 | -86.4 / -26.4 | -124.3 / -28.0 | -159.5 / -29.2 |

The m = 1 term carries the exponential sum to within 0.05-2.4 dB (x <= 0.5);
n_terms_for gives 10-11 terms, converged to 9e-10 dB (Sec. 1). The geikie
Gaussian value at 1.5 m (-154 dB) is why B1 had to invent a 1.35 cm / 0.6 m
effective pair; the exponential entry gives -26.6 dB directly (both sit on the
measured -59.3 dB PSD at k_B).

## 3. (b) Haynes-style nadir Fresnel disc

Nadir rough Fresnel disc (h = 8000 lam, 4-lam facets, l = 2 lam, gamma
-0.281; `mc_metrics.json["haynes_disc"]`): the coherent power is the SAME
array for both ACFs (-0.029 dB vs gamma^2/h^2 exp(-(2 k sigma)^2) at every
sigma -- the discretization value of the smooth disc), so the option changes
only the incoherent share:

| sigma/lam | k sigma | total vs Haynes: gauss area-only / exp / gauss exact | incoherent share dB: gauss / exp / gauss exact |
|---|---|---|---|
| 0.02 | 0.13 | -0.03 / -0.03 / -0.03 | -38.0 / -35.1 / -40.8 |
| 0.05 | 0.31 | -0.03 / -0.02 / -0.03 | -29.7 / -27.0 / -32.4 |
| 0.10 | 0.63 | -0.02 / -0.01 / -0.04 | -22.1 / -20.2 / -24.5 |
| 0.15 | 0.94 | -0.00 / +0.00 / -0.05 | -15.2 / -15.0 / -17.0 |
| 0.20 | 1.26 | +0.16 / -0.28 / -0.09 | -6.7 / -8.9 / -7.8 |
| 0.25 | 1.57 | +0.76 / -3.36 / -0.01 | -0.7 / -2.2 / -0.9 |

Total nadir power is ACF-insensitive to 0.03 dB for k sigma <= 1 (the
coherent term dominates; the exponential's incoherent share is +2..+3 dB
above the Gaussian's -- W_1(0) is 2 pi l^2 vs pi l^2, minus the faster
1/m^2 decay). Beyond k sigma ~ 1.3 the incoherent term takes over and the
ACF matters (-3.4 dB vs the Gaussian-ACF Haynes closed form at sigma =
0.25 lam, as it should: Haynes is a Gaussian-ACF result).

## 4. (c) B26 replication vs Culberg & Schroeder

`tools/run_b26_comparison.py --rough-runs 40:mcords,40:ar,20:mcords:exponential,20:mcords:gaussian:1
--only firn_N20_rough_mcords_exp,firn_N20_rough_mcords_gfx` on the 2019 P-3
frame 20190418_01_009 (the validated B26 configuration: 100 traces over 10 km,
+-600 m firn strip, 10.67 m facets, 15 dB/km, point-sampled Kovacs eps, N = 20
equal-placement layers). Every INTERNAL layer carries the C&S Fig. 11 MCoRDS3
(sigma, L) profile (2.97-5.46 cm, 2.57-3.47 m at the 20 depths), once as an
exponential ACF (`_exp`, the family the S-IEM inversion assumed) and once as
the current Gaussian mislabelling with the SAME grazing fix (`_gfx`), so the two
differ only in W_m; the legacy `firn_N40_rough_mcords` (Gaussian, no grazing
fix) and the smooth runs come from the cache. Other runs load as
"cache-stale" only because `--only` marks them so. 4 chunks x 82k facets x 20
layers: 1840.8 s (exp) / 1761.4 s (gfx) simulate wall, vs 1736 s for the
smooth N20 on this kernel era -- the exponential costs +4.5 %, the roughness
itself nothing measurable. Figures `outputs/b26_comparison/{depth_profile,
radargrams_nearsurface}.png`, `metrics.json`; analytic angular scattering
functions `outputs/roughness_exponential/{fig_layer_asf.png, layer_asf.json}`.

Depth-power profile rel own surface peak, band level vs the measured qlook
product (20-70 m: -20.5 dB; 80-120 m: -35.3 dB), profile correlation:

| run | 20-70 m delta | 80-120 m delta | corr (qlook) |
|---|---|---|---|
| surface+bed only | -40.5 | -32.5 | 0.655 |
| firn_N20 smooth | -13.6 | -9.9 | 0.943 |
| firn_N40 smooth | -16.8 | -12.7 | 0.945 |
| firn_N40_rough_mcords (Gaussian, legacy kernels) | -16.1 | -16.0 | 0.945 |
| **firn_N20_rough_mcords_gfx (Gaussian + grazing fix)** | **-15.2** | **-14.8** | 0.932 |
| **firn_N20_rough_mcords_exp (exponential + grazing fix)** | **-15.9** | **-13.8** | 0.921 |
| firn_N20_h1eff (effective contrast, smooth; the adopted standard) | -7.0 | -3.5 | 0.965 |

Plateau level and depth trend: exponential vs Gaussian on the same stack is
**-0.7 dB in 20-70 m and +1.0 dB in 80-120 m** -- i.e. the ACF of the layer
roughness moves the plateau by less than 1 dB either way, and the profile
correlation drops slightly (0.921 vs 0.932; the smooth N20 is 0.943). Both
rough N20 runs sit 1.6-2.3 dB BELOW the smooth N20 in 20-70 m, which is the
grazing-fix taper on the layers' specular lobes plus the coherent attenuation
exp(-sigma^2 K^2/2) (x = (2 k_f sigma)^2 = 0.1-0.3 -> -0.4..-1.3 dB), not the
ACF; against the smooth N40 reference the tool reports +0.9 (exp) / +1.6 (gfx)
dB (`roughness_band_delta`). Mid-column clutter (the sim's own surface arm) is
unchanged, the surface stays smooth in these runs by construction. The known
deficit stands: the point-sampled stacks are 14-17 dB below the measured
plateau and the effective-contrast standard 7 dB (the "~6 dB plateau deficit"
of `claude_notes/firn_bandwidth_haps_2026-08-26.md`, measured there as -6 dB
on 195/30 with the H1 stack) -- neither ACF of the C&S layer roughness
touches it, confirming with the correct family the 2026-07-28 verdict that
measured layer roughness adds < 1 dB to the plateau.

Doppler-spectrum shape: the cached runs hold 100 traces at ~100 m spacing,
which cannot resolve a Doppler spectrum within +-7 deg (angular aliasing period
lambda/(2 x 100 m) = 0.4 deg), and a dense along-track re-simulation (C&S: 1.4 km
aperture at 0.41 m, ~3400 traces, 100-aperture average) is ~35x the 100-trace
cost per layer -- not cheap, not done. Instead `layer_asf.py` evaluates the
angular scattering function the kernel implies per layer analytically: the
incoherent sigma0(theta) of the area-only law -- (k_f^2/pi) cos^2 e^{-x} sum
x^m/m! W_m(2 k_f sin theta), which IS the Kirchhoff term of C&S Eq. 5 with
their Eq. 6 W^n to cos^4(theta) (0.13 dB at 7 deg) -- relative to their
specular-disc peak sigma0_c (Eq. 7), at 195 MHz, h = 470 m, in-firn angle:

| depth | sigma / L | k_f sigma | ACF | 0 deg | 1 deg | 2 deg | 4 deg | 7 deg |
|---|---|---|---|---|---|---|---|---|
| 10 m | 5.2 cm / 2.76 m | 0.30 | gauss / exp | -19.3 / -16.5 | -19.7 / -18.2 | -20.6 / -21.5 | -24.4 / -27.5 | -33.8 / -33.7 |
| 30 m | 4.4 / 3.31 | 0.28 | gauss / exp | -18.1 / -15.3 | -18.7 / -18.0 | -20.4 / -22.4 | -26.8 / -29.3 | -41.4 / -35.9 |
| 55 m | 2.4 / 3.26 | 0.17 | gauss / exp | -22.9 / -20.0 | -23.6 / -23.0 | -25.5 / -27.8 | -32.9 / -35.0 | -51.2 / -41.9 |
| 80 m | 4.1 / 2.94 | 0.29 | gauss / exp | -18.8 / -16.0 | -19.3 / -18.6 | -20.9 / -22.9 | -27.2 / -29.8 | -41.5 / -36.4 |

With the Fig. 11 parameters the exponential layers reproduce C&S's
observation (Sec. II: the angular scattering function falls to >= 25 dB below
the peak within 7 deg on MCoRDS3, more specular with depth: the 55 m layer is
-35 dB at 4 deg) -- as they must, being the same law the inversion fitted.
The Gaussian mislabelling is within 3 dB of the exponential for |theta| <= 4 deg
at these sigma/L (the cusp region of the ACF is not probed by k_B L/m < 2) and
diverges only beyond ~5 deg (-41..-51 vs -36..-42 dB at 7 deg), which is why
the two B26 runs differ by < 1 dB: the plateau is dominated by the coherent
term and near-nadir incoherent power, where the ACF does not matter. The
exponential is nevertheless the right family for any wider-angle use of the
layers (HAPS geometries, surface-clutter-like layer returns at the bed delay).

## 5. (d) Real passes: geikie and westcoast pilots

`pilot_smoke_exp` (identical to `pilot_smoke_b1` except `surface_roughness:
{source: atm_exponential}`), segment pilot. Fixture / B1 columns replayed from
the B1 branch's outputs (`claude_notes/roughness_b1_2026-08-26.md`). Figures
`outputs/<line>/pilot_smoke_exp/{radargrams,decomposition,decomposition_trace,bed_tail}.png`;
`claude_notes/roughness_exponential/tabulate.py` prints the tables.

### greenland_geikie01_transit (ATM best family IS exponential: 5.15 cm / 5.28 m)

| pass | AGL m | roughness | measured midcol | fixture (err) | B1 (err) | exponential (err) |
|---|---|---|---|---|---|---|
| low | 465 | 5.15 cm / 5.28 m exp | -44.5 | -119.8 (-75.3) | -69.6 (-25.1) | **-68.0 (-23.5)** |
| high | 2483 | same | -41.7 | -87.3 (-45.6) | -60.3 (-18.6) | **-58.9 (-17.2)** |
| high - low | | | +2.8 | +32.5 (+29.8) | +9.3 (+6.5) | **+9.1 (+6.3)** |

Bed level rel surface / bed-tail excess at +1, +2, +3 us / surface alignment
p90: low -105.5 / +6.4, +0.7, -3.5 / 0.81 (B1 -104.0 / +8.4, +2.5, -2.4 /
0.92; measured -107.1); high **-90.2 / -6.9, -6.1, -6.2 / 0.82** (B1 -97.0 /
-14.5 x3 / 0.83; fixture -101.4 / -19.7, -25.4, -26.5; measured -84.0).

Mid-column: the measured exponential surface lands 1.4-1.6 dB above the B1
tangent Gaussian and 17-24 dB below the data, with the same +6 dB altitude-
trend error. Not a surprise: B1 was built to equal this same PSD at the 30 deg
Bragg point, and the mid-column of both passes is set by 30-70 deg surface
angles where the exponential's k^-3 tail is only a few dB above the tangent
Gaussian within +-0.3 octave (B1 residual table) -- the 465 m pass's 60-70 deg
angles gain the most (+1.6 dB). What the exponential DOES change is the
high-pass bed window: the bed-delay clutter comes from ~30-40 deg, and the bed
level moves 7 dB towards measured (-97.0 -> -90.2 vs -84.0) with the tail
excess halved (-14.5 -> -6 dB). So on the line where the exponential is the
measured family, the remaining mid-column deficit (-17..-24 dB) is NOT the
ACF form; it is what B1 already left: along-track aliasing of the 15 m posting
at the clutter angle, and the absence of firn layers.

### greenland_westcoast (power-law surface; exponential = least-squares fit through the 5 / 1.5 / 1 / 0.75 m Bragg medians)

| pass | AGL m | exp. fit sigma / l | measured midcol | fixture (err) | B1 (err) | exponential (err) |
|---|---|---|---|---|---|---|
| p3_2016 (200 MHz) | 484 | 4.78 cm / 10.0 m (l capped; beta 3.29 ~ k^-3) | -48.9 | -149.6 (-100.6) | -80.9 (-32.0) | **-75.3 (-26.4)** |
| p3_2017 | 500 | 3.33 / 1.04 | -59.4 | -117.5 (-58.1) | -69.7 (-10.2) | **-64.3 (-4.9)** |
| p3_2019 | 476 | 3.77 / 2.59 | -59.0 | -118.5 (-59.5) | -72.2 (-13.2) | **-67.0 (-8.0)** |

Bed level / tail excess (+1, +2, +3 us) / alignment: p3_2016 -94.5 / -21.6,
-24.6, -25.0 / 0.87; p3_2017 -90.3 / -0.7, +2.7, +2.2 / 0.67; p3_2019 -91.4 /
-17.7, -7.4, -6.5 / 1.37 (B1: -94.2, -90.7, -92.1; tails -26.2 / -2.4 / -10.4
at +2 us; alignment identical). The exponential fit is within +-0.5 dB of the
beta = 2.5-3.3 power law at all four Bragg points (the power law's k^-3 tail IS
the exponential's large-k limit), where the tangent Gaussian was 3-10 dB low
half to one octave from k_B -- and that is what the 500 m AGL mid-column
(60-70 deg, Lambda ~ 0.8 m) sees: +5.6 / +5.4 / +5.2 dB over B1, the 2017/2019
gap closes to -5 / -8 dB. The 2016 pass keeps its 26 dB excess (a radiometry /
system issue in the product, not roughness; see the B1 note). Bed levels move
< 0.7 dB, the post-bed tails rise 2-5 dB (the surface arm at the bed delay).
So on a self-affine surface an exponential fitted over the Bragg band gets
the low-altitude mid-column to within 5-8 dB -- as close as the tangent
Gaussian got at 2.5 km altitude on geikie; but it is a fit, not the family
(block-wise the exponential is best in only 16-35 % of westcoast blocks), and
a fixed (sigma, l) cannot follow the power law outside 0.75-5 m.

## 6. Runtimes and commands

Wall per invocation (this box, warm cache; logs in
`claude_notes/roughness_exponential/logs/`): MC script 1606 s (4x7 facet 289 s,
16x28 facet 1300 s; 300 realizations x 4 l x 2 ACFs); layer ASF 5 s;
pilot_smoke_exp geikie 932 s (per pass 544 / 379 s; B1 713 / 503, fixture
412 / 287 -- the exponential needs the SAME 10-11 D_Phi terms as the fixture,
B1's l = 0.6 m needed more), westcoast 710 s (223 / 246 / 230 s per pass; B1
266-297, fixture 150-166); B26 firn_N20_rough_mcords_exp 1841 s simulate,
firn_N20_rough_mcords_gfx 1761 s (+ report; the second was relaunched detached
after the queue process was killed between the two runs). Tests:
`uv run pytest -q tests/test_roughness_exponential.py tests/test_roughness.py
tests/test_roughness_kernel.py tests/test_grazing_fix.py
tests/test_surface_roughness_b1.py tests/test_firn.py tests/test_experiment_specs.py
tests/test_instruments.py tests/test_basal_hypotheses.py` (126 passed, 1 skipped)
plus test_config/coherent_kernel/multilayer*/layered_config/firn_investigation/
firn_plateau/kernel (35 passed).

    uv run python claude_notes/roughness_exponential/mc_exponential.py [n_real]   # (a), (b), W_m tail, convergence
    uv run python claude_notes/roughness_exponential/layer_asf.py                 # (c) analytic layer ASF
    bash claude_notes/roughness_exponential/run_queue.sh                          # geikie + westcoast pilots, B26 pair
    bash claude_notes/roughness_exponential/run_b26_gfx.sh                        # detached B26 re-launch
    uv run python claude_notes/roughness_exponential/tabulate.py                  # (d) tables
    uv run python tools/run_b26_comparison.py --report-only                       # B26 report from cache

Not done / caveats: no `full` segments (pilot only, as asked); no simulated
Doppler spectrum (above); getz/david have no exponential entry (power law,
refused by design); the westcoast 2016 exponential has l capped at 10 m
(beta 3.3 is the l -> inf limit); the B26 rough runs keep the point-sampled
eps stack, so their absolute plateau is the pre-H1 one.
