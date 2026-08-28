# Radar Statistical Reconnaissance (RSR) for the simulator's surface roughness: feasibility (2026-08-27)

Question: would RSR (Grima et al. 2012 Icarus, 2014 PSS) be a better route than laser
altimetry (ATM / ICESat-2) to the sub-facet roughness parameters the simulator needs?

Code: `claude_notes/rsr_roughness/rsr_prototype.py`, `rsr_addendum.py`
(`uv run`), log `prototype_run.log`. Outputs: `outputs/rsr_roughness/`
(`rsr_prototype_20170510_03_013.{json,png}`, `rsr_prototype_pilot_hist.png`).
Pulled: `outputs/cache/Data_20170510_03_013_qlook.mat` (2.2 MB) and
`Data_20170510_03_013_source.mat` (26 MB, the CSARP_standard source with its
param structs). Nothing committed; simulator source untouched.

**Verdict up front: (c)/(b) — not a replacement, and not a complement for the
off-nadir Bragg band.** On the public OPR products the coherent/incoherent split is
not identifiable (the surface-peak PDF is dominated by along-track mean variation and
roll, not speckle), and even with raw single-look data a nadir RSR at 195 MHz and
500 m AGL only constrains the surface PSD at wavelengths ≥ 5 m — the 60 MHz Bragg
scale, not the 1.5 m scale that sets 195 MHz wide-angle clutter. What RSR could give
is one band-limited σ (5–~100 m) per ~km, which ATM already gives with its own noise
budget. The radar-side complement for the wide-angle PSD is the along-track Doppler
spectrum (C&S 2020 §II), not RSR.

## 1. What RSR measures and inverts

**Observable.** Over a window of N_w consecutive traces (Grima uses 1000, sampled
every 250) the surface-echo peak *amplitudes* are histogrammed and fitted with a
Homodyned-K PDF (Rice amplitude with a Gamma-distributed incoherent power,
parameters a, s, μ; `cgrima/rsr` `pdf.py`, Destrempes & Cloutier 2010 compound
form). The fit yields Pc = a² (coherent power: the part with a fixed phasor across
the window) and Pn = 2 s² μ (incoherent: the zero-mean, speckle-like part), plus μ,
the "number of scatterers" shape parameter (μ → ∞ recovers Rice). Pc and Pn are in
the product's power units; a Rayleigh, Rice, K or HK law can be chosen.

**Inversion.** The package's only roughness inversion (`invert.py::spm`) is the
small-perturbation model in the *large-correlation-length* limit:

    Pc/Pn = exp(−(2kσ)²) / (2kσ)²,   then  |Γ|² = Pc · exp((2kσ)²)  →  ε from Γ.

That is: σ from the ratio alone (no l), and permittivity from the calibrated Pc
once σ is known. Grima 2014 GRL (Thwaites firn density) and Grima 2016 GRL (McMurdo
brine) use exactly this chain: Pc → ε → density; Chan et al. 2022/23 (Devon)
use Pc with a thin-layer reflectivity model. I could not access the 2014 PSS text
(paywalled) to confirm what l-dependent Kirchhoff/Gaussian-ACF form it used for the
landing-risk map; my recollection is that it produced (σ, l) *ranges* from a
Gaussian-ACF Kirchhoff model by assuming one to bound the other — **unverified**.
Adhikari & Li 2018 (IEEE RadarConf, Petermann, MCoRDS) fitted amplitude
distributions for surface *and bed* roughness; abstract only, method unverified.

**Derivation of the nadir ratio with a general PSD** (so the family question can be
answered). Mirror-image coherent return from a slightly rough plane at height h,
first-order SPM incoherent return integrated over the resolution cell (Ulaby SPM
σ⁰ = 8k⁴cos⁴θ|α|²W with W = 2πS/σ² in the ATM note's convention
S(k) = σ²l²/(4π)·exp(−k²l²/4), σ² = ∫∫S d²k):

    Pn/Pc = 16 k⁴ e^{(2kσ_c)²} (1/h²) ∫_cell S(2k r/h) dA
          = (2k σ_cell)² · e^{(2kσ_c)²},    σ_cell² = ∫_{|k_B| ≤ k_max} S d²k_B .

When the cell contains all the diffuse power (k_max ≫ 1/l: Gaussian, l not tiny)
σ_cell = σ and this is the rsr formula exactly. In general **RSR's ratio measures the
band-limited roughness variance in the wavenumber band the cell maps to Bragg
angles**, k_B = 2k sinθ with θ ≤ θ_max of the cell — regardless of ACF family. The
family only enters when you convert σ_cell into (σ, l) or (A, β), which is why the
Gaussian assumption looks harmless in RSR papers: it never gets tested there.

**Sensitivity scales for the 2017 P-3 geometry** (h ≈ 507 m, B = 30 MHz → 5 m range
resolution, pulse-limited radius r_pl = √(2h·δr) = 71 m, θ_max = 8.0°; SAR σ_x =
2.5 m, then 11 lines averaged = 27.5 m strip):

| term | scale it senses |
|---|---|
| coherent loss e^{−(2kσ_c)²} | σ_c = total height variance inside the cell (all scales < ~70 m, above the sub-metre) — for a self-affine surface dominated by the *largest* in-cell scales |
| incoherent Pn (cross-track) | k_B ≤ 2k sin 8° → Λ ≥ 5.5 m |
| incoherent Pn (along-track, SAR) | k_B ≤ π/δx → Λ ≥ 5.0 m |
| Pn overall | S(k) on Λ = 5–140 m, i.e. the 60 MHz-at-30° Bragg scale and the facet-tilt band |

Nothing in a nadir RSR at 195 MHz touches Λ = 1.5 m (the 195 MHz, 30° Bragg
wavelength), 1.0 m (300 MHz) or 0.75 m (400 MHz). The only way to push θ_max out is
a wider bandwidth (θ_max ∝ √(δr/h)) or a higher platform — 14 km HAPS altitude with
5 m δr gives θ_max = 1.5°, i.e. *narrower*; RSR from HAPS/orbit is even more nadir.

**Assumptions and degeneracies.** (i) stationarity over the window (15 km at 15 m
traces; Grima's HiCARS windows were ~1000 raw traces ≈ a few km); (ii) single-look
speckle statistics — the HK/Rice PDFs are for one look; N-look averaging changes the
law (noncentral-χ² with 2N dof) and, if ignored, is read as extra coherence;
(iii) permittivity/gain: Pc is absolute, so ε needs a calibrated system and vice
versa — with the ratio only, σ_cell comes out independent of |Γ|; (iv) the ratio is a
single number per window, so (σ, l) or (A, β) are degenerate: one of them must be
assumed; (v) SPM needs 2kσ_c ≲ 0.5–1; at 195 MHz that is σ_c ≲ 8–16 cm, and the ATM
5–64 m σ here is 5–17 cm (p5–p95), so the site sits at the edge of validity —
beyond it the ratio measures in-cell RMS slope (geometric optics), not height;
(vi) anything that modulates the mean return along the window (roll of an unsteered
array, facet tilt at the km scale, firn permittivity) is absorbed into the
"incoherent" part or into μ.

## 2. RSR versus ATM for our need

| | ATM L1B (done) | RSR (nadir surface echo) | Doppler spectrum (C&S §II) |
|---|---|---|---|
| samples the Bragg band at 30°? | yes, 0.9–50 m lags (1.5 m by < 1 octave extrapolation) | **no**: Λ ≥ 5 m at 195 MHz, 500 m AGL | yes: along-track angles out to the antenna/aliasing limit, Λ = λ/(2 sinθ) directly |
| form (Gaussian / exp / power law)? | yes, D(r) over two decades | no: one number per window | partly: S(k_B) along one axis over the Doppler band |
| σ | band-limited, own noise budget | σ_cell (5–~100 m band) if single-look and calibrated | no |
| own geometry / wavelength | no (needs the surface to be the same) | yes | yes |
| Greenland interior coverage | ATM1B where OIB flew (thin), ILATM2 blind | every OPR frame, if raw data | every OPR frame with raw/SAR-input data |

Two structural points:

1. RSR is nadir. The incoherent term comes from within θ_max = √(2δr/h); the
   wide-angle clutter question needs S(k) at 30° or beyond. The Doppler spectrum of
   the raw (pre-SAR) surface echo *is* the along-track angular scattering function:
   it samples k_B = 2k sinθ out to whatever angle the along-track antenna pattern
   and PRF allow (P-3 at 12 kHz PRF and 130 m/s: unaliased to sinθ = λ·PRF/(4v) —
   far beyond 30°; the element pattern is the limit). That is the radar-side
   measurement of the Bragg-band form, and it is independent of the coherent split.
2. The Gaussian ACF does not "reintroduce" the form problem inside RSR because RSR
   never resolves the form — it produces a band-limited variance. The problem
   re-enters the moment σ_cell is written into a Gaussian (σ, l) fixture and
   extrapolated to k_B(30°): the same −140 dB tail ATM exposed. An exponential or
   power-law inversion is trivial to write (the script does both:
   `sigma_gauss_band`, `sigma_pl_band`) but is under-determined by the same single
   number; the family has to come from elsewhere (ATM or Doppler).

## 3. Practical: what the data give

**RSSNR store.** Per decimated trace (~1.4 km sampling on this line, 32/42 traces
on _013, 42/42 on _014; `config/lines/greenland_westcoast.yaml`):
`surface_power_dB`, `bed_power_dB`, twtt/elevation picks, `required_surface_snr_dB`,
noise levels, `qc_pass`. These are *peak powers of already-multilooked traces at
1–2 km spacing*: no amplitude distribution, no coherent/incoherent split, no
per-trace series. An RSR-style analysis cannot be run on it; at best it gives the
km-scale mean surface return, which is the same "trend" component the prototype
finds dominating the PDF. (The store's surface_power series would be a fine
covariate for the *trend* — roll- and slope-driven — but not for roughness.)

**Public OPR products** (frame 20170510_03_013, from its own param structs):

| product | processing | traces / spacing | looks |
|---|---|---|---|
| CSARP_qlook | 50 presums, pulse compression, `inc_ave` 10, `decimate_factor` 50 | 286 / 174 m | 10 incoherent |
| CSARP_standard | f-k SAR, σ_x 2.5 m, `combine.rline_rng` −5..5 (11 lines incoherent), `dline` 6 | 3334 / 14.9 m | 11 incoherent, adjacent outputs share 5 of 11 lines |
| CSARP_mvdr / post | array-processed / picked versions of the same | — | — |

No public single-look product exists. Raw-frame RSR needs the pre-combine SAR
output (CReSIS `CSARP_out`, one look per subaperture at 2.5 m) or the raw records
reprocessed with `inc_ave = 1`; both mean running the CReSIS toolbox or an
equivalent SAR/pulse-compression chain ourselves. The same holds for the Doppler
method (needs raw or at least pre-SAR pulse-compressed data). Frames that qualify
*in principle* (raw data is public for all OPR seasons): westcoast 2016/2017/2019,
geikie 2014/2017, getz 2016 (DC-8); the UTIG/AWI David frames are HiCARS-family
products whose level-1 availability I did not check.

**Calibration.** The ratio Pn/Pc is gain-free; σ_cell follows from it without a
permittivity assumption. Pc in absolute terms needs the system gain, or
equivalently a permittivity pin (Grima pins σ from the ratio, then reads ε; C&S
pin ε and read gain) — the same γ_surface = −10 dB choice this repo already makes.

## 4. Prototype: frame 20170510_03_013 (westcoast, 195/30 MHz, s 0–50 km)

Per-trace surface peak = max of `Data` within ±6 bins of the OPR `Surface` pick
(peak sits 0–2 bins past the pick), corrected by (h/h₀)² for the mirror-return
range dependence (h varies < 3 %). Windows of 1000 traces (14.9 km), 50 % overlap,
plus the pilot slice 2678–3334 (s 40–50 km, 656 traces). Fits: Rice and HK on
amplitude (the RSR convention, one look), and the **N-look Rice** on power
(noncentral-χ², 2N dof) with N free and with N fixed at 11 (the product's nominal
looks), 5.5 and 1. Inversion: `(2kσ)² e^{(2kσ)²} = Pn/Pc` (rsr `spm`).

| window (s km) | CV of P | lag-1 corr of log P | HK Pn/Pc (dB), μ | N free | Pn/Pc (dB) N free | Pn/Pc (dB) N = 11 | Δnll N=11 vs free | σ_cell cm: HK / N free / N = 11 |
|---|---|---|---|---|---|---|---|---|
| 0–15 | 0.72 | 0.87 | −3.0, 19.5 | 0.98 | −3.3 | +75.8 | 2337 | 7.3 / 7.1 / 47 |
| 7–22 | 0.82 | 0.88 | −1.5, 17.4 | 1.39 | +29.2 | +77.3 | 2767 | 8.2 / 28 / 48 |
| 15–30 | 0.75 | 0.92 | −2.1, 19.7 | 0.98 | −2.4 | +75.8 | 2734 | 7.8 / 7.7 / 47 |
| 22–37 | 0.71 | 0.83 | −4.7, 2.8 | 2.15 | +60.5 | +74.1 | 1369 | 6.3 / 41 / 46 |
| 30–45 | 0.74 | 0.73 | −5.2, 1.8 | 2.49 | +65.3 | +75.1 | 1051 | 6.0 / 43 / 47 |
| **pilot 40–50** | 0.57 | 0.72 | **−7.0**, 18.8 | 3.07 | +12.3 | +72.3 | 459 | **5.0** / 18 / 46 |
| qlook, whole frame (10 looks known) | — | — | — | 0.99 | −0.1 | +75.0 (N = 10) | — | — / — / 47 |

Reading:

- **The product's PDF is not a speckle PDF.** With the true 11 looks imposed the
  best fit is a pure incoherent, nearly Gaussian bump (Pn/Pc = +75 dB, σ_cell =
  47 cm — nonsense) that misses the histogram completely (Δnll ≈ 10³,
  `rsr_prototype_pilot_hist.png`). Freed, N comes back as 1–3: the spread (CV
  0.6–0.8) is 2–3× what 11 looks of any Rice mixture can produce. The qlook (10
  known looks) gives the same: fitted N ≈ 1.
- The extra spread is along-track variation of the *mean* return, not speckle:
  log-power autocorrelation is 0.87 at lag 1 (15 m) and 0.46 at lag 10 (150 m),
  where pure 11-line/6-decimation speckle would give 0.45 at lag 1 and 0 beyond.
  Detrending with a running median leaves a residual CV of 0.35 at 74 m
  (compatible with ≥ 50 % incoherent power if the effective looks are ~6, and
  with anything if 11), rising to 0.73 at 1.2 km; the trend carries 2–3.7 dB
  std. |roll| correlates with the surface peak at r = −0.56 (roll p5–p95
  −9°…+8°, unsteered 7-element array), so a good part of the "incoherent" spread
  is antenna gain modulation, the rest km-scale slope/facies. This is degeneracy
  (vi) of §1 in action.
- The 1-look HK/Rice fits — the textbook RSR — therefore return whatever the
  trend variance dictates: Pn/Pc = −1.5…−7 dB, σ_cell = 5–8 cm, with μ jumping
  between 2 and 20 from window to window. These numbers are not a coherent split;
  they are a re-labelling of the mean's variability. **On CSARP_standard/qlook
  the RSR decomposition is not identifiable.**
- Consequently the σ_cell columns are not measurements. For the record, the
  1-look σ (5–8 cm) lands inside the ATM 5–64 m band σ range (median 8.8 cm,
  p5–p95 4.9–17 cm over the line; near the pilot: 18–20 cm at s 38–41, 8–9 cm at
  s 41–43, 6 cm at s 53–55; ATM blocks s 43–53 are missing from the 2017 CSV, so
  the pilot window is only half covered) — but so would almost any number between
  4 and 20 cm, and the N-free fits scatter 7–43 cm.

**What the ATM says RSR would face here.** The SPM prediction of Pn/Pc from the ATM
blocks (σ_cell = 5–64 m octaves, σ_c = 1–64 m) is median −0.2 dB, p5–p95 −7…+12 dB:
at 195 MHz this surface sits right at the coherent/incoherent transition
(2kσ_c = 0.65–1.5). That is the regime in which a clean single-look RSR would be
*most* informative on σ_cell — and also where SPM is marginal and roll/tilt trend
removal must be done first.

**What can still be derived from the public products:** the km-scale surface-return
trend (a roll-corrected mean-power series — useful as a check on the simulator's
facet-tilt and γ_surface handling, and already essentially in the RSSNR store), and
a weak lower bound on the incoherent fraction from the ≤ 100 m residual if the
effective look count of the product were pinned (it is not: 11 lines of 2.5 m
pixels at a SAR resolution that is itself ~2–3 m are 4–11 effective looks).

## 5. Verdict and plan

**(c) for the Bragg-band question; (b) at most for nadir σ.** RSR does not
replace ATM: it cannot see Λ < 5 m at any VHF sounder geometry we care about, it
returns one band-limited variance per window with the family unresolved, and it
needs single-look, roll-corrected, trend-removed data that the public OPR products
are not. Where it would add something (interior Greenland, where ATM1B is thin and
ILATM2 blind) it would add a 5–100 m-band σ per km — the persistent topographic
part the ATM study showed is *not* the seasonally re-forming Bragg-band part.

Radar-side complement that *does* address the need: the **along-track Doppler
spectrum** of the raw surface echo (C&S 2020 §II). It measures S(2k sinθ) along
track out to tens of degrees at the radar's wavelength, tests the power-law vs
exponential form directly, and needs the same raw-data reprocessing RSR would.
Effort: raw record access + pulse compression + Doppler analysis per frame,
~2–3 days for one frame with the CReSIS toolbox or a minimal own chain; then
compare S(k_B) at Λ = 1.5–5 m with the ATM pooled power law (−57 dB at 1.5 m,
−44 dB at 5 m, 2017). If it is done, a by-product single-look nadir series makes
an RSR σ_cell essentially free (~½ day) and gives the roll-corrected coherent
fraction as a test of the simulator's nadir surface return — worth having as a
validation, not as the roughness input.

What it changes in the simulator's roughness representation: nothing directly.
The representation question (path B: power-law/tabulated ACF, or B1 effective
Gaussian at each k_B) stays with the ATM result; the Doppler test is the check
that the ATM family transfers to the radar's own scattering; RSR is a nadir
consistency check on σ and γ_surface.

## Sources

- Grima, Kofman, Herique, Orosei, Seu (2012), *Icarus*, doi 10.1016/j.icarus.2012.04.017 — RSR on SHARAD (Mars).
- Grima, Schroeder, Blankenship, Young (2014), *PSS* 103, 191–204, doi 10.1016/j.pss.2014.07.018 — HiCARS Thwaites, landing-zone concept (text not accessed).
- Grima, Blankenship, Young, Schroeder (2014), *GRL*, doi 10.1002/2014GL061635 — Thwaites firn density from Pc (abstract verified).
- Grima, Greenbaum, Lopez Garcia, Soderlund et al. (2016), *GRL*, doi 10.1002/2016GL069524 — McMurdo brine (abstract verified).
- Rutishauser et al. (2016), Devon Ice Cap refrozen meltwater from RSR (cited in Chan et al.; not accessed).
- Chan, Grima, Rutishauser, Young et al. (2022/2023), *TC* 17, 1839, doi 10.5194/tc-2022-181 — Devon dual-frequency coherent component (abstract verified).
- Adhikari & Li (2018), IEEE RadarConf, doi 10.1109/RADAR.2018.8378707 — Petermann surface/bed roughness from MCoRDS amplitude distributions (abstract only).
- `github.com/cgrima/rsr` (master, `src/rsr/{pdf,invert,fit}.py`) — HK/Rice/K PDFs, `spm` inversion (verified from source).
- CReSIS param structs in `Data_20170510_03_013.mat` (standard and qlook) — processing chain numbers above.
