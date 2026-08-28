# When does higher frequency mean less surface clutter at the bed — and when more?

Model and figures: `claude_notes/freq_regimes/freq_regimes.py` →
`outputs/freq_regimes/freq_regimes.png`, `freq_crossovers.png`.

## The quantity

Surface clutter that competes with the bed return arrives from the off-nadir
angle θ_c whose two-way delay equals the bed's:

    cos θ_c = h / (h + n·d)        (h altitude, d ice thickness, n = 1.78)

| geometry | θ_c |
|---|---|
| airborne 500 m, 1 km ice | 77° |
| HAPS 14 km altitude (12.5 km AGL), 1 km ice | 29° |
| HAPS 14 km altitude, 3 km ice | 46° |
| HAPS 20 km altitude (18.5 km AGL), 1 km ice | 24° |
| HAPS 20 km altitude, 3 km ice | 39° |
| orbital 600 km, 2 km ice | 6° |

(The HAPS rows are at the stated platform altitude; "1 km" / "3 km" are ice
thicknesses. Going from 14 to 20 km moves θ_c inward by ~5–7°, i.e. toward
the orbital regime: f_A rises from 62 to 73 MHz (1 km ice) and the Gaussian
f* from 47 to 55 MHz — the array needs slightly more frequency before it helps.)

Clutter-to-bed ratio, keeping only the frequency-dependent factors:

    C(f) ∝ k⁴ cos⁴θ_c · S(2k sin θ_c) · AF²(θ_c; f) / B(f)

- k⁴ · S(k_B): first-order diffuse scattering (SPM / Kirchhoff m = 1 term);
  S is the 2-D surface height PSD and k_B = 2k sin θ_c the Bragg wavenumber.
  The k⁴ is the Rayleigh-like prefactor that makes any fixed roughness
  scatter more at shorter wavelength; S(k_B) is how much roughness exists at
  the scale λ/(2 sin θ_c) that scatters to θ_c.
- AF²: two-way antenna (array) gain at θ_c relative to nadir.
- B: the bed's own frequency dependence (specular bed: none; rough bed:
  coherent loss exp(−(2 k_ice σ_b)²)).
Spreading, attenuation, illuminated area per range bin are
frequency-independent to first order and cancel.

## Regime 1 — the surface scattering law

Combine k⁴ with each PSD family (all normalised to the same amplitude at
5 m wavelength, which is what the ATM data show is common across lines):

| family | S(k_B) | k⁴ S(k_B) | trend with f |
|---|---|---|---|
| Gaussian ACF, length l | exp(−k_B² l²/4) | rises as k⁴, then collapses once k_B l > 2 | **low f: more clutter with f; high f: less** — crossover at k l sin θ_c = √2, i.e. f* = c√2 / (2π l sin θ_c) |
| exponential ACF, length l | (1 + k_B² l²)^(−3/2) | ∝ k⁴ → ∝ k at k_B l ≫ 1 | more clutter with f, ~3 dB per doubling |
| power law, Hurst H (β = 2H+2) | k_B^(−β) | ∝ k^(4−β) = k^(2−2H) | more clutter with f, ~(2−2H)·3 dB per doubling (H = 0.4: +3.6 dB) |

Gaussian crossovers (left panel of `freq_crossovers.png`): l = 3 m gives
f* = 46 MHz at θ_c = 29° (HAPS/1 km), 32 MHz at 45°, 23 MHz at 77°, but
208 MHz at 6° (orbital). So with a Gaussian surface a HAPS is above its
crossover at any VHF frequency and higher f always helps; an orbital
sounder is below it until UHF and lower f helps — Culberg & Schroeder's
"HF/low-VHF for orbital" conclusion and the design study's "300 MHz for
HAPS" are the same Gaussian law on opposite sides of f*.

With an exponential or power-law surface there is no crossover in the
scattering law: **higher frequency always scatters more to any fixed
angle**, only mildly (k¹ to k^1.3). The ATM analysis found the coastal
Greenland and Getz surfaces are power-law (H 0.35–0.45) and geikie is
exponential, so the collapse the Gaussian predicts above f* does not
happen on real surfaces; the strong "higher f wins" result at HAPS
altitude was the Gaussian tail, not physics.

## Regime 2 — the antenna

What frequency does to the antenna depends on what is held fixed:

- **Fixed beamwidth** (elements at λ/2, so the aperture scales with λ):
  AF²(θ_c) is frequency-independent. The antenna neither helps nor hurts
  with frequency; the scattering law alone decides (top row of the figure).
- **Fixed physical span W** (a platform constraint): the beam narrows as
  λ/W. While θ_c is inside the main lobe (sin θ_c < λ/W, i.e.
  f < f_A = c/(W sin θ_c)) the array does nothing to bed-delay clutter.
  Above f_A the clutter angle falls into the sidelobes and AF² drops as
  roughly (λ/(W sin θ_c))^(2p) two-way — p = 1 for a uniform taper, ~3 for
  Hann — i.e. k^(−2) to k^(−6). That beats the scattering law's k^(+1..+1.3)
  for any realistic surface, so **above f_A, higher f wins regardless of
  the surface**, until element spacing exceeds ~λ and grating lobes return
  the clutter (16 elements on 10 m: above ~450 MHz).

f_A (right panel of `freq_crossovers.png`) for W = 10 m: 62 MHz at 29°,
42 MHz at 45°, 31 MHz at 77°, but 278 MHz at 6°. For an orbital sounder a
10 m array only starts to help at UHF; a 40 m array brings f_A to 70 MHz.

## Putting the two together (`freq_regimes.png`)

Top row (fixed beam) shows regime 1 alone: Gaussian curves peak and
collapse at f*; exponential/power-law curves rise ~3–4 dB per doubling
everywhere, including at HAPS altitude.

Bottom row (16 Hann elements on 10 m) shows regime 2 layered on top:
- Airborne 500 m (θ_c = 77°): f_A ≈ 30 MHz, so at every frequency of
  interest the clutter is in the sidelobes and rises/falls with the
  sidelobe envelope — the antenna dominates; frequency matters through
  sidelobe level, not the surface.
- HAPS 14 km with 1 km (29°) and 3 km (46°) ice, and 20 km with 1 km (24°) and
  3 km (39°) ice: flat to ~50–75 MHz, then the array takes
  over and the ratio falls 40–60 dB by 300 MHz for every surface family.
  The surface family only sets the slope in the 20–60 MHz plateau and the
  residual level. This is the regime the design study lives in — and why
  taper/element count "didn't matter" there: once θ_c is in the deep
  sidelobes, everything else in the simulation (along-track aliasing,
  firn layers) sits above the cross-track array's contribution.
- Orbital 6° (2 km ice): f_A ≈ 280 MHz. Below it the array is silent and
  the surface law rules: power-law/exponential surfaces give +10–15 dB
  from 60 to 300 MHz (worse), the Gaussian +20 dB then a turn-over. Only
  above ~300 MHz does the 10 m array begin to pull it back. This is
  C&S's regime: lower frequency, or a much larger cross-track aperture.

Rough bed (dashed): a bed with 0.10 m RMS roughness loses its coherent
return as exp(−(2 k_ice σ_b)²) — 3 dB at 100 MHz, 20 dB at 300 MHz in
ice — which adds a further penalty to high frequency in every geometry
that the surface-only picture omits. Whether the real bed is that rough
at the metre scale is a separate open question.

## Rules of thumb

1. Compute θ_c from altitude and thickness first; everything follows
   from where it sits relative to f* (surface) and f_A (antenna).
2. If θ_c is small (orbital, or very thick ice at low altitude), the
   antenna cannot help at VHF and the surface law rules: real (power-law)
   surfaces make higher frequency mildly worse, a Gaussian surface much
   worse up to f*. Lower frequency wins.
3. If θ_c is large (airborne, HAPS with km-scale ice), a fixed-span
   array puts the clutter in its sidelobes above f_A ≈ c/(W sin θ_c);
   above that, higher frequency wins through the antenna, by a lot, for
   any surface — bounded by grating lobes, pattern errors, and whatever
   other clutter path (along-track, firn layers, rough bed) is next in
   line.
4. The scattering-law trend alone (no antenna) is: more clutter with
   frequency for any realistic surface; the Gaussian ACF is the only
   family that reverses it, and the ATM data say it is the wrong family.
5. Frequency also raises the near-nadir (mid-column) clutter through
   k⁴ S and the firn plateau (C&S), independent of θ_c — a cost that
   applies in every geometry.

## What the simplified model leaves out

Along-track clutter and its rejection by SAR processing (independent of
the cross-track array; in the full simulator currently aliasing-limited),
firn-layer reflectivity and its bandwidth dependence, antenna pattern
errors (real sidelobe floors ~−30 to −40 dB one-way), volume scattering,
attenuation's weak frequency dependence, and thermal noise (which favours
higher f through galactic noise). These shift the levels, not the two
crossover frequencies.
