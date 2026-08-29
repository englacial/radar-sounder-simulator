# 14 km HAPS configuration study, Pine Island lines (2026-08-28)

Goal: maximise **Bed SCR** (`bed_window_bed_minus_surface_returns_db`) at 14 km AGL
on `antarctica_pineisland_south` and `_north`, within fc 50-300 MHz, fractional
bandwidth <= 40 %, array <= 10 m and <= 8 elements.

Runs used a temporary `probe` segment on both lines (= the pilot window with 5,
later 15, decomposition traces). **The probe segments have been removed.**
Experiments here are one-offs and are deliberately NOT in `config/experiments/`.

Logs `logs/`, parser `collect.py`. Shipped instruments: `config/instruments/haps14_pig_075.yaml`
(recommended) and `haps14_pig_050.yaml` (alternative).

## Result

Bed SCR median over 15 decomposition traces, bed roughness sigma 0.10 m:

| candidate | fc / B | array | north | south | worst |
|---|---|---|---|---|---|
| f075 | 75 / 30 MHz | 8 el, 0.357 lam, 9.99 m | **-9.8** | -4.0 | **-9.8** |
| f050 | 50 / 20 MHz | 8 el, 0.238 lam, 9.98 m | -14.6 | **-2.9** | -14.6 |
| f060 | 60 / 15 MHz | 8 el, 0.285 lam, 9.97 m | -15.3 | -5.1 | -15.3 |
| f150 | 150 / 60 MHz | 8 el, 0.714 lam, 9.99 m | -17.6 | -7.9 | -17.6 |
| f300 | 300 / 120 MHz | 8 el, 1.429 lam, 10.00 m | -26.3 | -15.5 | -26.3 |

**Low frequency wins, by 16 dB over 300 MHz.** Not what the beamwidth argument
predicts. The cause is the bed-roughness coherent loss: with the study's
`bed_roughness` fixture (sigma 0.10 m, l 0.886 m) the nadir coherent bed return
loses 0.6 dB at 50 MHz but **21.8 dB at 300 MHz**, while surface Bragg backscatter
rises ~7.8 dB over the same span. Together ~29 dB against high fc, versus ~10 dB
of beam gain from the aperture. The beam never gets to matter.

Two levers turned out to be null:
* **Pulse length: exactly no effect.** 20 / 3 / 1 us gave bit-identical Bed SCR.
* **Bandwidth: 0.9 dB for 2x.** The bed window is 2 us (tens of range cells), so
  clutter in it is set by geometry, not range resolution.
* **Grating lobes: harmless.** 300 MHz at 1.43 lam (GL at 44.4 deg) matched
  300 MHz at 1.0 lam (no GL) to 0.2 dB on BOTH lines, even on south whose
  competing clutter arrives from 34.6 deg. The full 10 m aperture is always usable.

## Sensitivity

Bed roughness sigma 0.10 -> 0.05 m (north, 15 traces): every candidate shifts
-5.2 dB but the **ranking is identical** (f075 > f050 > f060 > f150 > f300).
The optimum is robust to the fixture; the absolute levels are not.

## Caveats

* **Attenuation is frequency-independent in the model** (the line's solved A is
  applied to every pass). Real VHF ice absorption lies between constant and
  ~proportional to f. If alpha ~ f, 75 MHz would gain a further **6.3 dB (north)
  / 31.8 dB (south)** of two-way bed signal that the model does not credit. The
  bias runs AGAINST low fc, so correcting it strengthens this result.
* **Surface roughness is the provisional `aa_grounded_500_1500` stratum**
  (sigma 10.8 cm, l 13.5 m), ~1.6x north's measured ATM RMS. It sets the
  fc-dependent surface clutter, i.e. one of the two terms driving the answer.
* **Bed roughness is an unmeasured study fixture** and supplies the LARGER term.
  This is the biggest single uncertainty in the result.
* **No receiver noise or link budget.** SCR only. 75 MHz / 30 MHz on a
  stratospheric platform is power-plausible; nothing here checks that.
