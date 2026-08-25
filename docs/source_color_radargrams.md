# Source-coloured radargrams

The clutter-study radargram figure (`tools/run_basal_clutter.py`,
`radargrams.png`) can tint each **simulated** panel by which interface
supplies the energy. It is off by default; enable it per line:

```yaml
figures:
  radargram:
    y_us: [-1.0, 13.5]
    db: [-90.0, 5.0]
    scale: shared
    source_color: true    # default false
```

## What the colour means

The simulator returns one coherent field per interface, so every pixel has a
surface-return power `Ps`, a bed-return power `Pb`, and the total
`P = |Fs + Fb|²`. The coloured panel is built from the grey one:

- **Brightness** is the total power `P` on the panel's dB scale — exactly the
  grey image, unchanged. The measured panels stay grey, so brightness remains
  directly comparable across the figure.
- **Hue** is the dominant source: sky blue for surface returns, orange for
  bed returns.
- **Saturation** is how dominant it is, linear in the bed power fraction
  `fb = Pb / (Ps + Pb)`: `sat = 2·|fb − 0.5|`. A pixel where the two sources
  are equal is grey; one where a single source supplies everything is fully
  coloured.

Colour therefore only ever adds *which* information on top of *how much*;
it never changes what the brightness shows.

## Palette

Okabe-Ito, chosen to stay distinguishable under protanopia, deuteranopia and
tritanopia (`SOURCE_COLORS` in `tools/run_basal_clutter.py`):

| Source | Colour |
|---|---|
| surface returns | `#56B4E9` sky blue |
| bed returns | `#E69F00` orange |
| internal layers | `#009E73` bluish green — *reserved*; the kernel returns surface + bed only today |

Blue/orange is the safest pair for the two classes that exist now; green is
held for a future per-layer field so the scheme does not need to change.

## Reading it

At low altitude the two sources rarely share a pixel, so the panel is simply
a blue surface horizon over an orange bed. The option earns its keep at high
altitude (the `haps_*` design points), where off-nadir surface clutter fills
the whole column down through the bed: the column is blue, the bed comes out
orange only where bed power actually wins, and a speckled blue/orange fringe
marks where the two are within a few dB. That fringe is real information —
it is the surface-clutter-vs-bed contest the altitude study is about — rather
than something to smooth away.

Below the bed, orange islands show where the bed's own off-nadir tail still
beats the surface clutter. Dark bands in the mid-column that stay blue are
nulls in the surface-clutter pattern, not bed energy.

## Rejected alternatives

Prototyped 2026-08-25 (`claude_notes/proto_source_color/`):

- **Additive tints** (each source drawn in its own colour and summed):
  overlap sums toward white, so the region of interest at high altitude
  becomes a pale third colour and brightness no longer equals total power.
- **Hard dominance threshold** (grey unless one source wins by > N dB):
  cleaner fringe at 3 dB, but the extra parameter buys little over the
  linear law and at 6 dB starts fragmenting the bed horizon. Not
  implemented.
- **Tinted small multiples** (grey total + one single-hue panel per source):
  fully quantitative with colour bars, but costs a column per source. The
  existing `decomposition.png` already provides the per-source view as
  profiles.
