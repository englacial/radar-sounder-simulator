# Foundations review (2026-08-17)

Codebase review starting from `agent_handoff_2026-08-17.md`, at commit
`40c2503`, working tree clean, 352 unit tests green (verified: 74.9 s).

Scope: documentation-vs-code discrepancies, dead code / refactor
opportunities, and — the priority ask — **whether each active experiment has
a working reproduction pathway**.

Everything below was verified against the code and the output tree, not
inferred from the notes. Findings are ordered by how much they threaten the
science, not by how hard they are to fix.

---

## A. Reproduction pathways — the state of play

Short version: **the simulations are reproducible, the *experiments* are
not.** Chunk caching, snapshot pinning and cache-key hygiene are genuinely
good at the level of "this chunk of this pass". What is missing is the layer
above: the exact invocation that produced a named result, and a
machine-readable record tying a result directory to it.

### A1. The two headline configs have no recorded command (HIGH)

| experiment | command recorded? | where |
|---|---|---|
| Antarctic `att20_klevel` (50 km) | yes | `claude_notes/klevel_sweep.sh` |
| Antarctic `full_line` | partly | prose block, `basal_clutter_pilot_findings.md:1775` — records the `--passes high` staging run only |
| Antarctic `extended` | partly | `basal_clutter_pilot_findings.md:1566` |
| Greenland `full_pbed_proc_att14_rssnr` | **no** | described only as "everything else identical to the A = 14.0 constant-gamma run" (`greenland_pair_findings.md:679`) |
| T5 scan, att sweep | yes | `claude_notes/t5_scan.sh`, `att_sweep.sh` |

The Greenland run — the *active* line — cannot be re-issued without
reverse-engineering three sources: the findings prose, `run_config.json`, and
the `--companion-name` value (which appears nowhere at all).

**Fix**: an `experiments/` directory of small committed shell (or TOML)
recipes, one per named result, each ending in the exact `uv run …` line. Move
`klevel_sweep.sh` / `att_sweep.sh` / `t5_scan.sh` there out of
`claude_notes/` (they are reproduction assets, not session artifacts), and
add the two missing ones. This is the single highest-value change on the list.

### A2. The CLI defaults reproduce a *rejected* configuration (HIGH)

`tools/run_basal_clutter.py` defaults, as they stand today:

- `--att` default **31.0** (`main()`, line 4165) — the family analysis
  rejected 31 in favour of **A = 20**; the help text still calls 31 "the T2
  value the user adopted 2026-08".
- `--anchor` default **median** — both adopted configs use `level`.
- `LEVEL_ANCHOR_DEFICIT_DB = 14.8` (line 168) — the adopted Antarctic run
  supplies `--level-deficit-db 3.56` (contamination-aware, derived in
  `klevel_sweep.sh`'s header comment). 14.8 is the *stale* median-to-median
  value at att 31. The Greenland registry entry was updated to the
  contamination-aware derivation (−7.89); the Antarctic module default never
  was.

So `uv run python tools/run_basal_clutter.py --segment full --demogorgn-bed
--gamma-from-rssnr --anchor level` silently produces a K that matches nothing
in the findings. Worse, `run()`'s own signature defaults `att=rac.ATT_DB_PER_KM`
(= **15**), so a programmatic caller and a CLI caller disagree by 16 dB/km.

**Fix**: make the adopted per-line values registry entries (`ATT_DB_PER_KM`,
`ANCHOR`, `LEVEL_ANCHOR_DEFICIT_DB`), default the CLI to them, and delete the
`run()`/argparse default split. Keep the stale numbers only in a comment that
says they were rejected.

### A3. `full_line/metrics.json` describes one synthetic pass (HIGH)

`outputs/basal_clutter/full_line/` was built by running one pass at a time
(`--passes low`, `--passes mid`, …) and hand-renaming the outputs. The
directory holds `metrics_low.json`, `metrics_mid.json`, `metrics_high.json`,
`metrics_syn14km.json`, `metrics_syn300km.json` — and a top-level
`metrics.json` that is a leftover of the last run, containing **only
`syn300km`**.

`tools/make_report.py` reads only `metrics.json`, so the aggregate
verification report's `basal_clutter_full_line` case shows the syn300km
metrics beside all six passes' figures. There is no `altitude_trend` and no
`zone_split_low/mid/high` anywhere in the canonical file.

**Fix**: give `run()` a merge mode — when `--passes` is used, write
`metrics_<key>.json` by name and merge into `metrics.json` instead of
overwriting it (the per-pass metric keys are already namespaced by pass, so
a dict merge is sufficient). Alternatively have `make_report.py` glob
`metrics*.json` per case. The first is better: the canonical file should be
complete.

### A4. `run_config.json` cannot identify the code that produced it (HIGH)

It records the resolved physics config, but not:

- the command line,
- the git SHA (`make_report.py` has `_git_sha()`; no run tool uses it),
- the package versions / `uv.lock` hash,
- `--processing`, `--proc-cache`, `--passes`, `--anchor` (nested only),
  `--add-*` as top-level fields.

The schema has also drifted: `full_line` and `att20_klevel` were written by
different code generations and disagree on which keys exist (`line`,
`per_pass_figs`, `trace_decomp_s_km`, `k_anchor_segment` are present in some
and absent in others). Nothing detects this.

**Fix**: one `provenance` block written by a shared helper —
`{"argv", "git_sha", "git_dirty", "soundersim_version", "created",
"host"}` — plus a `config_schema_version` integer that is bumped whenever
the block changes.

### A5. Chunk cache keys omit physical inputs (MEDIUM, correctness risk)

`rac.run_level` invalidates strictly on the caller's `meta` dict
(`run_altitude_comparison.py:801`). `run_basal_clutter.chunk_meta`
(line 2120) keys on `"surf_rough": bool(...)` — **not** the values.
`rac.SURF_ROUGH_SIGMA_M = 0.049474` / `SURF_ROUGH_CL_M = 2.982179` and
`SimConfig.roughness_seed` (the speckle realization) are invisible to the
cache. So is `EPS_ICE` / `EPS_BED`.

Change any of those and 4.3 GB of Antarctic chunk cache silently serves the
old physics. Note `run_altitude_comparison` gets this right for itself
(`meta["surf_rough"] = [SIGMA, CL]`, line 1012) — `run_basal_clutter` is the
weaker one, and it is the one carrying the active studies.

Those same values are also absent from `run_config.json`, so a completed run
does not record the surface roughness it used.

**Fix**: put the tuple (and `roughness_seed`, `EPS_ICE`, `EPS_BED`) into
`chunk_meta` and into the config block. This invalidates existing caches
once — worth it, and it can be staged by keying on the values only when they
differ from today's constants (the pattern the file already uses for `att`).

### A6. No supported entry point for figure regeneration (MEDIUM)

`--proc-cache` writes 669 MB (Antarctic full_line) + 96 MB (Greenland) of
focused stacks whose stated purpose is "plot iterations drop from ~30 min to
seconds via `load_proc_pass`". But `load_proc_pass` has **no in-tool
consumer** — it is called only from `claude_notes/trace4_fig.py` (a scratch
script) and one smoke test. The handoff's open item "stale truncated
decomposition figures in the older run dirs — regenerate from proc caches if
ever shown" has no command behind it.

**Fix**: a `--figures-only` mode on `run()` that loads every pass through
`load_proc_pass`, rebuilds figures + metrics, and errors loudly if any pass
misses the cache. That also makes the staleness in the old run dirs
addressable in one command.

### A7. The 4.5 GB data cache has no manifest (MEDIUM)

`outputs/cache/` is gitignored, so a fresh clone re-fetches from xOPR, PGC
STAC, NSIDC (needs `~/.netrc` Earthdata credentials) and two icechunk S3
stores. Snapshot pinning is good (`DEMOGORGN_SNAPSHOT`, both
`RSSNR_SNAPSHOT`s), but there is no way to check that a rebuilt cache matches
the one the results came from.

**Fix**: `tools/cache_manifest.py` writing a committed JSON of
`{relative path: (size, sha256, source url/snapshot)}`, and a `--verify`
mode. Cheap, and it converts "it re-downloaded, probably fine" into a check.

---

## B. Documentation ↔ code discrepancies

### B1. The antenna is 7 elements; the tool says 5, and the `array8` bracket is mis-stated (HIGH — affects a recorded result)

`rocb.N_ELEMENTS = 7` ("P-3 center cross-track array, readme table"), used by
`run_basal_clutter.radar_grid` (line 1856) for every pass of both lines.

But:
- `--antenna` help: "'array' = the MCoRDS-like **5-element** cross-track array"
- `sim_cfg` comment (line 2084): "8 elements (**1.6x** the recorded 5-element aperture)"
- `docs/processing_simulation.md`: "a 5-element cross-track array removes 8–9 dB"

Two consequences. First, the `array8` bracket is not 1.6× anything: against
the actual 7-element array the aperture ratio is (8−1)/(7−1) = **1.17×**, so
the T4b "more-directive bracket" is far weaker than the recorded
justification claims — the conclusion drawn from `t4b_array8` should be
re-read with that in mind. Second, `tests/test_basal_hypotheses.py:48`
asserts `n_elements == 5`, but it builds its own `RadarConfig` with
`n_elements=5` and only checks that `sim_cfg` passes it through — the test
does not touch the production value and cannot catch this.

Separately: a 7-element **P-3** array is being applied to a 2016 **DC-8**
flight with no comment. That may be fine, but it is an unverified transfer
sitting under the headline study.

### B2. Greenland runs record 2016 Antarctic processing provenance (HIGH)

`process_standard` writes `"real_chain": REAL_CHAIN_2016` unconditionally
(line 1580). `outputs/greenland_pair/full_pbed_proc_att14_rssnr/run_config.json`
therefore claims, for a 2014/2017 Greenland P-3 pair:

> "sar": "… (2016 param_csarp/param_sar, scout-verified)",
> "combine": "… dline 6 -> 14.85 m posting … (11/6 not directly read from the
> 2016 structs — recorded assumption)"

The Greenland scout established a *different* fast-time bin (33.3859 vs
33.3333 ns) and its own combine params. `REAL_CHAIN_2016` is not in the
registry and not in `LINE_GLOBALS`, so the test that guards against exactly
this class of leak (`test_registry_entries_only_touch_line_globals`) does not
cover it.

**Fix**: `REAL_CHAIN` as a line-specific global; add it to `LINE_GLOBALS`.

### B3. `docs/clutter_studies.md` overstates the Greenland attenuation (MEDIUM)

The doc says "Adopted values: … Greenland 20140421_01 line **16 ± 2 dB/km**".
The handoff is explicit that adopting 16 is a *pending user action* and every
current Greenland run uses **A = 14.0**. The doc reads as if 16 is in the
results; no run in `outputs/` uses it.

### B4. `docs/output.md` documents a `git_commit` attr that does not exist (LOW)

"Global attrs carry full provenance … `soundersim_version`, `git_commit`,
`created`". `output.py:96–105` writes `soundersim_version` and `created`;
`git_commit` appears nowhere in `src/` or `tools/`. Either implement it
(cheap, and it would partly solve A4) or drop the claim.

### B5. "surface-borne" / "bed-borne" still leaks into user-visible output (MEDIUM)

The user-mandated terminology is "surface returns" / "bed returns".
`docs/clutter_studies.md` and the newer code (`zone_analysis`,
`fig_decomposition_zones`, `fig_decomposition_trace`) comply; 23 occurrences
of `-borne` remain in `run_basal_clutter.py`, and two of them are *not*
comments:

- `fig_decomposition` legend labels — "sim surface-borne" / "sim bed-borne"
  (lines 3059–3095), i.e. the deprecated term is printed on the main
  decomposition figure of every run;
- `analyze_pass`'s `verdict` values `"surface-borne"` / `"bed-borne"`
  (line 2524), which land in `metrics.json` as `midcol_verdict` and in every
  figure title.

### B6. Minor stale numbers

- `PASSES[*]["agl_med_m"]` (442 / 9150 / 10684) is **written but never read**
  by the tool — only asserted in tests. Everything real uses the computed
  per-segment `h_med` (449 m on `full`, whence the handoff's "449 m"). Two
  numbers for the same thing, one of them inert.
- The per-pass console header prints the *derived* tag, not `--out-name`:
  `att20_klevel.log` opens with `== low (full_dgn_rssnr_proc) ==`. Confusing
  when reading logs back.

---

## C. Latent bugs in the line-registry mechanism

The registry is a good design and its test file is careful, but four gaps
remain in the same class the module docstring declares "dead":

### C1. Four defaults bind line-specific globals at import time

```
run_basal_clutter.py:957   def bed_rough_nadir_db(sigma_m, f0=FC_HZ, ...)
run_basal_clutter.py:965   def zone_g2_stats(gmap, run_lo, run_hi, gl_km=GL_S_KM)
run_basal_clutter.py:2655  def zone_analysis(p, a, gl_km=GL_S_KM)
run_basal_clutter.py:2754  def fig_decomposition_zones(..., gl_km=GL_S_KM, ...)
```

`rel_mean_profile` already shows the right pattern (`if lo_us is None:
lo_us = PROFILE_REL_US[0]`) and `test_rel_mean_profile_extent_follows_the_active_line`
guards it. Of the four, `bed_rough_nadir_db` is live today: on the Greenland
line `--bed-rough` would compute the double-count guard at 190 MHz instead of
195 MHz. The two `zone_*` ones are latent only because Greenland has no
`full_line` segment.

### C2. `activate_line` is not reversible

`LINES[ANTARCTIC_LINE] = {}`, so `globals().update({})` restores nothing.
Greenland → Antarctic leaves Greenland's values in place. The test suite
knows this (`test_activation_round_trips` asserts the *leak* and relies on a
fixture to clean up), which is honest but means any process that activates
two lines — a future batch driver, a notebook — is silently wrong.

**Fix**: capture the Antarctic defaults into `LINES[ANTARCTIC_LINE]` at
import (`{k: globals()[k] for k in LINE_GLOBALS}`), which makes activation
idempotent and reversible with no behaviour change on the first activation.
Then export `LINE_GLOBALS` from the module so the test stops maintaining its
own copy.

### C3. `run(line=...)` mutates module state and never restores it

Same root cause; matters once anything drives several lines in one process.

---

## D. Dead code, duplication, simplification

Little outright dead code — the unused-symbol sweep across `src/` and
`tools/` came back essentially clean (`compare/` is verification machinery
used by the integration tests, not dead). The problems are structural.

### D1. `tools/` is an unpackaged library (HIGH-value refactor)

```
run_basal_clutter → run_altitude_comparison → run_opr_coherent_bed → run_opr_comparison
run_cross_season  → run_altitude_comparison
run_b26_overflights → run_b26_comparison + run_firn_investigation
```

each via `sys.path.insert(0, ROOT/"tools")`. Importing `run_basal_clutter`
executes a 1520-line analysis script for `mcords_params`, `base_scene`,
`run_level`, `leading_edge_gate`, `facet_spacing`, `_lonlat`. Physics
constants (`EPS_ICE`, `EPS_BED`, `ATT_DB_PER_KM`, `N_ELEMENTS`, `BETA`,
`GATE_BINS`) live in `run_opr_coherent_bed.py` and are re-exported twice.

And it has already forked: `run_b26_comparison` carries its *own*
`_lonlat`, `facet_spacing`, `leading_edge_gate`, `surface_peak_twtt`,
`radar_grids` — four near-duplicates of `rac`'s, with different signatures.

**Fix**: promote the shared layer into `src/soundersim/campaign/` (or
`soundersim/opr_tools.py`): constants, `mcords_params`, scene/grid builders,
`run_level`, the two gate helpers. `tools/*.py` then become thin drivers.
This is the change that makes everything else in this list easier.

### D2. Six near-identical HTML report builders

`_report` in `run_basal_clutter` (4108), `run_altitude_comparison` (1423),
`run_b26_comparison` (1931), `run_cross_season` (672), plus inline builders in
`run_firn_investigation` (784) and `make_report` (209). All five case-level
ones are the same shape: CSS, metrics table with pass/fail cells, base64
figures, config `<pre>`.

Collapsing them into one `write_case_report(out, case, config, metrics,
notes, figs, extra=…)` also gives a natural place to fix **E1** below (link
PNGs relatively instead of base64-embedding).

### D3. `run()` is 800 lines with 30 keyword parameters

`run_basal_clutter.run` (3305–4105) does argument validation, RSSNR mapping,
the pass loop, the companion run, the ablation runs, metric assembly for
eight metric families, config assembly, figure dispatch and verification
mirroring. `main()` is a 1:1 transcription of it into argparse.

**Fix**: a pydantic `RunSpec` (the project already depends on pydantic and
uses it for `SimConfig`) — argparse builds one, `run()` takes one, and it
serialises straight into `run_config.json`, which solves half of **A4** for
free. Then split out `_assemble_metrics(...)` and `_emit_figures(...)`.

### D4. Two globals are only settable from `main()`

`FIG_WIDTH_SCALE` and `BED_OVERLAY` are set via `global` in `main()`
(4341–4343), so `run()` called programmatically cannot control them and
always uses the module defaults. They belong in the `RunSpec` above.

### D5. `LAM_ICE_M` is dead

Defined at line 156, rebound in the Greenland registry entry (line 561),
**never read**. Its value is separately hardcoded into the
`BED_ROUGH_VALIDITY` string ("lambda_ice = 0.886 m at 190 MHz"), which is now
wrong for the Greenland line. Delete the constant or use it.

### D6. Packaging: the tools need dev dependencies

`scipy`, `icechunk` and `zarr` are in `[dependency-groups] dev`, but
`run_basal_clutter.py` imports scipy at module level and uses icechunk+zarr
for the RSSNR fetch. `uv sync` without the dev group produces an environment
in which the main analysis tool cannot start. Move them to `dependencies`
(or add an `analysis` extra and document it).

### D7. Lint is not usable as a gate

No `[tool.ruff]` section in `pyproject.toml`, so `uv run ruff check .` yields
81 errors: 42 `E741` (`l` as a variable — mostly correlation length, a
legitimate physics name), 17 `E731`, 9 `E402` (deliberate, from
`matplotlib.use("Agg")`), 7 `F401`, 6 `F841`. Adding a ruff config that
ignores the three deliberate classes turns the remaining ~13 into real
signal.

---

## E. Repo hygiene

### E1. `outputs/verification/report.html` is ~80 MB and is committed (HIGH)

19 versions in history. The ten largest blobs in the repo are all
`report.html`, totalling ~500 MB — essentially the entire 539 MB `.git`.
Every regeneration adds another ~80 MB permanently.

Root cause: `make_report.py` base64-embeds every PNG. Two options:

1. **Preferred**: stop embedding — write `report.html` alongside a
   `report_figs/` directory and reference PNGs relatively; commit neither, or
   commit only a small index. Add `outputs/verification/report.html` to
   `.gitignore` and regenerate on demand (`uv run python tools/make_report.py`
   already takes seconds).
2. Keep the self-contained artefact but stop committing it; publish it
   out-of-band when it needs sharing.

Either way the existing history needs a `git filter-repo` pass to actually
recover the 500 MB — worth doing *before* the tree grows further, and it is a
coordinated rewrite, so it should be a deliberate decision rather than a
drive-by fix.

### E2. `outputs/` is 11 GB

4.5 GB `cache/` (justified — see A7), 4.3 GB `basal_clutter/`, 641 MB
`greenland_pair/`. Not a problem per se, but it makes the "which run
directories are current" question expensive, and several directories mix
generations of figures (`full_line/` holds both the unsuffixed combined
figures from an early run and the per-pass suffixed set from the staged runs;
the handoff already flags the stale Greenland decomposition figures).

**Fix**: a `STATUS.md` or a `status` key in each run dir's `run_config.json`
(`current` / `superseded-by:<dir>` / `scratch`), and a small
`tools/prune_outputs.py --dry-run`.

---

## Suggested order of work

Sequenced so each step makes the next cheaper.

1. **`experiments/` recipes** (A1) — 4 files, no code change. Immediately
   makes both active lines re-runnable. Do this first regardless of anything
   else on the list.
2. **Provenance block + config schema version** (A4) and **`git_commit` in
   the Dataset attrs** (B4) — one shared helper, ~40 lines.
3. **Registry hardening** (C1–C3) + `REAL_CHAIN` per line (B2) + delete
   `LAM_ICE_M` (D5) — small, well covered by the existing test file, closes
   the last of the "Antarctic-hardcoding" class.
4. **Adopted defaults into the registry** (A2) and the antenna correction
   (B1) — both change what a default run *means*, so they want a deliberate
   decision and a note in the findings. The `array8` re-read is a science
   item, not a code item.
5. **Cache keys** (A5) — invalidates caches once; best done alongside 4 so
   there is a single re-simulation event.
6. **`full_line` metrics merge** (A3) + **`--figures-only`** (A6) — makes the
   headline result whole and the stale figures fixable.
7. **Extract the shared campaign library** (D1) and collapse the report
   builders (D2); then `RunSpec` (D3–D4). This is the big one and it should
   come after the correctness items, not before.
8. **Repo hygiene**: ruff config (D7), dependency groups (D6), then the
   `report.html` decision (E1).

Items 1–3 are cheap and carry no scientific risk. Items 4–5 change results
and should be paired with a re-run of `att20_klevel` as the regression check
(≈39 min sim + 6 min processing per the recorded timings).

---

# Status update — 2026-08-18

Work done against the list above, in one session. Test suite 373 passed
(was 352), integration config tests 19 passed, `ruff check` 81 errors -> 5.
The chunk-cache key test gated every change; all five adopted experiments
re-ran from their specs with **every chunk `[skip-exists]`**, which is the
proof that nothing moved.

## Done

- **A1 / A2 (recipes + defaults).** `experiments/` now holds a declarative
  YAML spec per adopted result, loaded through a pydantic `RunSpec`
  (`tools/clutter_spec.py`) and run with `--config`. `physics.att_db_per_km`
  is a required field with no default, so a spec cannot silently inherit a
  rejected attenuation. `experiments/README.md` indexes them.
- **A4 (provenance).** Every `run_config.json` now carries
  `config_schema_version`, a `config_fingerprint`, and a `provenance` block
  (argv, git sha + dirty flag, soundersim version, python, host, timestamp)
  — plus the **embedded spec** when the run came from `--config`, so an
  output directory literally contains its runnable input.
- **A3 (staged metrics).** `merge_metrics_doc()` accumulates a per-`--passes`
  build into one `metrics.json` instead of overwriting it, guarded by the
  fingerprint so a changed configuration starts a fresh file rather than
  blending two runs. `passes_present` / `passes_this_invocation` are recorded.
- **B1 (antenna).** The `--antenna` help and the `array8` comment claimed a
  5-element baseline; the tool has always used `rac.N_ELEMENTS = 7`. Strings
  are now computed, so the bracket reports its true **1.17x** aperture ratio
  rather than 1.6x, and the help states the provenance caveat (a 2017 P-3
  readme value applied unchanged to a 2016 DC-8 line).
  **The T4b conclusion should be re-read against 1.17x.**
- **B2 (REAL_CHAIN).** Was one constant named for 2016 and written into every
  line's config, so Greenland runs recorded 2016 DC-8 provenance. Now a line
  global with its own Greenland entry stating what was actually verified and
  what is assumed. Covered by `test_real_chain_is_line_specific`.
- **B3 / B4 / B5.** `docs/clutter_studies.md` no longer presents Greenland
  A = 16 as adopted (14 is what runs; 16 +/- 2 is pending); `git_commit` is
  now a real Dataset attr as `docs/output.md` always claimed; the deprecated
  "-borne" terminology is gone from figure legends and from the
  `midcol_verdict` value that lands in `metrics.json`.
- **C1 / C2 / C3 (registry).** The four import-time-bound line globals now
  resolve at call time (`bed_rough_nadir_db` was live — it computed the
  Greenland double-count guard at 190 MHz instead of 195). `activate_line`
  is reversible: the Antarctic entry is captured from the live globals, so
  Greenland -> Antarctic restores every value instead of leaking. The module
  exports `LINE_GLOBALS` and the tests assert against it.
- **D5.** `LAM_ICE_M` turned out not to be dead — a test read it, and
  `BED_ROUGH_VALIDITY` hardcoded its 190 MHz value into every Greenland
  config. It is now a live line global and the validity note is a function
  that formats against the active line.
- **D6 / D7.** `scipy` / `icechunk` / `zarr` moved from the dev group to
  `[project.dependencies]` (the tools import them at module level, so
  `uv sync` without dev produced an env where the main tool could not
  start); `pyyaml` added. A `[tool.ruff]` config ignores the three
  deliberate classes (E402/E741/E731) and excludes `claude_notes`.

## Deliberately NOT done — each needs a decision, not just time

- **A5 (cache keys).** Surface roughness sigma/l, `roughness_seed` and
  `EPS_ICE`/`EPS_BED` are still absent from `chunk_meta`. Adding them
  invalidates ~5 GB of chunk cache and turns any re-run into hours of real
  compute. It should be one deliberate event paired with a full re-sim, not
  smuggled in beside a refactor — and doing it now would have destroyed the
  cache-hit evidence that this refactor changed nothing.
- **A6 (`--figures-only`).** Still no supported entry point for regenerating
  figures from the proc caches; `load_proc_pass` remains callable only from
  scratch scripts. Additive scope, and not needed for the re-runs.
- **D1 / D2 (library extraction, report-builder collapse).** The
  `tools/` import chain and the six near-identical HTML builders are
  untouched. This is the big refactor and deserves its own effort.
- **E1 (80 MB `report.html` x 19 in history).** Requires a `git filter-repo`
  history rewrite — destructive and coordinated, so it needs explicit
  sign-off rather than a drive-by fix.
- Two sweep families (`att_sweep.sh`, `t5_scan.sh`) are still shell; they are
  multi-run and want the deferred `matrix:` block.
- 5 `F841` unused-locals remain in tests — real but trivial backlog.
