# Roadmap

Things not yet in `seisviz`, grouped by what they'd actually buy the
library. Grouping matters more than order here — pick whichever category
matches what you're trying to fix.

Have a use case for one of these, or something not listed? Open an issue —
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Reliability

Makes what already exists trustworthy on real, messy, large data.

- **Type hints across the public API.** Better IDE autocomplete/inference,
  and catches misuse (e.g. a wrong `line_type` string) before runtime rather
  than deep inside a plotting call.
- **Memory-mapped / chunked loading for large SEG-Y and `.npy` files.**
  Everything currently loads fully into RAM. `load_segy_auto_3d` already
  reads traces in blocks internally, but the *result* is still a single
  in-memory array — a genuinely large survey (tens of GB) can't be opened at
  all right now.

## Features

Grows what the library can actually show or compute.

- **Interactive viewer.** Scroll/pan through inline, xline, and depth slices
  with an `ipywidgets` slider in Jupyter (a `Plotly`-based volume viewer is
  an alternative worth evaluating instead of, not in addition to). Highest
  expected payoff of anything on this list — directly answers "why would I
  install this over writing my own `imshow` call."
- **Basic seismic attributes.** RMS amplitude, instantaneous phase (needs
  `scipy.signal.hilbert`) — pairs naturally with the existing slice plots.
- **Horizon and fault overlays.** Load a horizon/fault surface and draw it on
  a slice, similar to how `label` overlays facies today. `mask_labels` on
  `plot_2D_seismic` already covers the "hide a background class" half of
  this; the surface-loading half is still open.
- **Vertical exaggeration control.** `plot_2D_seismic`'s current `aspect='auto'`
  fills whatever box matplotlib gives it — no relationship to real-world
  units, and not even reproducible across `figsize` choices, so every plot
  today has some unintentional exaggeration or compression nobody chose.
  Design sketched out (2026-09-03), not yet built:
  - `sv.get_segy_geometry(path)` → `{'dt', 'dx_inline', 'dx_xline', 'units',
    'domain'}`. `dt` from segyio's parsed sample positions
    (`f.samples[1] - f.samples[0]`). `dx_inline`/`dx_xline` from CDP X/Y
    trace-header coordinates (with the SEG-Y coordinate scalar applied),
    averaged over several adjacent trace pairs rather than just the first
    two. Verify segyio's exact field names for the coordinate scalar against
    its source before implementing - getting that scaling wrong gives a
    confidently wrong number, which is worse than today's honest `'auto'`.
    Warn/raise on missing coordinates or inconsistent spacing rather than
    guessing.
  - **No auto-detection of time vs. depth domain** - confirmed against
    segyio's own trace/binary header field definitions (which mirror the
    SEG-Y spec byte-for-byte): no such field exists. `MeasurementSystem`
    only covers X/Y coordinate units, not the vertical axis, and free-text
    EBCDIC header hints aren't safely machine-parseable. Every interpretation
    package asks the user to declare domain for this exact reason - so does
    this: `load_seismic_data(path, return_geometry=True, domain='time')`.
    Default `'time'` (the common case for raw SEG-Y); declared once at load
    time rather than repeated on every plot call. `domain` rides along in
    `geometry` purely for axis labeling ("Time (ms)" vs "Depth (m)") - never
    for auto time-to-depth conversion, which would need a velocity model,
    out of scope here.
  - `load_seismic_data(path, return_geometry=True)` returns
    `(volume, geometry)` instead of a bare array - opt-in, backward
    compatible.
  - `plot_2D_seismic(..., geometry=geometry, vertical_exaggeration=1.0)`
    picks `dx_inline` vs `dx_xline` internally based on `line_type`, so the
    caller can't accidentally feed it the wrong one (inline and crossline
    bin spacing are usually different, e.g. a 25m x 12.5m survey), and reads
    `geometry['domain']` for the y-axis label. Computed
    aspect = `vertical_exaggeration * dt / dx`. Raw `dx=`/`dt=`/`domain=`
    stay available underneath for `.npy` users with no header to derive from.
  - Scoped to inline/xline slices for now, where "vertical exaggeration" is
    the actual established term. Map-view (`line_type='depth'`) distortion
    from unequal inline/xline spacing is a related but separate concern, not
    included in this item.
- **Slice-scroll GIF/MP4 export.** Animate a scroll through a volume;
  `save_seismic_slice` is the natural place this would build on.
- **A public synthetic-cube generator.** The demo notebook already builds a
  synthetic dipping-reflector cube with a facies stack so it runs without
  external data — worth promoting to a real `seisviz` utility
  (`generate_synthetic_cube()` or similar) instead of living only in the
  notebook, for testing and tutorials.

## Adoption

Makes a skeptical developer trust it enough to install it.

- **Example gallery with thumbnail images.** Usually moves a skeptical
  developer more than anything about the API itself — seeing the actual
  output on real-shaped data.
- Everything else in this category (CI badges, `CHANGELOG.md`,
  `CONTRIBUTING.md`, issue templates, cross-platform CI) shipped in 0.2.0.

## Already shipped (for context)

Items from the old "Potential Features" table that are now done, so they're
not duplicated above:

- **Volume metadata** (inline/xline/depth range auto-detection) —
  `get_volume_range_info()`.
- **ML label overlay** — `plot_2D_seismic(..., label=..., label_dict=...)`,
  including per-class colour consistency across slices and `mask_labels` to
  hide a background class.
- **Inspecting label classes before plotting** — `get_label_info()` reports
  the classes actually present in a label volume, their counts/proportions,
  and which ones a given `label_dict` is missing.
