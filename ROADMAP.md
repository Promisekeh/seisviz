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
- **Real-world coordinate tie-in.** Pull CDP X/Y from SEG-Y trace headers
  (already read by `get_segy_headers`, just not surfaced as coordinates) to
  give map-view context instead of raw inline/xline indices.
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
