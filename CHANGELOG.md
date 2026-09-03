# Changelog

All notable changes to `seisviz` are documented here.

## 0.2.0 — breaking

- **Volumes are now ordered `(inline, xline, depth)`**, matching `segyio` and
  most SEG-Y readers, so `load_seismic_data()` no longer transposes SEG-Y
  files on load. A cube in the old `seisviz` order (`xline, inline, depth`)
  can be upgraded with `reorder_volume(volume)` — no arguments needed.
  `load_seismic_data()` also takes `current_order` to reorder a `.npy` file
  as it loads.
- **Plotting functions return `(fig, ax)` and no longer call `plt.show()`.**
  Pass `show=True` for the old behaviour.
- Facies labels are drawn with a discrete norm, so each class keeps its own
  colour in every slice. Colorbars are named from `label_dict["class"]`.
- Diverging colormaps are centred on zero amplitude by default.
- Depth-slice axis labels corrected (the vertical axis is Inline, not Xline).
- Both 3D views now share one axis assignment (X=Inline, Y=Xline, Z=Depth)
  and one colour scale, and are drawn true to the survey's real proportions
  (`ax.set_box_aspect`) instead of a distorted equal-sided box, so a slice
  plane no longer looks foreshortened.
- `load_seismic_data()` and `normalize_volume()` now validate dtype and warn
  on NaN/Inf amplitudes instead of letting them propagate silently into
  colour scaling.
- `segyio` moved to the optional `[segy]` extra; `save_seismic_slice` exported.
- Requires Python 3.9+ and matplotlib 3.5+.
- CI now runs the test suite across Python 3.9–3.12 on Linux, macOS, and
  Windows, plus a dedicated job pinned to the declared dependency floor.

## 0.1.1

- Metadata fix; not published to PyPI (superseded by 0.2.0).

## 0.1.0

- Initial release: load `.sgy`/`.segy`/`.npy` volumes, 2D slice plotting with
  optional label overlays, random-slice QC, and two 3D orthogonal/sparse
  slice viewers.
