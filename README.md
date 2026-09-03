<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/seisviz_logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="images/seisviz_logo_light.png">
  <img src="images/seisviz_logo_light.png" alt="seisviz" width="300">
</picture>

# seisviz

[![tests](https://github.com/Promisekeh/seisviz/actions/workflows/test.yml/badge.svg)](https://github.com/Promisekeh/seisviz/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/seisviz.svg)](https://pypi.org/project/seisviz/)
[![Python versions](https://img.shields.io/pypi/pyversions/seisviz.svg)](https://pypi.org/project/seisviz/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A lightweight Python library for visualizing seismic data.

`seisviz` helps geoscientists, researchers, and students explore seismic volumes using simple, intuitive Python functions.  
Load `.sgy`, `.segy`, or `.npy` files, visualize slices, highlight features, and overlay facies or classification labels — all within a reproducible, matplotlib-based environment.

**Author**:  [Promise Ekeh](https://github.com/Promisekeh) 

**Co-author**: [Perkins Offi](https://github.com/Perkins-offi)



---

> “I created `seisviz` to give the geoscience and ML community a clean, Python-native way to visualize seismic volumes — without the overhead of legacy software.”  
> — *Promise Ekeh*

---

## Key Features (v 0.2.0)

- Load seismic volumes from `.sgy`, `.segy`, or `.npy`
- View inline, xline, or depth slices with customizable colormaps
- Overlay or compare facies/label data on slices
- Visualize multiple orthogonal 3D slices in one plot
- Explore sparse **amplitude-based structures** in 3D space
- Choose between side-by-side and overlay display modes
- Instantly view a random inline/xline/depth slice

---

## Installation

```bash
pip install seisviz
```

Reading SEG-Y files needs the optional `segyio` dependency:

```bash
pip install "seisviz[segy]"
```

`segyio` is optional because its compiled wheels lag new Python releases — if
you only work with `.npy` volumes, you don't need it.

To work on `seisviz` itself:

```bash
git clone https://github.com/Promisekeh/seisviz.git
cd seisviz
pip install -e ".[dev]"
pytest
```

---

## Quick Start

```python
import matplotlib.pyplot as plt
import seisviz as sv

volume = sv.load_seismic_data("train_seismic.npy")
labels = sv.load_seismic_data("train_labels.npy")

fig, ax = sv.plot_2D_seismic(volume, line_number=100, line_type="inline")
fig.savefig("inline_100.png", dpi=300)
plt.show()
```

Plotting functions **return `(fig, ax)`** and do not display on their own.
In Jupyter the figure still renders automatically; in a script, call
`plt.show()` or pass `show=True`.

---

## Module Overview

| Function                            | Description                                           |
|-------------------------------------|-------------------------------------------------------|
| `load_seismic_data()`               | Load `.npy`, `.sgy`, or `.segy` files                 |
| `plot_2D_seismic()`                 | Plot any seismic line with optional label overlay     |
| `show_random_line()`                | Pick a random inline/xline/depth for quick inspection |
| `plot_seismic_3d_slices()`          | Threshold-based 3D amplitude structure viewer         |
| `plot_multiple_seismic_slices_3d()` | Combine multiple orthogonal slices in one 3D plot     |
| `save_seismic_slice()`              | Render a slice straight to an image file              |
| `get_segy_headers()`                | Read textual, binary, and trace headers from SEG-Y    |
| `reorder_volume()`                  | Transpose a cube into the `(inline, xline, depth)` order |
| `get_volume_range_info()`           | Report the valid index range on each axis             |
| `get_label_info()`                  | See which classes are actually present in a label volume |

---

## Labelling Facies

Not sure what classes are actually in your label volume yet? Check before
building a `label_dict`, rather than guessing:

```python
sv.get_label_info(labels)
# {'classes': [0, 1, 3, 5],
#  'counts': {0: 30000, 1: 30000, 3: 30000, 5: 30000},
#  'proportions': {0: 0.25, 1: 0.25, 3: 0.25, 5: 0.25},
#  'names': {}, 'missing_from_label_dict': None}
```

Pass a `label` cube the same shape as your volume, plus a `label_dict`
describing the classes:

```python
label_dict = {
    "class": {0: "Class A", 1: "Class B", 2: "Class C"},   # colorbar names
    "color": {0: "dodgerblue", 1: "darkorange", 2: "skyblue"},
}

# Translucent labels over the amplitudes
fig, ax = sv.plot_2D_seismic(
    volume, 220, line_type="depth",
    label=labels, label_dict=label_dict,
    mask_labels=[0],          # leave the background class transparent
)

# Seismic and labels as two panels
fig, (ax_seis, ax_lab) = sv.plot_2D_seismic(
    volume, 550, line_type="inline",
    label=labels, label_dict=label_dict,
    display_mode="side_by_side",
)
```

Each class keeps the same colour in every slice, whether or not it appears in
that slice. Classes missing from `label_dict["color"]` are drawn in grey and
reported with a warning — or check up front with
`get_label_info(labels, label_dict)["missing_from_label_dict"]`.

---

## Amplitude Scaling

Diverging colormaps such as `seismic` place their neutral colour at the middle
of the value range, so amplitude limits must be symmetric about zero or peaks
and troughs are drawn with the wrong polarity. `seisviz` handles this by
default, clipping to the 99th percentile of `|amplitude|`:

```python
sv.plot_2D_seismic(volume, 100, clip_percentile=99.0)   # default
sv.plot_2D_seismic(volume, 100, clip_percentile=None)   # full symmetric range
sv.plot_2D_seismic(volume, 100, vmin=-2.0, vmax=2.0)    # explicit limits
```

Sequential colormaps (`viridis`, `gray`) use the data range unchanged.

---

## Orthogonal 3D Slices

`plot_multiple_seismic_slices_3d()` draws one flat 2D plane per index you
pass — it does **not** render a filled solid volume. With one index per axis,
the two vertical planes typically look like a narrow "open book" standing in
an otherwise empty box:

```python
fig, ax = sv.plot_multiple_seismic_slices_3d(
    volume, inline_idxs=[249], xline_idxs=[249], depth_idxs=[249],
)
```

Large blank areas are the bounding box's interior showing through wherever
none of the requested planes reach — that's expected, not a bug. (The box is
drawn true to the survey's real inline/xline/depth proportions, so a plane's
full extent renders correctly rather than looking foreshortened.) Pass more
indices per axis for denser coverage:

```python
fig, ax = sv.plot_multiple_seismic_slices_3d(
    volume,
    inline_idxs=[100, 249, 400],
    xline_idxs=[100, 249, 500],
    depth_idxs=[50, 150, 249],
)
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what's planned — an interactive
`ipywidgets` viewer and memory-mapped loading for large surveys are the
current top picks.

---
## Contributing

Pull requests and feedback are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)
for dev setup and testing, and [ROADMAP.md](ROADMAP.md) for what's planned —
especially if you want to help with interactivity, overlays, or ML-facing
features.

## Axis Assumption

Every function in `seisviz` orders the cube as **`(inlines, xlines, depth)`**
— inline-major, matching `segyio` and most SEG-Y readers, so SEG-Y files load
with no reordering needed.

`.npy` files carry no axis-order metadata, so `load_seismic_data()` assumes
one is already `(inline, xline, depth)` unless you tell it otherwise with
`current_order`:

```python
# a .npy that was saved (xline, inline, depth) — e.g. by seisviz <0.3,
# or a benchmark dataset documented as crossline-major
volume = sv.load_seismic_data("train_seismic.npy", current_order="xid")
```

Already have it loaded? `reorder_volume` does the same transpose directly:

```python
volume = sv.reorder_volume(volume, current_order="xid", target_order="ixd")

sv.get_volume_range_info(volume)
# {'inline_range': (0, 400), 'xline_range': (0, 700),
#  'depth_sample_range': (0, 254), 'shape': (401, 701, 255)}
```

**There's no way to detect the order from the array itself** — a plain
`.npy` is just numbers, with nothing recording which axis is which. If you
don't know it:
- Check the source. A survey's inline and crossline counts are usually in its
  documentation, an accompanying SEG-Y's headers (`get_segy_headers`), or the
  acquisition report — whichever axis length matches the *inline* count is
  axis 0 in the seisviz convention.
- Or plot it both ways with `line_type='depth'` and look: a genuine depth
  (time) slice reads as a coherent map view — channels, blobs, a plausible
  geological shape — while slicing the wrong axis produces a section that
  looks like layered inline/xline reflectors instead. If a "depth" slice
  looks stripy rather than map-like, your axes are swapped.

---

## Time vs. Depth Domain

The vertical axis on an inline/xline slice is labeled "Depth" by default,
but your data might actually be in **time** (milliseconds, two-way travel
time) — the common case for raw SEG-Y. Declare it once at load time:

```python
volume, geometry = sv.load_seismic_data("survey.sgy", domain="time",  # the default
                                        return_geometry=True)
fig, ax = sv.plot_2D_seismic(volume, 100, line_type="inline", geometry=geometry)
# y-axis now reads "Time" instead of the generic "Depth"
```

**seisviz never infers this from the file** — no SEG-Y header reliably
records it (verified against `segyio`'s own binary/trace header field
definitions; the closest field, `MeasurementSystem`, covers X/Y coordinate
units, not the vertical axis). Every interpretation package asks the user to
declare domain for the same reason. `domain` only changes axis labeling —
it never triggers a time-to-depth conversion, which would need a velocity
model `seisviz` doesn't have.

---

## Changelog

0.2.0 is a breaking release: the axis convention flipped to
`(inline, xline, depth)`, plotting functions return `(fig, ax)` instead of
calling `plt.show()`, and several silent rendering bugs (label colours,
zero-centred amplitude scaling, 3D box proportions) are fixed.
See [CHANGELOG.md](CHANGELOG.md) for the full list.

---

## 📄 License
MIT License © 2025 Promise Ekeh

