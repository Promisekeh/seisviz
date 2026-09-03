<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/seisviz_logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="images/seisviz_logo_light.png">
  <img src="images/seisviz_logo_light.png" alt="seisviz" width="300">
</picture>

# seisviz

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

---

## Labelling Facies

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
reported with a warning.

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

## Potential Features / Modules:


| Feature           | Description                                                           |
|---------------------------|-----------------------------------------------------------------------|
| ML Integration          | Seamless overlay of ML-predicted labels and attributes                |
| Fault Visualization     | Highlight structural features like faults and horizons                |
| Interactive Viewer     | Scroll, pan, and slice through volumes using `ipywidgets` or `Plotly` |
| Horizon & Mask Overlays | Display stratigraphic boundaries and region-based annotations         |
| Volume Metadata         | Inline/xline/depth range auto-detection                               |
| Volume animation  | Scroll through slices          |
| Synthetic Data Support  | Load and test synthetic cubes for research or ML prototyping          |

---
## Contributing

Pull requests and feedback are welcome!  
Open an issue, fork the repo, and contribute — especially if you want to help add interactivity, masks, or ML layers.

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

## Changelog

### 0.2.0 — breaking

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
- `segyio` moved to the optional `[segy]` extra; `save_seismic_slice` exported.
- Requires Python 3.9+ and matplotlib 3.5+.

---

## 📄 License
MIT License © 2025 Promise Ekeh

