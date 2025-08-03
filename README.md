# seisviz

A lightweight Python library for visualizing seismic data.

`seisviz` helps geoscientists, researchers, and students explore seismic volumes using simple, intuitive Python functions.  
Load `.sgy`, `.segy`, or `.npy` files, visualize slices, highlight features, and overlay facies or classification labels — all within a reproducible, matplotlib-based environment.

**Author**:  [Promise Ekeh](https://github.com/Promisekeh) 

**Co-author**: Perkins Offi



---

> “I created `seisviz` to give the geoscience and ML community a clean, Python-native way to visualize seismic volumes — without the overhead of legacy software.”  
> — *Promise Ekeh*

---

## Key Features (v 0.1.0)

- Load seismic volumes from `.sgy`, `.segy`, or `.npy`
- View inline, xline, or depth slices with customizable colormaps
- Overlay or compare facies/label data on slices
- Visualize multiple orthogonal 3D slices in one plot
- Explore sparse **amplitude-based structures** in 3D space
- Choose between side-by-side and overlay display modes
- Instantly view a random inline/xline/depth slice

---

## Module Overview

| Function                            | Description                                           |
|-------------------------------------|-------------------------------------------------------|
| `load_seismic_data()`               | Load `.npy`, `.sgy`, or `.segy` files                 |
| `plot_2D_seismic()`                 | Plot any seismic line with optional label overlay     |
| `show_random_line()`                | Pick a random inline/xline/depth for quick inspection |
| `plot_seismic_3d_slices()`          | Threshold-based 3D amplitude structure viewer         |
| `plot_multiple_seismic_slices_3d()` | Combine multiple orthogonal slices in one 3D plot     |

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
All visualization functions in seisviz assume the seismic cube is ordered as (xlines, inlines, depth).
This may differ from other libraries or SEG-Y readers that use (inlines, xlines, depth) or (depth, inline, xline) formats.
Please reorder your volume accordingly using `reorder_volume` if needed before plotting.



## 📄 License
MIT License © 2025 Promise Ekeh

---

## Installation

> 🔧 For now, clone and install locally:

```bash
git clone https://github.com/Promisekeh/seisviz.git
cd seisviz
pip install -e .


