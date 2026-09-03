"""seisviz - a lightweight Python library for visualizing 3D seismic data.

Volumes are ordered (inlines, xlines, depth) throughout. Use `reorder_volume`
to convert a cube that arrives in a different order.
"""

from .load_seismic import (
    load_seismic_data,
    load_segy_auto_3d,
    normalize_volume,
    get_segy_headers,
)
from .seismic_slice_plot import plot_2D_seismic, show_random_line
from .seismic_volume_plot import (
    plot_seismic_3d_slices,
    plot_multiple_seismic_slices_3d,
)
from .save_seismic_slice import save_seismic_slice
from .utils import reorder_volume, get_volume_range_info

__all__ = [
    "load_seismic_data",
    "load_segy_auto_3d",
    "normalize_volume",
    "get_segy_headers",
    "plot_2D_seismic",
    "show_random_line",
    "plot_seismic_3d_slices",
    "plot_multiple_seismic_slices_3d",
    "save_seismic_slice",
    "reorder_volume",
    "get_volume_range_info",
]

__version__ = "0.2.0"
