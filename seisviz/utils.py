import numpy as np

def reorder_volume(volume, current_order='ixd', target_order='xid'):
    """
    Reorder the axes of a 3D seismic volume to match expected orientation.

    Args:
        volume (np.ndarray): 3D array to reorder.
        current_order (str): Current axis order ('i' = inline, 'x' = xline, 'd' = depth).
        target_order (str): Desired axis order.

    Returns:
        np.ndarray: Reordered volume.
    """
    order_map = {'i': 0, 'x': 1, 'd': 2}
    try:
        current_axes = [order_map[c] for c in current_order]
        target_axes = [current_order.index(c) for c in target_order]
    except KeyError:
        raise ValueError("Use only 'i', 'x', or 'd' in axis order strings.")

    return np.transpose(volume, target_axes)

def get_volume_range_info(volume):
    """
    Returns index-based min/max ranges for a 3D seismic volume.

    Args:
        volume (np.ndarray): Seismic volume [xlines, inlines, depth]

    Returns:
        dict: Dictionary with min/max indices for each axis
    """
    if volume.ndim != 3:
        raise ValueError("Input must be a 3D array (xlines, inlines, depth).")

    n_xlines, n_inlines, n_depth = volume.shape

    return {
        "xline_range": (0, n_xlines - 1),
        "inline_range": (0, n_inlines - 1),
        "depth_sample_range": (0, n_depth - 1),
        "shape": volume.shape
    }
