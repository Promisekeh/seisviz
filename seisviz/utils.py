import numpy as np

_AXIS_CODES = frozenset('ixd')


def _validate_order(order, name):
    """Check that an axis-order string is a permutation of 'i', 'x' and 'd'."""
    if len(order) != 3 or set(order) != _AXIS_CODES:
        raise ValueError(
            f"{name} must be a permutation of 'i' (inline), 'x' (xline) and "
            f"'d' (depth), each used once; got {order!r}."
        )


def reorder_volume(volume, current_order='xid', target_order='ixd'):
    """
    Reorder the axes of a 3D seismic volume to match the expected orientation.

    seisviz assumes cubes are ordered (inlines, xlines, depth), i.e. 'ixd'.
    The defaults convert from seisviz's pre-0.3 axis order ('xid'), so
    `reorder_volume(volume)` upgrades an old-convention cube with no other
    arguments.

    Args:
        volume (np.ndarray): 3D array to reorder.
        current_order (str): Current axis order ('i' = inline, 'x' = xline,
            'd' = depth).
        target_order (str): Desired axis order.

    Returns:
        np.ndarray: A view of `volume` with axes transposed.
    """
    _validate_order(current_order, 'current_order')
    _validate_order(target_order, 'target_order')

    if volume.ndim != 3:
        raise ValueError(
            f"Input must be a 3D array; got {volume.ndim}D with shape "
            f"{volume.shape}."
        )

    target_axes = [current_order.index(c) for c in target_order]
    return np.transpose(volume, target_axes)


def get_volume_range_info(volume):
    """
    Return index-based min/max ranges for a 3D seismic volume.

    Args:
        volume (np.ndarray): Seismic volume (inlines, xlines, depth).

    Returns:
        dict: Min/max indices for each axis, plus the shape.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Input must be a 3D array (inlines, xlines, depth); got "
            f"{volume.ndim}D with shape {volume.shape}."
        )

    n_inlines, n_xlines, n_depth = volume.shape

    return {
        "inline_range": (0, n_inlines - 1),
        "xline_range": (0, n_xlines - 1),
        "depth_sample_range": (0, n_depth - 1),
        "shape": volume.shape,
    }
