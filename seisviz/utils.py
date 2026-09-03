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


def get_label_info(labels, label_dict=None):
    """
    Summarize the classes present in a label volume.

    Useful for building a `label_dict` (see `plot_2D_seismic`) without
    guessing which values are actually present, or for checking an existing
    `label_dict` against real data before plotting.

    Args:
        labels (np.ndarray): Label volume.
        label_dict (dict, optional): {'class': {...}, 'color': {...}} to
            cross-reference against the data.

    Returns:
        dict:
            "classes": sorted list of unique values present in `labels`.
            "counts": {class: voxel count}.
            "proportions": {class: fraction of the volume}.
            "names": {class: name}, from `label_dict['class']`, for classes
                present that it names. Empty if `label_dict` is None.
            "missing_from_label_dict": classes present but absent from
                `label_dict['color']` (these render in grey - see
                `plot_2D_seismic`). None if `label_dict` is None.
    """
    values, counts = np.unique(labels, return_counts=True)
    classes = [v.item() for v in values]
    total = labels.size

    info = {
        "classes": classes,
        "counts": {c: int(n) for c, n in zip(classes, counts)},
        "proportions": {c: float(n) / total for c, n in zip(classes, counts)},
        "names": {},
        "missing_from_label_dict": None,
    }

    if label_dict is not None:
        names = label_dict.get('class') or {}
        colors = label_dict.get('color') or {}
        info["names"] = {c: names[c] for c in classes if c in names}
        info["missing_from_label_dict"] = sorted(set(classes) - set(colors))

    return info
