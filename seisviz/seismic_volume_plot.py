# seismic_volume_plot.py

import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .seismic_slice_plot import is_diverging

# Both 3D views use the same axis assignment as each other and as the 2D
# views: plot X is Inline, plot Y is Xline, plot Z is Depth (inverted).


def _volume_norm(seismic_volume, cmap, vmin=None, vmax=None,
                 clip_percentile=99.0):
    """
    Build one Normalize for the whole volume so every slice shares a colour scale.
    """
    if vmin is None or vmax is None:
        if is_diverging(cmap):
            if clip_percentile is None:
                limit = float(np.max(np.abs(seismic_volume)))
            else:
                limit = float(np.percentile(np.abs(seismic_volume), clip_percentile))
            limit = limit or 1.0
            lo, hi = -limit, limit
        else:
            lo, hi = float(np.min(seismic_volume)), float(np.max(seismic_volume))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    return Normalize(vmin=vmin, vmax=vmax)


def _as_index_list(idxs, name, limit):
    """
    Normalise an index argument to a list, tolerating numpy arrays.

    `if idxs:` raises on a numpy array with more than one element, which is the
    most natural thing for a caller to pass, so length is tested explicitly.
    """
    if idxs is None:
        return []
    idxs = [int(i) for i in np.atleast_1d(idxs)]

    in_range = [i for i in idxs if 0 <= i < limit]
    dropped = [i for i in idxs if i not in in_range]
    if dropped:
        warnings.warn(
            f"{name} {dropped} out of range and skipped; valid indices are "
            f"0-{limit - 1}.",
            stacklevel=3,
        )
    return in_range


def plot_seismic_3d_slices(seismic_volume,
                           direction='depth',
                           start=0,
                           step=20,
                           threshold=None,
                           point_size=1,
                           alpha=0.5,
                           cmap='seismic',
                           vmin=None,
                           vmax=None,
                           clip_percentile=99.0,
                           figsize=(12, 8),
                           ax=None,
                           show=False):
    """
    Create a sparse 3D view by scattering amplitude points above a threshold.

    All slices share one colour scale taken from the whole volume, so amplitude
    differences between slices are real rather than an artefact of per-slice
    normalisation.

    Args:
        seismic_volume (np.ndarray): 3D cube (inlines, xlines, depth).
        direction (str): 'inline', 'xline', or 'depth'.
        start (int): Starting slice index.
        step (int): Step between slices.
        threshold (float, optional): Minimum |amplitude| to draw. None picks a
            small fraction of the 99.5th percentile of |amplitude|.
        point_size (float): Marker size.
        alpha (float): Marker opacity.
        cmap (str): Colormap name.
        vmin, vmax (float, optional): Explicit amplitude limits.
        clip_percentile (float): Percentile of |amplitude| used for symmetric
            limits when `cmap` is diverging.
        figsize (tuple): Figure size in inches.
        ax (mpl_toolkits.mplot3d.Axes3D, optional): Draw into an existing 3D axis.
        show (bool): Call plt.show() before returning.

    Returns:
        tuple: (fig, ax)
    """
    if direction not in ('inline', 'xline', 'depth'):
        raise ValueError(
            f"Invalid direction {direction!r}: use 'inline', 'xline', or 'depth'."
        )

    n_inlines, n_xlines, n_depth = seismic_volume.shape
    norm = _volume_norm(seismic_volume, cmap, vmin, vmax, clip_percentile)

    if threshold is None:
        threshold = np.percentile(np.abs(seismic_volume), 99.5) * 0.01

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        # Without this, mplot3d squashes every axis into an equal-sized box
        # regardless of the survey's true proportions, so a plane's real
        # extent looks foreshortened whenever the axis ranges differ a lot.
        ax.set_box_aspect((n_inlines, n_xlines, n_depth))
    else:
        fig = ax.get_figure()

    limit = {'depth': n_depth, 'inline': n_inlines, 'xline': n_xlines}[direction]

    for n in range(start, limit, step):
        if direction == 'depth':
            slice_data = seismic_volume[:, :, n]      # (inlines, xlines)
        elif direction == 'inline':
            slice_data = seismic_volume[n, :, :]       # (xlines, depth)
        else:
            slice_data = seismic_volume[:, n, :]       # (inlines, depth)

        a, b = np.where(np.abs(slice_data) > threshold)
        amp = slice_data[a, b]
        if amp.size == 0:
            continue

        if direction == 'depth':
            # a indexes inlines, b indexes xlines
            coords = (a, b, np.full_like(a, n))
        elif direction == 'inline':
            # inline is fixed at n; a indexes xlines, b indexes depth
            coords = (np.full_like(a, n), a, b)
        else:
            # xline is fixed at n; a indexes inlines, b indexes depth
            coords = (a, np.full_like(a, n), b)

        ax.scatter(*coords, c=amp, cmap=cmap, norm=norm,
                   s=point_size, alpha=alpha)

    ax.set_xlabel('Inline')
    ax.set_ylabel('Xline')
    ax.set_zlabel('Depth')
    ax.invert_zaxis()
    ax.set_title(f'3D Seismic Amplitude View ({direction})')
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_multiple_seismic_slices_3d(seismic_volume,
                                    inline_idxs=None,
                                    xline_idxs=None,
                                    depth_idxs=None,
                                    cmap='seismic',
                                    alpha=0.8,
                                    elev=30,
                                    azim=-60,
                                    vmin=None,
                                    vmax=None,
                                    clip_percentile=99.0,
                                    stride=1,
                                    figsize=(12, 10),
                                    ax=None,
                                    show=False):
    """
    Plot orthogonal inline, xline, and depth slices together in one 3D axis.

    Each requested index draws one flat 2D plane; this does not render a
    filled solid volume. Most of the bounding box will appear empty wherever
    none of the requested planes reach it - with only one index per axis,
    the two vertical planes typically look like a narrow "open book" against
    an otherwise blank box. That's expected, not a rendering bug. Pass more
    indices per axis (or fewer, if you want less clutter) for denser coverage,
    e.g. `inline_idxs=[100, 249, 400]`.

    Args:
        seismic_volume (np.ndarray): 3D cube (inlines, xlines, depth).
        inline_idxs (sequence[int], optional): Inline indices to draw.
        xline_idxs (sequence[int], optional): Xline indices to draw.
        depth_idxs (sequence[int], optional): Depth indices to draw.
        cmap (str): Colormap name.
        alpha (float): Surface transparency.
        elev (float): Elevation viewing angle.
        azim (float): Azimuth viewing angle.
        vmin, vmax (float, optional): Explicit amplitude limits.
        clip_percentile (float): Percentile of |amplitude| used for symmetric
            limits when `cmap` is diverging.
        stride (int): Downsample factor for the drawn surfaces. Raise it on
            large cubes; stride=1 draws one quad per sample and is slow.
        figsize (tuple): Figure size in inches.
        ax (mpl_toolkits.mplot3d.Axes3D, optional): Draw into an existing 3D axis.
        show (bool): Call plt.show() before returning.

    Returns:
        tuple: (fig, ax)
    """
    n_inlines, n_xlines, n_depth = seismic_volume.shape

    inline_idxs = _as_index_list(inline_idxs, 'inline_idxs', n_inlines)
    xline_idxs = _as_index_list(xline_idxs, 'xline_idxs', n_xlines)
    depth_idxs = _as_index_list(depth_idxs, 'depth_idxs', n_depth)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        # Without this, mplot3d squashes every axis into an equal-sized box
        # regardless of the survey's true proportions, so a plane's real
        # extent looks foreshortened whenever the axis ranges differ a lot.
        ax.set_box_aspect((n_inlines, n_xlines, n_depth))
    else:
        fig = ax.get_figure()

    norm = _volume_norm(seismic_volume, cmap, vmin, vmax, clip_percentile)
    colormap = matplotlib.colormaps[cmap]
    s = max(int(stride), 1)

    def facecolors(data):
        rgba = colormap(norm(data))
        rgba[..., -1] = alpha  # bake alpha in; plot_surface ignores alpha= here
        return rgba

    # Depth slices, spanning the Inline-Xline plane
    for z in depth_idxs:
        data = seismic_volume[:, :, z].T[::s, ::s]          # (xlines, inlines)
        X, Y = np.meshgrid(np.arange(n_inlines)[::s], np.arange(n_xlines)[::s])
        ax.plot_surface(X, Y, np.full_like(X, z, dtype=float),
                        facecolors=facecolors(data),
                        rstride=1, cstride=1, shade=False)

    # Inline slices, spanning the Xline-Depth plane at a fixed Inline
    for i in inline_idxs:
        data = seismic_volume[i, :, :].T[::s, ::s]          # (depth, xlines)
        Y, Z = np.meshgrid(np.arange(n_xlines)[::s], np.arange(n_depth)[::s])
        ax.plot_surface(np.full_like(Y, i, dtype=float), Y, Z,
                        facecolors=facecolors(data),
                        rstride=1, cstride=1, shade=False)

    # Xline slices, spanning the Inline-Depth plane at a fixed Xline
    for x in xline_idxs:
        data = seismic_volume[:, x, :].T[::s, ::s]          # (depth, inlines)
        X, Z = np.meshgrid(np.arange(n_inlines)[::s], np.arange(n_depth)[::s])
        ax.plot_surface(X, np.full_like(X, x, dtype=float), Z,
                        facecolors=facecolors(data),
                        rstride=1, cstride=1, shade=False)

    ax.set_xlim(0, n_inlines)
    ax.set_ylim(0, n_xlines)
    ax.set_zlim(0, n_depth)
    ax.set_xlabel('Inline')
    ax.set_ylabel('Xline')
    ax.set_zlabel('Depth')

    title = "3D Orthogonal Seismic Slices"
    parts = []
    if inline_idxs:
        parts.append(f"Inlines {inline_idxs}")
    if xline_idxs:
        parts.append(f"Xlines {xline_idxs}")
    if depth_idxs:
        parts.append(f"Depths {depth_idxs}")
    if parts:
        title += " (" + ", ".join(parts) + ")"
    ax.set_title(title)

    ax.view_init(elev=elev, azim=azim)
    ax.invert_zaxis()
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax
