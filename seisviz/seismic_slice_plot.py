import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Diverging colormaps place their neutral colour at the midpoint of the data
# range, so amplitude limits must be symmetric about zero or the polarity of
# every peak and trough is drawn wrong.
_DIVERGING_CMAPS = {
    'seismic', 'bwr', 'coolwarm', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn',
    'PuOr', 'BrBG', 'PiYG', 'PRGn', 'Spectral', 'berlin', 'managua', 'vanimo',
}

_DISPLAY_MODES = ('overlay', 'side_by_side')
# 'time' is accepted as an alias for 'depth' - both mean "fix the
# vertical/sample axis at this index"; they slice identically. Which word
# ends up in a plot title is driven by `domain` (from `geometry`), not by
# which of the two spellings was typed - see _normalize_line_type.
_LINE_TYPES = ('inline', 'xline', 'depth', 'time')


def _normalize_line_type(line_type):
    """Validate line_type and map the 'time' alias to the canonical 'depth'."""
    if line_type not in _LINE_TYPES:
        raise ValueError(
            f"Invalid line_type {line_type!r}: use 'inline', 'xline', "
            f"'depth', or 'time'."
        )
    return 'depth' if line_type == 'time' else line_type

# Drawn for label values that label_dict['color'] does not cover.
_UNMAPPED_COLOR = 'gray'


def is_diverging(cmap):
    """Return True if `cmap` names a diverging colormap."""
    name = cmap if isinstance(cmap, str) else getattr(cmap, 'name', '')
    return name.removesuffix('_r') in _DIVERGING_CMAPS


def get_amplitude_limits(data, cmap='seismic', vmin=None, vmax=None,
                         clip_percentile=99.0):
    """
    Resolve the amplitude limits used to colour a seismic slice.

    Seismic amplitudes are not symmetric about zero, so letting matplotlib
    autoscale a diverging colormap puts its neutral colour at some non-zero
    amplitude. For diverging colormaps the limits are made symmetric around
    zero from a percentile of |amplitude|, which is the display convention
    interpreters expect.

    Args:
        data (np.ndarray): The slice being plotted.
        cmap (str): Colormap name.
        vmin (float, optional): Explicit lower limit; overrides everything.
        vmax (float, optional): Explicit upper limit; overrides everything.
        clip_percentile (float): Percentile of |amplitude| used for the
            symmetric limits. None disables clipping and uses the full range.

    Returns:
        tuple[float, float]: (vmin, vmax)
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax

    if is_diverging(cmap):
        if clip_percentile is None:
            limit = float(np.max(np.abs(data)))
        else:
            limit = float(np.percentile(np.abs(data), clip_percentile))
        if limit == 0:
            limit = 1.0
        resolved = (-limit, limit)
    else:
        resolved = (float(np.min(data)), float(np.max(data)))

    return (vmin if vmin is not None else resolved[0],
            vmax if vmax is not None else resolved[1])


def get_label_color(labels, label_dict):
    """
    Build a discrete colormap and norm that map each class to its own colour.

    Every class gets a fixed colour band, so a class keeps the same colour no
    matter which classes happen to appear in a given slice.

    Args:
        labels (np.ndarray): Label array; its unique values are used when
            `label_dict` supplies no colours.
        label_dict (dict, optional): {'color': {class: colour}, 'class':
            {class: name}}.

    Returns:
        tuple: (ListedColormap, BoundaryNorm, list of class values)
    """
    color_map = (label_dict or {}).get('color')

    if color_map:
        present = {c.item() for c in np.unique(labels)}
        missing = sorted(present - set(color_map))
        if missing:
            # Giving these classes their own bands keeps every other class on
            # its correct colour; folding them in would shift the bands.
            warnings.warn(
                f"Label classes {missing} have no entry in label_dict['color'] "
                f"and are drawn in {_UNMAPPED_COLOR}.",
                stacklevel=2,
            )
        classes = sorted(set(color_map) | present)
        colors_list = [color_map.get(c, _UNMAPPED_COLOR) for c in classes]
    else:
        # No palette given: still give each distinct class its own band, taken
        # from the full label array so slices remain comparable.
        classes = [c.item() for c in np.unique(labels)]
        base = plt.get_cmap('viridis')
        n = max(len(classes), 1)
        colors_list = [base(i / max(n - 1, 1)) for i in range(n)]

    label_cmap = colors.ListedColormap(colors_list)

    # Boundaries sit halfway between neighbouring class values, so a class is
    # matched to its colour by value rather than by rank.
    if len(classes) == 1:
        boundaries = [classes[0] - 0.5, classes[0] + 0.5]
    else:
        boundaries = (
            [classes[0] - 0.5]
            + [(a + b) / 2 for a, b in zip(classes, classes[1:])]
            + [classes[-1] + 0.5]
        )
    label_norm = colors.BoundaryNorm(boundaries, label_cmap.N)

    return label_cmap, label_norm, classes


# The vertical-axis label for inline/xline slices depends on domain, which
# isn't derivable from the data itself (see load_seismic_data). Unset/unknown
# domain keeps the historical generic 'Depth' label rather than guessing.
_VERTICAL_AXIS_LABELS = {'time': 'Time', 'depth': 'Depth'}


def get_line_type(line_type, line_number, seismic_volume, domain=None):
    """
    Extract a 2D slice from a cube ordered (inlines, xlines, depth).

    Args:
        line_type (str): 'inline', 'xline', 'depth', or 'time' ('time' is
            an alias for 'depth' - see `_normalize_line_type`).
        line_number (int): Index along that axis.
        seismic_volume (np.ndarray): 3D cube (inlines, xlines, depth).
        domain (str, optional): 'time' or 'depth', from `load_seismic_data`'s
            `geometry`. Only affects the y-axis label text for 'inline'/
            'xline' slices; anything else falls back to 'Depth'.

    Returns:
        tuple: (slice, y-axis label, x-axis label) oriented for imshow.
    """
    line_type = _normalize_line_type(line_type)
    vertical_label = _VERTICAL_AXIS_LABELS.get(domain, 'Depth')

    if line_type == 'inline':
        slice_seismic = seismic_volume[line_number, :, :].T
        yaxis_label, xaxis_label = vertical_label, 'Xline'
    elif line_type == 'xline':
        slice_seismic = seismic_volume[:, line_number, :].T
        yaxis_label, xaxis_label = vertical_label, 'Inline'
    else:
        # 'depth' (or 'time', already normalized to 'depth' above). Shape
        # is (inlines, xlines): imshow puts rows on y, columns on x.
        slice_seismic = seismic_volume[:, :, line_number]
        yaxis_label, xaxis_label = 'Inline', 'Xline'
    return slice_seismic, yaxis_label, xaxis_label


def _slice_labels(label, line_type, line_number):
    """Take the same slice out of a label cube as get_line_type takes out of a volume."""
    if line_type == 'inline':
        return label[line_number, :, :].T
    if line_type == 'xline':
        return label[:, line_number, :].T
    return label[:, :, line_number]


def _check_bounds(volume, line_type, line_number):
    line_type = _normalize_line_type(line_type)
    axis = {'inline': 0, 'xline': 1, 'depth': 2}[line_type]
    limit = volume.shape[axis]
    if not 0 <= line_number < limit:
        raise IndexError(
            f"{line_type} index {line_number} is out of range; the volume has "
            f"{limit} {line_type}s (valid 0-{limit - 1})."
        )


def _add_label_colorbar(fig, cax, mappable, classes, label_dict):
    """Attach a discrete colorbar ticked at class centres and named where possible."""
    cbar = fig.colorbar(mappable, cax=cax, ticks=classes)
    names = (label_dict or {}).get('class')
    if names:
        cbar.ax.set_yticklabels([str(names.get(c, c)) for c in classes])
    return cbar


def plot_2D_seismic(seismic_volume, line_number, line_type='inline', label=None,
                    cmap='seismic', label_dict=None, display_mode='overlay',
                    vmin=None, vmax=None, clip_percentile=99.0,
                    aspect='auto', figsize=None, label_alpha=0.5,
                    mask_labels=None, ax=None, show=False, geometry=None):
    """
    Plot a 2D seismic slice, optionally with a label overlay or side-by-side
    comparison.

    Args:
        seismic_volume (np.ndarray): 3D cube (inlines, xlines, depth).
        line_number (int): Index of the slice.
        line_type (str): 'inline', 'xline', 'depth', or 'time' ('time' is an
            alias for 'depth' - identical slicing, just matches time-domain
            vocabulary). The title word used for a vertical-axis slice
            follows `geometry['domain']` when given, regardless of which of
            the two spellings was typed here.
        label (np.ndarray, optional): Label cube with the same shape.
        cmap (str): Colormap for the seismic amplitudes.
        label_dict (dict, optional): {'color': {class: colour}, 'class':
            {class: name}}.
        display_mode (str): 'overlay' or 'side_by_side'.
        vmin, vmax (float, optional): Explicit amplitude limits.
        clip_percentile (float): Percentile of |amplitude| for symmetric limits
            when `cmap` is diverging. None uses the full range.
        aspect (str or float): Passed to imshow. 'auto' fills the axes, which
            suits sections where sample count and trace count differ widely.
        figsize (tuple, optional): Figure size in inches.
        label_alpha (float): Overlay transparency in 'overlay' mode.
        mask_labels (iterable, optional): Class values to leave transparent,
            e.g. a background class.
        ax (matplotlib.axes.Axes, optional): Draw into an existing axis.
            'overlay' mode only.
        show (bool): Call plt.show() before returning.
        geometry (dict, optional): From `load_seismic_data(...,
            return_geometry=True)`. Only `geometry['domain']` ('time' or
            'depth') is used so far, to label the vertical axis on 'inline'/
            'xline' slices ("Time" vs "Depth") instead of the generic
            'Depth'. Sample interval and trace spacing (for true vertical
            exaggeration control) are planned - see ROADMAP.md.

    Returns:
        tuple: (fig, ax) in 'overlay' mode, or (fig, (ax_seismic, ax_label))
        in 'side_by_side' mode when a label is given.
    """
    if display_mode not in _DISPLAY_MODES:
        raise ValueError(
            f"Invalid display_mode {display_mode!r}: use "
            f"{' or '.join(repr(m) for m in _DISPLAY_MODES)}."
        )
    canonical_line_type = _normalize_line_type(line_type)

    domain = (geometry or {}).get('domain')
    # A vertical-axis slice's title word follows the declared domain; without
    # one, it honors whichever of 'depth'/'time' the caller actually typed.
    if canonical_line_type == 'depth':
        title_word = domain if domain in ('time', 'depth') else line_type
    else:
        title_word = canonical_line_type

    line_number = int(line_number)
    _check_bounds(seismic_volume, canonical_line_type, line_number)

    if label is not None and label.shape != seismic_volume.shape:
        raise ValueError(
            f"label shape {label.shape} does not match volume shape "
            f"{seismic_volume.shape}."
        )

    slice_seismic, yaxis_label, xaxis_label = get_line_type(
        canonical_line_type, line_number, seismic_volume, domain=domain,
    )
    vmin, vmax = get_amplitude_limits(
        slice_seismic, cmap=cmap, vmin=vmin, vmax=vmax,
        clip_percentile=clip_percentile,
    )

    slice_label = None
    if label is not None:
        slice_label = _slice_labels(label, canonical_line_type, line_number)
        label_cmap, label_norm, classes = get_label_color(label, label_dict)
        if mask_labels is not None:
            slice_label = np.ma.masked_where(
                np.isin(slice_label, list(mask_labels)), slice_label
            )

    if display_mode == 'side_by_side' and slice_label is not None:
        if ax is not None:
            raise ValueError(
                "ax= draws into a single axis and cannot be used with "
                "display_mode='side_by_side'."
            )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (12, 6))

        im1 = ax1.imshow(slice_seismic, cmap=cmap, vmin=vmin, vmax=vmax,
                         aspect=aspect)
        ax1.set_title(f'Seismic - {title_word}: {line_number}')
        ax1.set_ylabel(yaxis_label)
        ax1.set_xlabel(xaxis_label)
        cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1)

        im2 = ax2.imshow(slice_label, cmap=label_cmap, norm=label_norm,
                         aspect=aspect)
        ax2.set_title(f'Label - {title_word}: {line_number}')
        ax2.set_ylabel(yaxis_label)
        ax2.set_xlabel(xaxis_label)
        cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.1)
        _add_label_colorbar(fig, cax2, im2, classes, label_dict)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax1, ax2)

    # 'overlay', or 'side_by_side' with nothing to compare against
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(slice_seismic, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.set_title(f'Seismic - {title_word}: {line_number}')
    ax.set_ylabel(yaxis_label)
    ax.set_xlabel(xaxis_label)
    divider = make_axes_locatable(ax)
    fig.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.05))

    if slice_label is not None and display_mode == 'overlay':
        im_label = ax.imshow(slice_label, cmap=label_cmap, norm=label_norm,
                             alpha=label_alpha, aspect=aspect)
        cax_label = divider.append_axes("right", size="5%", pad=0.55)
        _add_label_colorbar(fig, cax_label, im_label, classes, label_dict)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def show_random_line(seismic_volume, line_type='inline', label=None,
                     cmap='seismic', display_mode='overlay', label_dict=None,
                     seed=None, **kwargs):
    """
    Plot a randomly chosen slice, for quick QC of a volume.

    Args:
        seismic_volume (np.ndarray): 3D cube (inlines, xlines, depth).
        line_type (str): 'inline', 'xline', 'depth', or 'time' (an alias for
            'depth' - see `plot_2D_seismic`).
        label (np.ndarray, optional): Label cube with the same shape.
        cmap (str): Colormap for the seismic amplitudes.
        display_mode (str): 'overlay' or 'side_by_side'.
        label_dict (dict, optional): Class colours and names.
        seed (int, optional): Seed for a local RNG, so a figure can be
            reproduced. The global random state is never touched.
        **kwargs: Forwarded to plot_2D_seismic.

    Returns:
        tuple: (fig, ax) as returned by plot_2D_seismic.
    """
    canonical_line_type = _normalize_line_type(line_type)
    axis = {'inline': 0, 'xline': 1, 'depth': 2}[canonical_line_type]

    rng = random.Random(seed)
    line_number = rng.randint(0, seismic_volume.shape[axis] - 1)

    print(f"Displaying random {line_type} slice: {line_number}")
    return plot_2D_seismic(
        seismic_volume,
        line_number,
        line_type=line_type,
        label=label,
        cmap=cmap,
        label_dict=label_dict,
        display_mode=display_mode,
        **kwargs,
    )
