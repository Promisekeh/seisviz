import os

import matplotlib.pyplot as plt

from .seismic_slice_plot import plot_2D_seismic


def save_seismic_slice(seismic_volume, line_number, line_type='inline',
                       output_path='slice.png', cmap='seismic', dpi=300,
                       axis=False, **kwargs):
    """
    Render a 2D seismic slice and write it to an image file.

    Shares the rendering path with plot_2D_seismic, so the saved image uses the
    same zero-centred amplitude limits and axis labels as the on-screen figure.
    Any missing parent directories in `output_path` are created.

    Args:
        seismic_volume (np.ndarray): 3D cube (inlines, xlines, depth).
        line_number (int): Index of the slice to save.
        line_type (str): 'inline', 'xline', 'depth', or 'time' (an alias for
            'depth' - see `plot_2D_seismic`).
        output_path (str): Where to write the image, e.g. 'output/inline_45.png'.
        cmap (str): Matplotlib colormap to use.
        dpi (int): Output resolution.
        axis (bool): Draw axes, labels and colorbar. False writes the bare image.
        **kwargs: Forwarded to plot_2D_seismic (label, label_dict, vmin, ...).

    Returns:
        str: The path written.
    """
    fig, ax = plot_2D_seismic(
        seismic_volume,
        line_number,
        line_type=line_type,
        cmap=cmap,
        show=False,
        **kwargs,
    )

    if not axis:
        for a in fig.axes:
            a.set_axis_off()
        ax.set_title("")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path
