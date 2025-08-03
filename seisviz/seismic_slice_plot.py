import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import numpy as np
import random

def get_label_color(labels, label_dict):
    """
    Generate colormap for label overlay.
    """
    if label_dict and label_dict.get('color') is not None and labels is not None:
        unique_labels = np.unique(labels)
        colors_list = [label_dict['color'].get(lab, 'gray') for lab in unique_labels]
        label_cmap = colors.ListedColormap(list(colors_list))
    else:
        label_cmap = 'viridis'
    return label_cmap

def get_line_type(line_type, line_number, seismic_volume):
    """
    Extract seismic slice for a specific line type and index.
    """
    if line_type == 'xline':
        slice_seismic = seismic_volume[line_number, :, :].T
        yaxis_label = 'Depth'
        xaxis_label = 'Inline'
    elif line_type == 'inline':
        slice_seismic = seismic_volume[:, line_number, :].T
        yaxis_label = 'Depth'
        xaxis_label = 'Xline'
    elif line_type == 'depth':
        slice_seismic = seismic_volume[:, :, line_number]
        yaxis_label = 'Inline'
        xaxis_label = 'Xline'
    else:
        raise ValueError("Invalid line type: Use 'inline', 'xline', or 'depth'")
    return slice_seismic, yaxis_label, xaxis_label

def plot_2D_seismic(seismic_volume, line_number, line_type='inline', label=None,
                    cmap='seismic', label_dict=None, display_mode='overlay'):
    """
    Plot a 2D seismic slice with optional label overlay or side-by-side comparison.
    """
    line_number = int(line_number)
    slice_seismic, yaxis_label, xaxis_label = get_line_type(line_type, line_number, seismic_volume)

    if display_mode == 'overlay':
        fig, ax = plt.subplots()
        im = ax.imshow(slice_seismic, cmap=cmap)
        ax.set_title(f'Seismic - {line_type}: {line_number}')
        ax.set_ylabel(yaxis_label)
        ax.set_xlabel(xaxis_label)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        if label is not None:
            if line_type == 'xline':
                slice_label = label[line_number, :, :].T
            elif line_type == 'inline':
                slice_label = label[:, line_number, :].T
            elif line_type == 'depth':
                slice_label = label[:, :, line_number]

            im1 = ax.imshow(slice_label, cmap=get_label_color(slice_label, label_dict), alpha=0.5)
            cax1 = divider.append_axes("right", size="5%", pad=0.55)
            fig.colorbar(im1, cax=cax1)
        plt.show()

    elif display_mode == 'side_by_side':
        if label is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            im1 = ax1.imshow(slice_seismic, cmap=cmap)
            ax1.set_title(f'Seismic - {line_type}: {line_number}')
            ax1.set_ylabel(yaxis_label)
            ax1.set_xlabel(xaxis_label)
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im1, cax=cax1)

            if line_type == 'xline':
                slice_label = label[line_number, :, :].T
            elif line_type == 'inline':
                slice_label = label[:, line_number, :].T
            elif line_type == 'depth':
                slice_label = label[:, :, line_number]

            im2 = ax2.imshow(slice_label, cmap=get_label_color(slice_label, label_dict))
            ax2.set_title(f'Label - {line_type}: {line_number}')
            ax2.set_ylabel(yaxis_label)
            ax2.set_xlabel(xaxis_label)
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im2, cax=cax2)
            plt.tight_layout()
            plt.show()

        else:
            fig, ax = plt.subplots()
            im = ax.imshow(slice_seismic, cmap=cmap)
            ax.set_title(f'Seismic - {line_type}: {line_number}')
            ax.set_ylabel(yaxis_label)
            ax.set_xlabel(xaxis_label)
            plt.tight_layout()
            plt.show()

def show_random_line(seismic_volume, line_type='inline', label=None,
                     cmap='seismic', display_mode='overlay', label_dict=None):
    """
    Plot a random seismic slice based on the chosen direction.
    """
    if line_type == 'xline':
        line_number = random.randint(0, seismic_volume.shape[0] - 1)
    elif line_type == 'inline':
        line_number = random.randint(0, seismic_volume.shape[1] - 1)
    elif line_type == 'depth':
        line_number = random.randint(0, seismic_volume.shape[2] - 1)
    else:
        raise ValueError("Invalid line type: Use 'inline', 'xline', or 'depth'")

    print(f"Displaying random {line_type} slice: {line_number}")
    plot_2D_seismic(
        seismic_volume,
        line_number,
        line_type=line_type,
        label=label,
        cmap=cmap,
        label_dict=label_dict,
        display_mode=display_mode
    )
