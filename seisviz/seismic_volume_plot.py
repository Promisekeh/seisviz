# seismic_volume_plot.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib

def plot_seismic_3d_slices(seismic_volume, 
                            direction='depth', 
                            start=0,
                            step=20, 
                            threshold=0, 
                            # threshold=0.1, 
                            point_size=1, 
                            alpha=0.5, 
                            cmap='seismic'):
    """
    
    Create a sparse 3D visualization by plotting amplitude points above a threshold.

    Args:
        seismic_volume (np.ndarray): 3D volume (xlines, inlines, depths)
        direction (str): 'inline', 'xline', or 'depth'
        start (int): Starting slice index
        step (int): Step between slices
        threshold (float): Min amplitude to display
        point_size (float): Marker size
        alpha (float): Opacity
        cmap (str): Colormap name
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    xlines, inlines, depths = seismic_volume.shape
    if threshold == 0:
        threshold = np.percentile(np.abs(seismic_volume), 99.5) * 0.01

    if direction == 'depth':
        for z in range(start, depths, step):
            slice_data = seismic_volume[:, :, z]
            x, y = np.where(np.abs(slice_data) > threshold)
            amp = slice_data[x, y]
            if amp.size == 0:
                continue
            ax.scatter(x, y, np.full_like(x, z), c=amp / np.max(np.abs(amp)), cmap=cmap, s=point_size, alpha=alpha)

    elif direction == 'inline':
        for i in range(start, inlines, step):
            slice_data = seismic_volume[:, i, :]
            x, y = np.where(np.abs(slice_data) > threshold)
            amp = slice_data[x, y]
            if amp.size == 0:
                continue
            ax.scatter(x, np.full_like(x, i), y, c=amp / np.max(np.abs(amp)), cmap=cmap, s=point_size, alpha=alpha)

    elif direction == 'xline':
        for xline in range(start, xlines, step):
            slice_data = seismic_volume[xline, :, :]
            x, y = np.where(np.abs(slice_data) > threshold)
            amp = slice_data[x, y]
            if amp.size == 0:
                continue
            ax.scatter(np.full_like(x, xline), x, y, c=amp / np.max(np.abs(amp)), cmap=cmap, s=point_size, alpha=alpha)

    else:
        raise ValueError("direction must be one of: 'inline', 'xline', 'depth'")

    ax.set_xlabel('Xline')
    ax.set_ylabel('Inline')
    ax.set_zlabel('Depth')
    ax.invert_zaxis()
    ax.set_title(f'3D Seismic Amplitude View ({direction})')
    plt.tight_layout()
    plt.show()


def plot_multiple_seismic_slices_3d(seismic_volume,
                                    inline_idxs=None,
                                    xline_idxs=None,
                                    depth_idxs=None,
                                    cmap='seismic',
                                    alpha=0.8,
                                    elev=30,
                                    azim=-60):
    """
    Plot multiple orthogonal slices (inline, xline, depth) in a 3D space.

    Args:
        seismic_volume (np.ndarray): 3D cube (xlines, inlines, depths)
        inline_idxs (list[int], optional): Inline indices
        xline_idxs (list[int], optional): Xline indices
        depth_idxs (list[int], optional): Depth slice indices
        cmap (str): Colormap name
        alpha (float): Transparency
        elev (float): Elevation angle
        azim (float): Azimuth angle
    """
    xlines, inlines, depths = seismic_volume.shape
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(vmin=-np.max(np.abs(seismic_volume)), vmax=np.max(np.abs(seismic_volume)))

    # Depth slices (XY)
    if depth_idxs:
        for z in depth_idxs:
            if 0 <= z < depths:
                data = seismic_volume[:, :, z]
                x, y = np.meshgrid(np.arange(inlines), np.arange(xlines))
                ax.plot_surface(x, y, np.full_like(x, z),
                                facecolors=matplotlib.colormaps[cmap](norm(data)),
                                rstride=1, cstride=1, shade=False, alpha=alpha)

    # Inline slices (YZ)
    if inline_idxs:
        for i in inline_idxs:
            if 0 <= i < inlines:
                data = seismic_volume[:, i, :].T
                y, z = np.meshgrid(np.arange(xlines), np.arange(depths))
                ax.plot_surface(np.full_like(y, i), y, z,
                                facecolors=matplotlib.colormaps[cmap](norm(data)),
                                rstride=1, cstride=1, shade=False, alpha=alpha)

    # Xline slices (XZ)
    if xline_idxs:
        for x in xline_idxs:
            if 0 <= x < xlines:
                data = seismic_volume[x, :, :].T
                x2, z = np.meshgrid(np.arange(inlines), np.arange(depths))
                ax.plot_surface(x2, np.full_like(x2, x), z,
                                facecolors=matplotlib.colormaps[cmap](norm(data)),
                                rstride=1, cstride=1, shade=False, alpha=alpha)

    ax.set_xlim(0, inlines)
    ax.set_ylim(0, xlines)
    ax.set_zlim(0, depths)
    ax.set_xlabel('Inline')
    ax.set_ylabel('Xline')
    ax.set_zlabel('Depth')
    title = "3D Orthogonal Seismic Slices"
    if inline_idxs or xline_idxs or depth_idxs:
        lines = []
        if inline_idxs:
            lines.append(f"Inlines {inline_idxs}")
        if xline_idxs:
            lines.append(f"Xlines {xline_idxs}")
        if depth_idxs:
            lines.append(f"Depths {depth_idxs}")
        title += " (" + ", ".join(lines) + ")"
    ax.set_title(title)   
    ax.view_init(elev=elev, azim=azim)
    ax.invert_zaxis()
    plt.tight_layout()
    plt.show()
