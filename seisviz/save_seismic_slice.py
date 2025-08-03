def save_seismic_slice(seismic_volume, line_number, line_type='inline', output_path='slice.png', cmap='seismic'):
    """
    Save a 2D seismic slice as an image file.

    Args:
        seismic_volume (np.ndarray): 3D seismic cube
        line_number (int): Index of the slice to save
        line_type (str): 'inline', 'xline', or 'depth'
        output_path (str): Path to save the image (e.g., 'output/inline_45.png')
        cmap (str): Matplotlib colormap to use
    """
    import matplotlib.pyplot as plt
    slice_number = int(line_number)

    if line_type == 'inline':
        data = seismic_volume[:, slice_number, :].T
    elif line_type == 'xline':
        data = seismic_volume[slice_number, :, :].T
    elif line_type == 'depth':
        data = seismic_volume[:, :, slice_number]
    else:
        raise ValueError("line_type must be one of: 'inline', 'xline', or 'depth'")

    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap=cmap)
    plt.title(f"{line_type.capitalize()} Slice {line_number}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {line_type} slice {line_number} to {output_path}")