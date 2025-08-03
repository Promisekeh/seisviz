import numpy as np
import segyio
import os 


def load_segy_auto_3d(path):
    """
    Load SEG-Y file and reshape into a 3D cube using INLINE_3D and CROSSLINE_3D headers.

    Returns:
        np.ndarray: 3D seismic volume [xlines, inlines, depth]
    """
    with segyio.open(path, "r", strict=False) as f:
        f.mmap()  # enable fast access

        ilines = []
        xlines = []
        trace_data = []

        for i in range(f.tracecount):
            header = f.header[i]
            il = header[segyio.TraceField.INLINE_3D]
            xl = header[segyio.TraceField.CROSSLINE_3D]
            ilines.append(il)
            xlines.append(xl)
            trace_data.append(f.trace[i])

        unique_ilines = sorted(set(ilines))
        unique_xlines = sorted(set(xlines))
        n_il, n_xl = len(unique_ilines), len(unique_xlines)
        n_samples = len(f.samples)

        # Create mapping from inline/crossline to array index
        il_map = {v: i for i, v in enumerate(unique_ilines)}
        xl_map = {v: i for i, v in enumerate(unique_xlines)}

        volume = np.zeros((n_xl, n_il, n_samples), dtype=np.float32)

        for i in range(f.tracecount):
            il = ilines[i]
            xl = xlines[i]
            x_idx = xl_map[xl]
            i_idx = il_map[il]
            volume[x_idx, i_idx, :] = trace_data[i]

        return volume
def normalize_volume(volume, method="minmax"):
    if method == "minmax":
        v_min, v_max = np.min(volume), np.max(volume)
        if v_max - v_min == 0:
            return volume  # prevent divide-by-zero
        return 2 * ((volume - v_min) / (v_max - v_min)) - 1
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def load_seismic_data(path, normalize=False):
    """
    Load seismic volume data from a .npy or .sgy/.segy file.

    Args:
        path (str): Path to the seismic data file.
        normalize (bool): If True, normalize data to [-1, 1] using min-max scaling.

    Returns:
        np.ndarray: 3D seismic volume with shape (xlines, inlines, depths).
    """
    if path.endswith('.npy'):
        seismic_data = np.load(path)

    elif path.endswith(('.sgy', '.segy')):
        try: 
            with segyio.open(path, "r", strict=False) as f:
                if f.ilines and f.xlines:
                    seismic_data = segyio.tools.cube(f)
                else:
                    raise Exception("Missing ilines/xlines")   

        except Exception as e:
            try:
                print(f"[INFO] Falling back to header-based loader: {e}")
                seismic_data = load_segy_auto_3d(path)
            except Exception as fallback_error:
                raise ValueError(
                    f"SEG-Y file could not be loaded: {fallback_error}"
                )

    else:
        raise ValueError("Unsupported file format. Use .npy, .sgy, or .segy.")
    
    if normalize:
        seismic_data = normalize_volume(seismic_data, method="minmax")

    return seismic_data


def get_segy_headers(path, trace_idx=0):
    """
    Return SEG-Y headers if the file is .sgy/.segy. 
    Otherwise, indicate that no headers are available.

    Args:
        path (str): Path to the seismic file.
        trace_idx (int): Index of the trace to get header for.

    Returns:
        dict or str: Dictionary of headers or message if not available.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.sgy', '.segy']:
        with segyio.open(path, "r", strict=False) as f:
            
            return {
                "textual_header": f.text[0].decode("ascii", errors="ignore"),
                "binary_header": dict(f.bin.items()),
                "trace_header": dict(f.header[trace_idx].items())
            }
    elif ext == '.npy':
        return "No SEG-Y headers available for .npy files."
    else:
        return "Unsupported file type. Only .sgy, .segy, and .npy are supported."

