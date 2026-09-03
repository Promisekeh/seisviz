import os
import warnings

import numpy as np

from .utils import reorder_volume

# Traces are read in blocks so peak memory stays bounded on large surveys
# instead of holding a second full copy of the cube.
_TRACE_BLOCK = 20_000

_SEGY_EXTENSIONS = ('.sgy', '.segy')


def _require_segyio():
    """
    Import segyio on demand.

    segyio ships compiled wheels that lag new Python releases, so it is an
    optional dependency: users who only load .npy volumes can install and use
    seisviz without it.
    """
    try:
        import segyio
    except ImportError as e:
        raise ImportError(
            "Reading SEG-Y files requires the 'segyio' package, which is not "
            "installed. Install it with: pip install 'seisviz[segy]'"
        ) from e
    return segyio


def _extension(path):
    """Return the lowercased file extension, so SURVEY.SGY works like survey.sgy."""
    return os.path.splitext(path)[1].lower()


def _decode_textual_header(raw):
    """
    Decode a SEG-Y 3200-byte textual header.

    The SEG-Y standard specifies EBCDIC, but ASCII files are common in
    practice. Decoding EBCDIC as ASCII does not fail, it just returns
    plausible-looking garbage, so both are tried and the result with more
    printable characters wins.
    """
    best_text, best_score = "", -1
    for encoding in ("cp037", "ascii"):
        text = raw.decode(encoding, errors="replace")
        score = sum(ch.isprintable() or ch in "\r\n" for ch in text)
        if score > best_score:
            best_text, best_score = text, score
    return best_text


def load_segy_auto_3d(path):
    """
    Load a SEG-Y file and reshape it into a 3D cube using the INLINE_3D and
    CROSSLINE_3D trace headers.

    Used as a fallback for files whose geometry segyio cannot infer. Cells with
    no corresponding trace stay zero, and a warning reports the coverage so
    padding is never mistaken for quiet real data.

    Args:
        path (str): Path to the SEG-Y file.

    Returns:
        np.ndarray: 3D seismic volume with shape (inlines, xlines, depth).
    """
    segyio = _require_segyio()

    with segyio.open(path, "r", strict=False) as f:
        f.mmap()

        # Vectorised header reads: one call per field rather than one per trace.
        ilines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        _, il_idx = np.unique(ilines, return_inverse=True)
        _, xl_idx = np.unique(xlines, return_inverse=True)
        n_il = int(il_idx.max()) + 1
        n_xl = int(xl_idx.max()) + 1
        n_samples = len(f.samples)

        volume = np.zeros((n_il, n_xl, n_samples), dtype=np.float32)
        filled = np.zeros((n_il, n_xl), dtype=bool)

        for start in range(0, f.tracecount, _TRACE_BLOCK):
            stop = min(start + _TRACE_BLOCK, f.tracecount)
            block_i = il_idx[start:stop]
            block_x = xl_idx[start:stop]
            volume[block_i, block_x, :] = f.trace.raw[start:stop]
            filled[block_i, block_x] = True

    coverage = float(filled.mean())
    if coverage < 1.0:
        warnings.warn(
            f"Only {coverage:.1%} of the {n_il}x{n_xl} inline/crossline grid was "
            f"populated from {len(ilines)} traces; {int((~filled).sum())} cells "
            f"remain zero-filled. The survey geometry may be irregular.",
            stacklevel=2,
        )

    return volume


def normalize_volume(volume, method="minmax"):
    """
    Scale a volume into [-1, 1].

    Args:
        volume (np.ndarray): Seismic volume.
        method (str): Only 'minmax' is currently supported.

    Returns:
        np.ndarray: Normalized volume. Always a new array.
    """
    if method != "minmax":
        raise ValueError(f"Unsupported normalization method: {method}")

    v_min, v_max = np.min(volume), np.max(volume)
    if v_max - v_min == 0:
        warnings.warn(
            "Volume is constant; returning zeros rather than dividing by zero.",
            stacklevel=2,
        )
        return np.zeros_like(volume, dtype=np.float32)

    return (2 * ((volume - v_min) / (v_max - v_min)) - 1).astype(np.float32)


def load_seismic_data(path, normalize=False, current_order=None):
    """
    Load a seismic volume from a .npy or .sgy/.segy file.

    The returned cube always uses the seisviz axis convention,
    (inlines, xlines, depth), regardless of which loader path was taken.

    Args:
        path (str): Path to the seismic data file.
        normalize (bool): If True, scale amplitudes to [-1, 1] by min-max.
        current_order (str, optional): Axis order of a .npy file's data on
            disk (see `reorder_volume`), e.g. 'xid' for a cube saved by
            seisviz <0.3, or any other permutation of 'i', 'x', 'd'. When
            given, the volume is transposed into 'ixd' after loading. There
            is no way to detect this from the array itself; SEG-Y files
            don't need it; their axis order is always resolved by the loader.

    Returns:
        np.ndarray: 3D seismic volume with shape (inlines, xlines, depth).

    Raises:
        ValueError: If the extension is unsupported or the data is not 3D.
        ImportError: If a SEG-Y file is given without segyio installed.
    """
    ext = _extension(path)

    if ext == '.npy':
        seismic_data = np.load(path)
        if current_order is not None and current_order != 'ixd':
            seismic_data = reorder_volume(
                seismic_data, current_order=current_order, target_order='ixd'
            )

    elif ext in _SEGY_EXTENSIONS:
        segyio = _require_segyio()
        try:
            with segyio.open(path, "r", strict=False) as f:
                if not (f.ilines is not None and f.xlines is not None):
                    raise ValueError(
                        "segyio could not infer inline/crossline geometry."
                    )
                # segyio.tools.cube is already inline-major, (inlines, xlines,
                # samples), matching the seisviz convention directly.
                seismic_data = segyio.tools.cube(f)

        except (ValueError, RuntimeError, OSError) as primary_error:
            warnings.warn(
                f"Falling back to the header-based loader: {primary_error}",
                stacklevel=2,
            )
            try:
                seismic_data = load_segy_auto_3d(path)
            except Exception as fallback_error:
                raise ValueError(
                    f"SEG-Y file could not be loaded. segyio reported "
                    f"'{primary_error}'; the header-based fallback then failed "
                    f"with '{fallback_error}'."
                ) from fallback_error

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Use .npy, .sgy, or .segy."
        )

    if seismic_data.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume (inlines, xlines, depth), got a "
            f"{seismic_data.ndim}D array with shape {seismic_data.shape}."
        )

    if normalize:
        seismic_data = normalize_volume(seismic_data, method="minmax")

    return seismic_data


def get_segy_headers(path, trace_idx=0):
    """
    Return the textual, binary, and trace headers of a SEG-Y file.

    Args:
        path (str): Path to a .sgy or .segy file.
        trace_idx (int): Index of the trace whose header to read.

    Returns:
        dict: Keys 'textual_header', 'binary_header', and 'trace_header'.

    Raises:
        ValueError: If the file is not SEG-Y, or trace_idx is out of range.
        ImportError: If segyio is not installed.
    """
    ext = _extension(path)

    if ext == '.npy':
        raise ValueError(
            "No SEG-Y headers available for .npy files; they store the array only."
        )
    if ext not in _SEGY_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Headers are only available for "
            f".sgy and .segy files."
        )

    segyio = _require_segyio()

    with segyio.open(path, "r", strict=False) as f:
        if not 0 <= trace_idx < f.tracecount:
            raise ValueError(
                f"trace_idx {trace_idx} is out of range; the file has "
                f"{f.tracecount} traces."
            )

        return {
            "textual_header": _decode_textual_header(bytes(f.text[0])),
            "binary_header": dict(f.bin.items()),
            "trace_header": dict(f.header[trace_idx].items()),
        }
