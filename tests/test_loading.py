"""
Loader behaviour that does not need a real SEG-Y file: extension handling,
shape validation, normalisation, and the optional-segyio contract.
"""

import builtins

import numpy as np
import pytest

from seisviz import load_seismic_data, normalize_volume, get_segy_headers
from seisviz.load_seismic import _decode_textual_header, _extension


@pytest.fixture
def npy_path(cube, tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, cube)
    return path


class TestExtensionHandling:
    def test_loads_npy(self, npy_path, cube):
        assert np.array_equal(load_seismic_data(str(npy_path)), cube)

    def test_extension_matching_is_case_insensitive(self, cube, tmp_path):
        """SURVEY.NPY used to be rejected as an unsupported format."""
        upper = tmp_path / "VOLUME.NPY"
        # np.save would append a second, lowercase .npy, so write it directly.
        with open(upper, "wb") as fh:
            np.save(fh, cube)
        assert load_seismic_data(str(upper)).shape == cube.shape

    @pytest.mark.parametrize("name", ["survey.sgy", "SURVEY.SGY", "survey.SeGy"])
    def test_segy_extensions_all_route_to_the_segy_path(self, name, tmp_path):
        assert _extension(str(tmp_path / name)) in ('.sgy', '.segy')

    def test_unsupported_extension_names_what_it_got(self, tmp_path):
        path = tmp_path / "survey.txt"
        path.write_text("not seismic")
        with pytest.raises(ValueError, match=r"Unsupported file format '\.txt'"):
            load_seismic_data(str(path))


class TestCurrentOrder:
    def test_reorders_a_npy_file_on_load(self, cube, tmp_path):
        """current_order lets a caller declare a .npy's on-disk axis order."""
        xline_major = np.transpose(cube, (1, 0, 2))  # (xline, inline, depth)
        path = tmp_path / "legacy.npy"
        np.save(path, xline_major)

        loaded = load_seismic_data(str(path), current_order='xid')
        assert np.array_equal(loaded, cube)

    def test_default_assumes_the_file_is_already_ixd(self, npy_path, cube):
        assert np.array_equal(load_seismic_data(str(npy_path)), cube)


class TestShapeValidation:
    def test_rejects_2d_arrays(self, tmp_path):
        """A 2D .npy used to load fine and fail later inside a plotting call."""
        path = tmp_path / "flat.npy"
        np.save(path, np.zeros((10, 10)))
        with pytest.raises(ValueError, match="Expected a 3D volume"):
            load_seismic_data(str(path))

    def test_error_reports_the_shape_received(self, tmp_path):
        path = tmp_path / "flat.npy"
        np.save(path, np.zeros((10, 10)))
        with pytest.raises(ValueError, match=r"\(10, 10\)"):
            load_seismic_data(str(path))


class TestAmplitudeValidation:
    def test_rejects_non_numeric_dtype(self, tmp_path):
        path = tmp_path / "text.npy"
        np.save(path, np.full((3, 3, 3), "a", dtype='<U1'))
        with pytest.raises(ValueError, match="numeric"):
            load_seismic_data(str(path))

    def test_warns_on_nan_without_failing(self, cube, tmp_path):
        volume = cube.copy()
        volume[0, 0, 0] = np.nan
        path = tmp_path / "with_nan.npy"
        np.save(path, volume)

        with pytest.warns(UserWarning, match="1 NaN/Inf"):
            loaded = load_seismic_data(str(path))
        assert np.isnan(loaded[0, 0, 0])  # not silently dropped or replaced

    def test_warns_on_inf(self, cube, tmp_path):
        volume = cube.copy()
        volume[0, 0, 0] = np.inf
        path = tmp_path / "with_inf.npy"
        np.save(path, volume)

        with pytest.warns(UserWarning, match="NaN/Inf"):
            load_seismic_data(str(path))

    def test_integer_volumes_skip_the_finite_check(self, tmp_path):
        """np.isfinite has no meaning for integer dtypes; must not raise."""
        path = tmp_path / "ints.npy"
        np.save(path, np.zeros((3, 3, 3), dtype=np.int32))
        load_seismic_data(str(path))  # no warning, no error

    def test_normalize_volume_warns_on_non_finite(self):
        volume = np.ones((3, 3, 3))
        volume[0, 0, 0] = np.inf
        with pytest.warns(UserWarning, match="NaN/Inf"):
            normalize_volume(volume)


class TestNormalization:
    def test_scales_into_minus_one_to_one(self, npy_path):
        volume = load_seismic_data(str(npy_path), normalize=True)
        assert volume.min() == pytest.approx(-1.0)
        assert volume.max() == pytest.approx(1.0)

    def test_constant_volume_does_not_divide_by_zero(self):
        with pytest.warns(UserWarning, match="constant"):
            out = normalize_volume(np.full((3, 3, 3), 7.0))
        assert np.all(out == 0)

    def test_returns_a_new_array(self):
        source = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        normalize_volume(source)
        assert source[0, 0, 0] == 0  # untouched

    def test_unknown_method_raises(self, cube):
        with pytest.raises(ValueError, match="Unsupported normalization"):
            normalize_volume(cube, method="zscore")


class TestTextualHeader:
    def test_decodes_ebcdic(self):
        """SEG-Y specifies EBCDIC; decoding it as ASCII returned quiet garbage."""
        text = "CLIENT: SEISVIZ SURVEY 2026"
        assert _decode_textual_header(text.encode("cp037")).startswith("CLIENT")

    def test_still_decodes_ascii(self):
        text = "CLIENT: SEISVIZ SURVEY 2026"
        assert _decode_textual_header(text.encode("ascii")) == text


class TestOptionalSegyio:
    def test_importing_seisviz_does_not_require_segyio(self):
        """
        segyio wheels lag new Python releases; the package must stay usable for
        .npy-only workflows without it.
        """
        import seisviz
        assert seisviz.load_seismic_data is not None

    def test_missing_segyio_names_the_extra(self, monkeypatch, tmp_path):
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "segyio":
                raise ImportError("No module named 'segyio'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match=r"seisviz\[segy\]"):
            load_seismic_data(str(tmp_path / "survey.sgy"))

    def test_headers_rejected_for_npy(self, npy_path):
        with pytest.raises(ValueError, match="No SEG-Y headers"):
            get_segy_headers(str(npy_path))

    def test_headers_rejected_for_other_types(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_segy_headers(str(tmp_path / "notes.txt"))
