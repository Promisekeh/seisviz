"""
Amplitude scaling, the figure-returning contract, and input validation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from seisviz import (
    plot_2D_seismic,
    show_random_line,
    plot_seismic_3d_slices,
    plot_multiple_seismic_slices_3d,
    save_seismic_slice,
)
from seisviz.seismic_slice_plot import get_amplitude_limits, is_diverging


class TestAmplitudeLimits:
    def test_diverging_colormaps_are_centred_on_zero(self, asymmetric_cube):
        """
        'seismic' puts white at the midpoint of the range. Autoscaling a
        [-1, 5] slice drew that neutral colour at +2, misassigning polarity
        across the whole section.
        """
        vmin, vmax = get_amplitude_limits(asymmetric_cube, cmap='seismic')
        assert vmin == pytest.approx(-vmax)
        assert vmin < 0 < vmax

    def test_sequential_colormaps_use_the_full_range(self, asymmetric_cube):
        vmin, vmax = get_amplitude_limits(asymmetric_cube, cmap='viridis')
        assert vmin == pytest.approx(asymmetric_cube.min())
        assert vmax == pytest.approx(asymmetric_cube.max())

    def test_percentile_clips_outliers(self):
        data = np.concatenate([np.full(1000, 0.1), np.array([500.0])])
        clipped, _ = get_amplitude_limits(data, cmap='seismic', clip_percentile=99.0)
        full, _ = get_amplitude_limits(data, cmap='seismic', clip_percentile=None)
        assert abs(clipped) < abs(full)

    def test_explicit_limits_win(self, asymmetric_cube):
        assert get_amplitude_limits(asymmetric_cube, 'seismic', vmin=-2, vmax=3) == (-2, 3)

    def test_constant_data_does_not_collapse_the_scale(self):
        vmin, vmax = get_amplitude_limits(np.zeros((4, 4)), cmap='seismic')
        assert vmin < vmax

    @pytest.mark.parametrize("cmap, expected", [
        ('seismic', True), ('RdBu_r', True), ('coolwarm', True),
        ('viridis', False), ('gray', False),
    ])
    def test_diverging_detection(self, cmap, expected):
        assert is_diverging(cmap) is expected

    def test_the_plot_uses_symmetric_limits(self, asymmetric_cube):
        _, ax = plot_2D_seismic(asymmetric_cube, 0, line_type='inline')
        vmin, vmax = ax.images[0].get_clim()
        assert vmin == pytest.approx(-vmax)


class TestFigureContract:
    def test_2d_returns_fig_and_ax(self, cube):
        fig, ax = plot_2D_seismic(cube, 0, line_type='inline')
        assert isinstance(fig, Figure) and isinstance(ax, Axes)

    def test_side_by_side_returns_both_axes(self, cube, label_cube, label_dict):
        fig, axes = plot_2D_seismic(cube, 0, line_type='depth', label=label_cube,
                                    label_dict=label_dict,
                                    display_mode='side_by_side')
        assert len(axes) == 2
        assert all(isinstance(a, Axes) for a in axes)

    def test_3d_functions_return_fig_and_ax(self, cube):
        for fig, ax in (plot_seismic_3d_slices(cube, step=20),
                        plot_multiple_seismic_slices_3d(cube, depth_idxs=[5])):
            assert isinstance(fig, Figure)
            assert hasattr(ax, 'get_zlabel')

    def test_draws_into_a_supplied_axis(self, cube):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_2D_seismic(cube, 0, ax=ax)
        assert returned_ax is ax and returned_fig is fig

    def test_nothing_is_shown_by_default(self, cube, monkeypatch):
        called = []
        monkeypatch.setattr(plt, 'show', lambda *a, **k: called.append(1))
        plot_2D_seismic(cube, 0)
        assert not called
        plot_2D_seismic(cube, 0, show=True)
        assert called

    def test_random_line_is_reproducible_with_a_seed(self, cube, capsys):
        plot_2D_seismic  # noqa: B018 - keep the import meaningful
        show_random_line(cube, line_type='inline', seed=7)
        first = capsys.readouterr().out
        show_random_line(cube, line_type='inline', seed=7)
        assert capsys.readouterr().out == first


class TestValidation:
    def test_unknown_display_mode_raises(self, cube):
        """A typo used to fall through every branch and draw nothing at all."""
        with pytest.raises(ValueError, match="Invalid display_mode"):
            plot_2D_seismic(cube, 0, display_mode='side-by-side')

    def test_unknown_line_type_raises(self, cube):
        with pytest.raises(ValueError, match="Invalid line_type"):
            plot_2D_seismic(cube, 0, line_type='timeslice')

    @pytest.mark.parametrize("line_type, bad", [
        ('xline', 99), ('inline', 99), ('depth', 999), ('inline', -1),
    ])
    def test_out_of_range_index_raises(self, cube, line_type, bad):
        with pytest.raises(IndexError, match="out of range"):
            plot_2D_seismic(cube, bad, line_type=line_type)

    def test_unknown_direction_raises(self, cube):
        with pytest.raises(ValueError, match="Invalid direction"):
            plot_seismic_3d_slices(cube, direction='timeslice')


class TestIndexArguments:
    def test_numpy_arrays_are_accepted(self, cube):
        """`if depth_idxs:` used to raise on any array with >1 element."""
        fig, ax = plot_multiple_seismic_slices_3d(cube, depth_idxs=np.arange(0, 30, 10))
        assert isinstance(fig, Figure)

    def test_out_of_range_indices_warn_rather_than_vanish(self, cube):
        with pytest.warns(UserWarning, match="out of range and skipped"):
            plot_multiple_seismic_slices_3d(cube, depth_idxs=[5, 9999])

    def test_scalar_is_accepted(self, cube):
        fig, _ = plot_multiple_seismic_slices_3d(cube, xline_idxs=2)
        assert isinstance(fig, Figure)


class TestSaving:
    def test_creates_missing_directories(self, cube, tmp_path):
        """The docstring's own example path used to raise FileNotFoundError."""
        target = tmp_path / "output" / "nested" / "inline_2.png"
        written = save_seismic_slice(cube, 2, line_type='inline',
                                     output_path=str(target))
        assert target.exists() and target.stat().st_size > 0
        assert written == str(target)

    def test_saves_into_the_working_directory(self, cube, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_seismic_slice(cube, 0, output_path='slice.png')
        assert (tmp_path / 'slice.png').exists()
