"""
The library-wide contract: volumes are ordered (inlines, xlines, depth).

These tests pin that contract at every point it is relied on, so a transpose
introduced anywhere fails loudly instead of producing a plausible picture.
"""

import numpy as np
import pytest

from seisviz import (
    reorder_volume, get_volume_range_info, plot_2D_seismic, show_random_line,
)
from seisviz.seismic_slice_plot import get_line_type
from seisviz.seismic_volume_plot import (
    plot_seismic_3d_slices,
    plot_multiple_seismic_slices_3d,
)

from conftest import N_INLINES, N_XLINES, N_DEPTH


@pytest.mark.parametrize(
    "line_type, expected_shape, expected_y, expected_x",
    [
        ('inline', (N_DEPTH, N_XLINES), 'Depth', 'Xline'),
        ('xline', (N_DEPTH, N_INLINES), 'Depth', 'Inline'),
        ('depth', (N_INLINES, N_XLINES), 'Inline', 'Xline'),
    ],
)
def test_slice_shape_and_labels_agree(cube, line_type, expected_shape,
                                      expected_y, expected_x):
    """
    imshow puts rows on y and columns on x, so the y label must name the axis
    whose length is the row count. The depth case is the regression: the y axis
    has N_INLINES rows and must be labelled 'Inline', not 'Xline'.
    """
    data, ylabel, xlabel = get_line_type(line_type, 0, cube)

    assert data.shape == expected_shape
    assert (ylabel, xlabel) == (expected_y, expected_x)

    axis_lengths = {'Xline': N_XLINES, 'Inline': N_INLINES, 'Depth': N_DEPTH}
    assert data.shape[0] == axis_lengths[ylabel]
    assert data.shape[1] == axis_lengths[xlabel]


def test_slices_come_from_the_named_axis(cube):
    """Values encode their source index, so the slice can be traced back."""
    inline_slice, _, _ = get_line_type('inline', 3, cube)
    assert np.all(inline_slice // 10_000 == 3)

    xline_slice, _, _ = get_line_type('xline', 4, cube)
    assert np.all((xline_slice % 10_000) // 100 == 4)

    depth_slice, _, _ = get_line_type('depth', 7, cube)
    assert np.all(depth_slice % 100 == 7)


def test_get_line_type_rejects_unknown_type(cube):
    with pytest.raises(ValueError, match="Invalid line_type"):
        get_line_type('timeslice', 0, cube)


@pytest.mark.parametrize(
    "domain, expected_label",
    [
        (None, 'Depth'),      # unchanged default: no geometry supplied
        ('time', 'Time'),
        ('depth', 'Depth'),
        ('nonsense', 'Depth'),  # falls back rather than raising mid-plot
    ],
)
def test_vertical_axis_label_follows_domain(cube, domain, expected_label):
    """
    domain only affects inline/xline slices' y-axis label text - it never
    changes what data is plotted, since there's no reliable way to detect or
    convert domain from the file itself (see ROADMAP.md).
    """
    for line_type in ('inline', 'xline'):
        _, ylabel, _ = get_line_type(line_type, 0, cube, domain=domain)
        assert ylabel == expected_label

    # depth slices are unaffected either way - domain describes the axis
    # being sliced away, not the map-view result
    _, ylabel, _ = get_line_type('depth', 0, cube, domain=domain)
    assert ylabel == 'Inline'


def test_plot_2D_seismic_reads_domain_from_geometry(cube):
    _, ax = plot_2D_seismic(cube, 0, line_type='inline')
    assert ax.get_ylabel() == 'Depth'  # unchanged default with no geometry

    _, ax_time = plot_2D_seismic(
        cube, 0, line_type='inline', geometry={'domain': 'time'}
    )
    assert ax_time.get_ylabel() == 'Time'


class TestTimeDepthAlias:
    """'time' is accepted wherever 'depth' is, and slices identically -
    they differ only in which word ends up in a plot title."""

    def test_time_and_depth_slice_identically(self, cube):
        by_depth, ylabel_d, xlabel_d = get_line_type('depth', 3, cube)
        by_time, ylabel_t, xlabel_t = get_line_type('time', 3, cube)
        assert np.array_equal(by_depth, by_time)
        assert (ylabel_d, xlabel_d) == (ylabel_t, xlabel_t)

    def test_check_bounds_accepts_the_alias(self, cube):
        # depth axis has N_DEPTH samples; an out-of-range index must still
        # raise under the 'time' spelling exactly as it does under 'depth'.
        # The message names the canonical axis ('depth'), not the caller's
        # spelling - bounds errors describe the axis, not the vocabulary.
        with pytest.raises(IndexError, match="depth index"):
            plot_2D_seismic(cube, N_DEPTH, line_type='time')

    def test_show_random_line_accepts_the_alias(self, cube, capsys):
        fig, ax = show_random_line(cube, line_type='time', seed=1)
        assert isinstance(ax, type(plot_2D_seismic(cube, 0)[1]))
        assert "random time slice" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "line_type, domain, expected_title_word",
        [
            ('depth', None, 'depth'),   # unchanged default
            ('time', None, 'time'),     # alias alone, no geometry: honored
            ('depth', 'time', 'time'),  # geometry overrides the literal typed
            ('time', 'depth', 'depth'), # geometry overrides the alias too
            ('inline', 'time', 'inline'),  # domain never touches inline/xline
        ],
    )
    def test_title_word_follows_domain_over_the_literal_spelling(
        self, cube, line_type, domain, expected_title_word
    ):
        geometry = {'domain': domain} if domain else None
        _, ax = plot_2D_seismic(cube, 0, line_type=line_type, geometry=geometry)
        assert ax.get_title() == f'Seismic - {expected_title_word}: 0'

    def test_rejects_unknown_line_type_mentions_time(self, cube):
        with pytest.raises(ValueError, match="'depth', or 'time'"):
            plot_2D_seismic(cube, 0, line_type='timeslice')


def test_range_info_matches_the_convention(cube):
    info = get_volume_range_info(cube)
    assert info['inline_range'] == (0, N_INLINES - 1)
    assert info['xline_range'] == (0, N_XLINES - 1)
    assert info['depth_sample_range'] == (0, N_DEPTH - 1)


def test_both_3d_views_use_the_same_axis_assignment(cube):
    """
    The two 3D functions used to disagree: one put Xline on the plot X axis,
    the other put Inline there, so the same cube looked mirrored between them.
    """
    _, ax_scatter = plot_seismic_3d_slices(cube, direction='depth', step=20)
    _, ax_surface = plot_multiple_seismic_slices_3d(cube, depth_idxs=[5])

    for ax in (ax_scatter, ax_surface):
        assert ax.get_xlabel() == 'Inline'
        assert ax.get_ylabel() == 'Xline'
        assert ax.get_zlabel() == 'Depth'


def test_3d_surface_limits_follow_the_convention(cube):
    _, ax = plot_multiple_seismic_slices_3d(cube, depth_idxs=[5])
    assert ax.get_xlim() == (0, N_INLINES)
    assert ax.get_ylim() == (0, N_XLINES)


def test_3d_views_use_the_surveys_true_proportions(cube):
    """
    mplot3d otherwise squashes every axis into an equal-sized box regardless
    of the data's real range, so a full-extent flat plane on a survey whose
    axes differ a lot (as here: N_INLINES != N_XLINES != N_DEPTH) looks
    foreshortened, as if it stopped partway through its own axis. matplotlib
    normalises set_box_aspect's magnitude internally, so only the ratios
    between axes are checked here.
    """
    expected = np.array([N_INLINES, N_XLINES, N_DEPTH], dtype=float)
    expected /= expected[0]

    for _, ax in (
        plot_seismic_3d_slices(cube, direction='depth', step=20),
        plot_multiple_seismic_slices_3d(cube, depth_idxs=[5]),
    ):
        aspect = np.array(ax.get_box_aspect())
        aspect /= aspect[0]
        assert np.allclose(aspect, expected)


def test_3d_box_aspect_is_not_overridden_on_a_supplied_axis():
    """A caller who already styled their own 3D axis keeps their aspect."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2, 3, 5))
    before = tuple(ax.get_box_aspect())

    volume = np.zeros((4, 6, 8))
    _, returned_ax = plot_multiple_seismic_slices_3d(volume, depth_idxs=[1], ax=ax)

    assert tuple(returned_ax.get_box_aspect()) == before


class TestReorderVolume:
    def test_default_xid_to_ixd(self):
        """Defaults upgrade a pre-0.3 (xline, inline, depth) cube in one call."""
        volume = np.zeros((3, 5, 7))  # xline-major, the old seisviz convention
        assert reorder_volume(volume).shape == (5, 3, 7)

    def test_roundtrip_is_identity(self, cube):
        there = reorder_volume(cube, 'ixd', 'xid')
        back = reorder_volume(there, 'xid', 'ixd')
        assert np.array_equal(back, cube)

    def test_values_move_with_their_axis(self, cube):
        swapped = reorder_volume(cube, 'ixd', 'xid')
        assert swapped[4, 3, 7] == cube[3, 4, 7]

    @pytest.mark.parametrize("current, target", [
        ('ixd', 'xiz'),   # unknown code in target
        ('ixz', 'xid'),   # unknown code in current
        ('iid', 'xid'),   # repeated code
        ('ix', 'xid'),    # too short
        ('ixdd', 'xid'),  # too long
    ])
    def test_rejects_malformed_orders(self, cube, current, target):
        with pytest.raises(ValueError, match="permutation"):
            reorder_volume(cube, current, target)

    def test_rejects_non_3d(self):
        with pytest.raises(ValueError, match="3D"):
            reorder_volume(np.zeros((4, 4)), 'xid', 'ixd')
