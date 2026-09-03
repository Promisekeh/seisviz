"""
Facies labels must be drawn in their own class colour, in every slice.

The old implementation built a colormap from only the classes present in a
slice and let imshow spread them linearly over [min, max], so a class could be
painted in a neighbour's colour with no warning.
"""

import numpy as np
import pytest
from matplotlib.colors import to_hex

from seisviz import plot_2D_seismic
from seisviz.seismic_slice_plot import get_label_color


def rendered_classes(image, color_map):
    """Map the RGBA a mappable actually draws back to class values."""
    lookup = {to_hex(colour): cls for cls, colour in color_map.items()}
    rgba = image.cmap(image.norm(image.get_array()))
    return np.vectorize(lambda *px: lookup.get(to_hex(px), None),
                        signature='(n)->()')(rgba)


@pytest.mark.parametrize("present", [
    [0, 2, 5],        # evenly spaced: the case that used to pass by luck
    [0, 1, 5],        # class 1 used to render as class 0
    [0, 1, 2, 5],     # classes 1 and 2 both used to shift down
    [1, 2],
    [5],              # single class
])
def test_every_class_renders_in_its_own_colour(label_dict, present):
    labels = np.array([present])
    cmap, norm, classes = get_label_color(labels, label_dict)

    colours = cmap(norm(labels))[0]
    lookup = {to_hex(c): cls for cls, c in label_dict['color'].items()}
    assert [lookup.get(to_hex(px)) for px in colours] == present


def test_a_class_keeps_its_colour_across_slices(label_dict):
    """
    Colours come from the full palette, not from whatever the slice contains,
    so two slices with different class content stay comparable.
    """
    cmap_a, norm_a, _ = get_label_color(np.array([[0, 1]]), label_dict)
    cmap_b, norm_b, _ = get_label_color(np.array([[1, 5]]), label_dict)

    assert np.allclose(cmap_a(norm_a(np.array([1]))),
                       cmap_b(norm_b(np.array([1]))))


def test_class_missing_from_the_palette_warns_and_stays_separate(label_dict):
    """
    An unmapped class must not be folded into a neighbour's band, or it would
    reintroduce exactly the silent mis-colouring this module exists to prevent.
    """
    labels = np.array([[0, 1, 3]])  # class 3 has no colour in label_dict

    with pytest.warns(UserWarning, match=r"Label classes \[3\]"):
        cmap, norm, classes = get_label_color(labels, label_dict)

    assert classes == [0, 1, 2, 3, 5]

    lookup = {to_hex(c): cls for cls, c in label_dict['color'].items()}
    drawn = [lookup.get(to_hex(px)) for px in cmap(norm(labels))[0]]
    assert drawn[:2] == [0, 1]      # mapped classes keep their own colours
    assert drawn[2] is None         # class 3 is the fallback, not class 2's blue
    assert to_hex(cmap(norm(np.array([3])))[0]) == to_hex('gray')


def test_classes_stay_distinct_without_a_palette(label_cube):
    """With no label_dict, each distinct class still gets its own band."""
    cmap, norm, classes = get_label_color(label_cube, None)
    assert classes == [0, 1, 2, 5]

    drawn = cmap(norm(np.array(classes)))
    assert len({to_hex(px) for px in drawn}) == len(classes)


def test_colorbar_is_ticked_and_named_by_class(cube, label_cube, label_dict):
    fig, ax = plot_2D_seismic(cube, 0, line_type='depth', label=label_cube,
                              label_dict=label_dict)

    # The label colorbar is the last axis added to the figure.
    ticks = list(fig.axes[-1].get_yticks())
    names = [t.get_text() for t in fig.axes[-1].get_yticklabels()]

    assert ticks == [0, 1, 2, 5]
    assert names == ['Class A', 'Class B', 'Class C', 'Class F']


def test_masked_classes_are_transparent(cube, label_cube, label_dict):
    fig, ax = plot_2D_seismic(cube, 0, line_type='inline', label=label_cube,
                              label_dict=label_dict, mask_labels=[0])
    overlay = ax.images[-1].get_array()
    assert np.ma.is_masked(overlay)
    assert overlay.mask.any()


def test_label_shape_must_match_the_volume(cube, label_dict):
    with pytest.raises(ValueError, match="does not match volume shape"):
        plot_2D_seismic(cube, 0, label=np.zeros((2, 2, 2)), label_dict=label_dict)
