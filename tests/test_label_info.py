"""
get_label_info: inspecting what classes are actually present in a label
volume, for building or cross-checking a label_dict before plotting.
"""

import numpy as np
import pytest

from seisviz import get_label_info


def test_classes_are_sorted_and_unique(label_cube):
    info = get_label_info(label_cube)
    assert info["classes"] == [0, 1, 2, 5]  # non-contiguous, per the fixture


def test_counts_and_proportions_are_consistent(label_cube):
    info = get_label_info(label_cube)

    assert sum(info["counts"].values()) == label_cube.size
    assert set(info["counts"]) == set(info["classes"])

    total_proportion = sum(info["proportions"].values())
    assert total_proportion == pytest.approx(1.0)

    for c in info["classes"]:
        expected = info["counts"][c] / label_cube.size
        assert info["proportions"][c] == pytest.approx(expected)


def test_without_label_dict_names_and_missing_are_empty(label_cube):
    info = get_label_info(label_cube)
    assert info["names"] == {}
    assert info["missing_from_label_dict"] is None


def test_with_label_dict_names_present_classes(label_cube, label_dict):
    info = get_label_info(label_cube, label_dict)
    assert info["names"] == {0: 'Class A', 1: 'Class B', 2: 'Class C', 5: 'Class F'}


def test_flags_classes_missing_from_the_palette(label_cube, label_dict):
    """
    Mirrors the scenario get_label_color warns about: a class present in the
    data but absent from label_dict['color'] renders in grey. This should be
    discoverable up front, not just as a warning during plotting.
    """
    incomplete = {
        'class': {0: 'Class A'},
        'color': {0: 'dodgerblue'},
    }
    info = get_label_info(label_cube, incomplete)

    assert info["missing_from_label_dict"] == [1, 2, 5]
    assert info["names"] == {0: 'Class A'}  # only the mapped class is named


def test_no_classes_missing_when_palette_is_complete(label_cube, label_dict):
    info = get_label_info(label_cube, label_dict)
    assert info["missing_from_label_dict"] == []


def test_handles_a_single_class_volume():
    labels = np.zeros((3, 4, 5), dtype=np.int64)
    info = get_label_info(labels)
    assert info["classes"] == [0]
    assert info["counts"] == {0: 60}
    assert info["proportions"] == {0: 1.0}
