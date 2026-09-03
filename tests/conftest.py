import matplotlib

# Must be selected before pyplot is imported anywhere, so the suite runs headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test opens at least one figure; close them so the suite doesn't
    trip matplotlib's open-figure-count warning."""
    yield
    plt.close('all')

# Three distinct axis lengths. Any accidental transpose changes a shape, so a
# single fixture catches axis-ordering bugs in the loaders, the 2D slicers and
# both 3D views.
N_INLINES, N_XLINES, N_DEPTH = 5, 9, 40


@pytest.fixture
def cube():
    """
    Cube ordered (inlines, xlines, depth) whose values encode their own index.

    volume[i, x, d] == i * 10_000 + x * 100 + d, so a slice can be checked
    against the exact positions it should have come from.
    """
    i = np.arange(N_INLINES)[:, None, None]
    x = np.arange(N_XLINES)[None, :, None]
    d = np.arange(N_DEPTH)[None, None, :]
    return (i * 10_000 + x * 100 + d).astype(np.float32)


@pytest.fixture
def asymmetric_cube():
    """Amplitudes deliberately not symmetric about zero: range [-1, 5]."""
    rng = np.random.default_rng(0)
    volume = rng.uniform(-1.0, 5.0, size=(N_INLINES, N_XLINES, N_DEPTH))
    volume[0, 0, 0] = -1.0
    volume[0, 0, 1] = 5.0
    return volume.astype(np.float32)


@pytest.fixture
def label_cube():
    """
    Labels using non-contiguous classes {0, 1, 2, 5}.

    Even spacing is what made the old rank-based colormap look correct; these
    gaps are what exposed it.
    """
    labels = np.zeros((N_INLINES, N_XLINES, N_DEPTH), dtype=np.int64)
    labels[:, :, 10:20] = 1
    labels[:, :, 20:30] = 2
    labels[:, :, 30:] = 5
    return labels


@pytest.fixture
def label_dict():
    return {
        'class': {0: 'Class A', 1: 'Class B', 2: 'Class C', 5: 'Class F'},
        'color': {
            0: 'dodgerblue',
            1: 'darkorange',
            2: 'skyblue',
            5: 'hotpink',
        },
    }
