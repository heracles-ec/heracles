import itertools

import matplotlib.pyplot as plt
import numpy as np

from heracles.plot import postage_stamps


def test_postage_stamps():
    cl = [1, 2, 3, 4]

    plot = {
        ("E", "E", 0, 0): cl,
        ("B", "B", 0, 0): cl,
        ("E", "E", 0, 1): cl,
        ("B", "B", 0, 1): cl,
        ("E", "E", 1, 1): cl,
        ("B", "B", 1, 1): cl,
    }

    transpose = {
        ("P", "P", 0, 0): cl,
        ("P", "P", 0, 1): cl,
        ("P", "P", 1, 1): cl,
    }

    fig = postage_stamps(plot, transpose, shift_transpose=2, hatch_empty=True)

    assert len(fig.axes) == 2 * 4

    axes = np.reshape(fig.axes, (2, 4))
    # i: rows, j: columns - 2 + shift
    for i, j in itertools.product(range(2), range(4)):
        lines = axes[i, j].get_lines()
        if i >= j:
            assert len(lines) == 3  # E, B and axhline in lower
        elif i + 1 == j:
            assert len(lines) == 0  # empty diagonal
        else:
            assert len(lines) == 2  # P and axhline in upper

    plt.close()
