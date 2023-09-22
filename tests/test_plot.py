def test_postage_stamps():
    import matplotlib.pyplot as plt
    import numpy as np

    from heracles.plot import postage_stamps

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

    fig = postage_stamps(plot, transpose, trxshift=2, tryshift=3, hatch_empty=True)

    assert len(fig.axes) == 5 * 4

    axes = np.reshape(fig.axes, (5, 4))

    for i in range(5):  # rows: 2 + tryshift
        for j in range(4):  # columns: 2 + trxshift
            lines = axes[i, j].get_lines()
            if i - j > 2:
                assert len(lines) == 2  # E, B in lower
            elif i - j < -1:
                assert len(lines) == 1  # P in upper
            else:
                assert len(lines) == 0  # empty diagonal

    plt.close()
