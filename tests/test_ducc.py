import importlib.util

import numpy as np
import numpy.testing as npt
import pytest

HAVE_DUCC = importlib.util.find_spec("ducc0") is not None

skipif_no_ducc = pytest.mark.skipif(not HAVE_DUCC, reason="test requires ducc")


@skipif_no_ducc
def test_resample():
    from heracles.ducc import DiscreteMapper

    lmax = 200

    alm = np.concatenate(
        [np.arange(m, lmax + 1) for m in range(lmax + 1)],
        dtype=complex,
    )

    out = DiscreteMapper(lmax).resample(alm)

    npt.assert_array_equal(out, alm)

    lmax_out = lmax // 2
    out = DiscreteMapper(lmax_out).resample(alm)

    assert out.shape == ((lmax_out + 1) * (lmax_out + 2) // 2,)
    i = j = 0
    for m in range(lmax_out + 1):
        i, j = j, j + lmax_out - m + 1
        npt.assert_array_equal(out[i:j], np.arange(m, lmax_out + 1))

    lmax_out = lmax * 2
    out = DiscreteMapper(lmax_out).resample(alm)

    assert out.shape == ((lmax_out + 1) * (lmax_out + 2) // 2,)
    i = j = 0
    for m in range(lmax + 1):
        i, j = j, j + lmax_out - m + 1
        expected = np.pad(np.arange(m, lmax + 1), (0, lmax_out - lmax))
        npt.assert_array_equal(out[i:j], expected)
    npt.assert_array_equal(out[j:], 0.0)
