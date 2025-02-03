from unittest.mock import call, patch

import numpy as np
import pytest

import heracles


def test_result(rng):
    lmax = 57

    arr = rng.random(lmax + 1)
    obj = heracles.Result(arr)
    np.testing.assert_array_equal(obj, arr)
    assert type(obj) is heracles.Result
    assert obj.axis == 0
    assert obj.ell is None
    assert obj.lower is None
    assert obj.upper is None
    assert obj.weight is None

    sliced = obj[1:]
    np.testing.assert_array_equal(sliced, arr[1:])
    assert type(sliced) is heracles.Result
    assert sliced.axis == 0
    assert sliced.ell is None
    assert sliced.lower is None
    assert sliced.upper is None
    assert sliced.weight is None

    ell = np.arange(lmax + 1)
    ellmin = ell
    ellmax = ell + 1
    weight = rng.random(lmax + 1)
    obj = heracles.Result(arr, ell, lower=ellmin, upper=ellmax, weight=weight)
    np.testing.assert_array_equal(obj, arr)
    assert type(obj) is heracles.Result
    assert obj.axis == 0
    assert obj.ell is ell
    assert obj.lower is ellmin
    assert obj.upper is ellmax
    assert obj.weight is weight

    sliced = obj[1:]
    np.testing.assert_array_equal(sliced, arr[1:])
    assert type(sliced) is heracles.Result
    assert sliced.axis == 0
    assert sliced.ell is ell
    assert sliced.lower is ellmin
    assert sliced.upper is ellmax
    assert sliced.weight is weight

    arr = rng.random((lmax + 1, 100))
    obj = heracles.Result(arr, ell, lower=ellmin, upper=ellmax, weight=weight, axis=0)
    np.testing.assert_array_equal(obj, arr)
    assert type(obj) is heracles.Result
    assert obj.ell is ell
    assert obj.lower is ellmin
    assert obj.upper is ellmax
    assert obj.weight is weight
    assert obj.axis == 0

    copy = obj.copy()
    copy[:] += 1.0
    np.testing.assert_array_equal(copy, arr + 1.0)
    assert type(copy) is heracles.Result
    assert copy.ell is ell
    assert copy.lower is ellmin
    assert copy.upper is ellmax
    assert copy.weight is weight
    assert copy.axis == 0

    view = obj.view(heracles.Result)
    np.testing.assert_array_equal(view, arr)
    assert type(view) is heracles.Result
    assert view.ell is ell
    assert view.lower is ellmin
    assert view.upper is ellmax
    assert view.weight is weight
    assert view.axis == 0

    with pytest.raises(ValueError, match="axis 1 is out of bounds"):
        heracles.Result([], axis=1)


def test_covmatrix(rng):
    lmax1 = 30
    lmax2 = 200
    ell_1 = np.arange(lmax1 + 1)
    ell_2 = np.linspace(0, lmax2, num=21)
    ellmin_1 = ell_1
    ellmax_1 = ell_1 + 1
    ellmin_2 = ell_2
    ellmax_2 = ell_2 + 10
    weight = rng.random((lmax1 + 1, lmax2 + 1))

    arr = rng.random((lmax1 + 1, lmax2 + 1))
    obj = heracles.CovMatrix(
        arr,
        ell_1,
        ell_2,
        lower_1=ellmin_1,
        upper_1=ellmax_1,
        lower_2=ellmin_2,
        upper_2=ellmax_2,
        weight=weight,
    )
    np.testing.assert_array_equal(obj, arr)
    assert type(obj) is heracles.CovMatrix
    assert obj.ell_1 is ell_1
    assert obj.ell_2 is ell_2
    assert obj.lower_1 is ellmin_1
    assert obj.upper_1 is ellmax_1
    assert obj.lower_2 is ellmin_2
    assert obj.upper_2 is ellmax_2
    assert obj.weight is weight
    assert obj.axis == (-2, -1)

    copy = obj.copy()
    copy[:] += 1.0
    np.testing.assert_array_equal(copy, arr + 1.0)
    assert type(copy) is heracles.CovMatrix
    assert copy.ell_1 is ell_1
    assert copy.ell_2 is ell_2
    assert copy.lower_1 is ellmin_1
    assert copy.upper_1 is ellmax_1
    assert copy.lower_2 is ellmin_2
    assert copy.upper_2 is ellmax_2
    assert copy.weight is weight
    assert copy.axis == (-2, -1)

    view = obj.view(heracles.CovMatrix)
    np.testing.assert_array_equal(view, arr)
    assert type(view) is heracles.CovMatrix
    assert view.ell_1 is ell_1
    assert view.ell_2 is ell_2
    assert view.lower_1 is ellmin_1
    assert view.upper_1 is ellmax_1
    assert view.lower_2 is ellmin_2
    assert view.upper_2 is ellmax_2
    assert view.weight is weight
    assert view.axis == 0


@pytest.mark.parametrize("weight", [None, "l(l+1)", "2l+1", "<rand>"])
@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (3, 1)])
def test_binned(ndim, axis, weight, rng):
    shape = rng.integers(0, 100, ndim)
    lmax = shape[axis] - 1

    data = heracles.Result(rng.standard_normal(shape), axis=axis)

    nbins = rng.integers(1, 10)
    bins = rng.integers(1, lmax + 1, nbins, endpoint=True)
    bins.sort()

    if weight == "<rand>":
        weight = rng.random(lmax + 1)

    result = heracles.binned(data, bins, weight)

    ell = np.arange(lmax + 1)

    if weight is None:
        w = np.ones_like(ell)
    elif isinstance(weight, str):
        if weight == "l(l+1)":
            w = ell * (ell + 1)
        elif weight == "2l+1":
            w = 2 * ell + 1
        else:
            raise ValueError(weight)
    else:
        w = weight

    out_shape = (*shape[:axis], nbins - 1, *shape[axis + 1 :])

    binned_data = np.zeros(out_shape)
    binned_ell = np.zeros(nbins - 1)
    binned_weight = np.zeros(nbins - 1)
    for i, (a, b) in enumerate(zip(bins, bins[1:])):
        inbin = (a <= ell) & (ell < b)
        if not np.any(inbin):
            continue
        binned_ell[i] = np.average(ell[inbin], weights=w[inbin])
        for j in np.ndindex(*shape[:axis]):
            for k in np.ndindex(*shape[axis + 1 :]):
                data_inbin = data[(*j, inbin, *k)]
                binned_data[(*j, i, *k)] = np.average(data_inbin, weights=w[inbin])
        binned_weight[i] = w[inbin].sum()

    np.testing.assert_array_almost_equal(result, binned_data)
    np.testing.assert_array_almost_equal(result.ell, binned_ell)
    np.testing.assert_array_equal(result.lower, bins[:-1])
    np.testing.assert_array_equal(result.upper, bins[1:])
    np.testing.assert_array_almost_equal(result.weight, binned_weight)


def test_binned_mapping():
    result = {
        object(): object(),
        object(): object(),
        object(): object(),
    }

    bins = object()
    weight = object()

    with patch("heracles.result.binned") as mock:
        out = heracles.binned(result, bins, weight)

        assert mock.call_count == len(out) == len(result)

    for i, key in enumerate(result):
        assert mock.call_args_list[i] == call(result[key], bins, weight)
        assert out[key] is mock.return_value
