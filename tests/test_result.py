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
    assert obj.array is arr
    assert obj.axis == (0,)
    assert obj.ell is None
    assert obj.lower is None
    assert obj.upper is None
    assert obj.weight is None

    sliced = obj[1:]
    np.testing.assert_array_equal(sliced, arr[1:])
    assert type(sliced) is np.ndarray

    ell = np.arange(lmax + 1)
    ellmin = ell
    ellmax = ell + 1
    weight = rng.random(lmax + 1)
    obj = heracles.Result(arr, ell, lower=ellmin, upper=ellmax, weight=weight)
    np.testing.assert_array_equal(obj, arr)
    assert type(obj) is heracles.Result
    assert obj.array is arr
    assert obj.axis == (0,)
    assert obj.ell is ell
    assert obj.lower is ellmin
    assert obj.upper is ellmax
    assert obj.weight is weight

    sliced = obj[1:]
    np.testing.assert_array_equal(sliced, arr[1:])
    assert type(sliced) is np.ndarray

    arr = rng.random((lmax + 1, 100))
    obj = heracles.Result(arr, ell, lower=ellmin, upper=ellmax, weight=weight, axis=0)
    np.testing.assert_array_equal(obj, arr)
    assert type(obj) is heracles.Result
    assert obj.array is arr
    assert obj.ell is ell
    assert obj.lower is ellmin
    assert obj.upper is ellmax
    assert obj.weight is weight
    assert obj.axis == (0,)

    with pytest.raises(ValueError, match="axis 1 is out of bounds"):
        heracles.Result(np.array([]), axis=1)


def test_result_2d(rng):
    lmax1 = 30
    lmax2 = 200
    ell_1 = np.arange(lmax1 + 1)
    ell_2 = np.linspace(0, lmax2, num=21)
    ellmin_1 = ell_1
    ellmax_1 = ell_1 + 1
    ellmin_2 = ell_2
    ellmax_2 = ell_2 + 10
    weight_1 = rng.random((lmax1 + 1, lmax2 + 1))
    weight_2 = rng.random((lmax1 + 1, lmax2 + 1))

    arr = rng.random((5, lmax1 + 1, lmax2 + 1))
    obj = heracles.Result(
        arr,
        (ell_1, ell_2),
        lower=(ellmin_1, ellmin_2),
        upper=(ellmax_1, ellmax_2),
        weight=(weight_1, weight_2),
    )
    np.testing.assert_array_equal(obj, arr)
    assert obj.axis == (1, 2)
    assert obj.ell == (ell_1, ell_2)
    assert obj.lower == (ellmin_1, ellmin_2)
    assert obj.upper == (ellmax_1, ellmax_2)
    assert obj.weight == (weight_1, weight_2)


@pytest.mark.parametrize("weight", [None]) # "l(l+1)", "2l+1", "<rand>"])
# cl['POS', 'POS'], mms['POS', 'POS'], cl['POS', 'SHE'], mms['SHE', 'SHE'], cl['SHE', 'SHE']
@pytest.mark.parametrize(
    "ndim,axes", [(1, (0))] #, (2, (0,)), (2, (1,)), (3, (1,)), (3, (0, 1))]
)
def test_binned(ndim, axes, weight, rng):

    def norm(a, b):
        """divide a by b if a is nonzero"""
        out = np.zeros(np.broadcast(a, b).shape)
        return np.divide(a, b, where=(a != 0), out=out)

    shape = rng.integers(1, 100, ndim)
    data = heracles.Result(rng.standard_normal(shape), axis=axes)

    bins = []
    ells = []
    weights = []
    axes = heracles.result.normalize_result_axis(axes, data, data.ell)
    for axis in axes:
        lmax = shape[axis] - 1
        ell = np.arange(lmax + 1)
        nbins = rng.integers(1, 10)
        b = rng.integers(1, lmax + 1, nbins, endpoint=True)
        b.sort()
        if weight == "l(l+1)":
            w = ell * (ell + 1)
        elif weight == "2l+1":
            w = (2 * ell + 1)
        elif weight is None:
            w = np.ones_like(ell)
        elif weight == "<rand>":
            w = rng.random(lmax + 1)
        bins.append(b)
        ells.append(ell)
        weights.append(w)

    ells = tuple(ells)
    bins = tuple(bins)
    weights = tuple(weights)
    print(bins)
    print(ells)
    print(weights)

    result = heracles.binned(data, bins, weights)

    # make a copy of the array to apply the binning
    out = np.copy(result.array)

    # this will hold the binned ells and weigths
    binned_ell = ()
    binned_weight = ()

    # apply binning over each axis
    for axis, ell, w, b in zip(axes, ells, weights, bins):
        # number of bins for this axis
        m = b.size

        # get the bin index for each ell
        index = np.digitize(ell, b)

        # compute the binned weight
        wb = np.bincount(index, weights=w, minlength=m)[1:m]

        # compute the binned ell
        ellb = norm(np.bincount(index, w * ell, m)[1:m], wb)

        # output shape, axis is turned into size m
        shape = out.shape[:axis] + (m - 1,) + out.shape[axis + 1 :]

        # create an empty binned output array
        tmp = np.empty(shape)
        # compute the binned result axis by axis
        for before in np.ndindex(shape[:axis]):
            for after in np.ndindex(shape[axis + 1 :]):
                k = (*before, slice(None), *after)
                tmp[k] = norm(np.bincount(index, w * out[k], m)[1:m], wb)

        # array is now binned over axis
        out = tmp

        # store outputs
        binned_ell += (ellb,)
        binned_weight += (wb,)

    # compute bin edges
    binned_lower = tuple(b[:-1] for b in bins)
    binned_upper = tuple(b[1:] for b in bins)

    np.testing.assert_array_almost_equal(result, out)
    np.testing.assert_array_almost_equal(result.ell, binned_ell)
    np.testing.assert_array_almost_equal(result.weight, binned_weight)
    np.testing.assert_array_equal(result.lower, binned_lower)
    np.testing.assert_array_equal(result.upper, binned_upper)


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


def test_binned_metadata():
    md = {"test": object()}

    result = np.zeros(3, dtype=np.dtype(float, metadata=md))
    assert result.dtype.metadata == md

    binned = heracles.binned(result, np.array([0, 1, 2]))
    assert binned.dtype.metadata == md
