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


def bin1(data, bins, weight, axis):
    """bin data over a single axis"""

    ell = np.arange(data.shape[axis])

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

    out_shape = (*data.shape[:axis], bins.size - 1, *data.shape[axis + 1 :])

    binned_data = np.zeros(out_shape)
    binned_ell = np.zeros(bins.size - 1)
    binned_weight = np.zeros(bins.size - 1)
    for i, (a, b) in enumerate(zip(bins, bins[1:])):
        inbin = (a <= ell) & (ell < b)
        if not np.any(inbin):
            continue
        binned_ell[i] = np.average(ell[inbin], weights=w[inbin])
        for j in np.ndindex(*data.shape[:axis]):
            for k in np.ndindex(*data.shape[axis + 1 :]):
                data_inbin = data[(*j, inbin, *k)]
                binned_data[(*j, i, *k)] = np.average(data_inbin, weights=w[inbin])
        binned_weight[i] = w[inbin].sum()

    return binned_data, binned_ell, binned_weight


@pytest.mark.parametrize("weight", [None, "l(l+1)", "2l+1", "<rand>"])
@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (3, 1)])
def test_binned(ndim, axis, weight, rng):
    shape = rng.integers(1, 100, ndim)
    lmax = shape[axis] - 1

    data = heracles.Result(rng.standard_normal(shape), axis=axis)

    nbins = rng.integers(1, 10)
    bins = rng.integers(1, lmax + 1, nbins, endpoint=True)
    bins.sort()

    if weight == "<rand>":
        weight = rng.random(lmax + 1)

    result = heracles.binned(data, bins, weight)

    binned_data, binned_ell, binned_weight = bin1(data, bins, weight, axis)

    np.testing.assert_array_almost_equal(result, binned_data)
    np.testing.assert_array_almost_equal(result.ell, binned_ell)
    np.testing.assert_array_equal(result.lower, bins[:-1])
    np.testing.assert_array_equal(result.upper, bins[1:])
    np.testing.assert_array_almost_equal(result.weight, binned_weight)


def test_binned_2d(rng):
    ndim = 3
    axes = (0, 2)
    weight = ("2l+1", "l(l+1)")

    shape = rng.integers(1, 100, ndim)
    data = heracles.Result(rng.standard_normal(shape), axis=axes)

    bins = tuple(
        np.sort(rng.integers(1, shape[axis], rng.integers(1, 10), endpoint=True))
        for axis in axes
    )

    result = heracles.binned(data, bins, weight)

    binned = data.array
    for i, axis in enumerate(axes):
        binned, binned_ell, binned_weight = bin1(binned, bins[i], weight[i], axis)
        np.testing.assert_array_almost_equal(result.ell[i], binned_ell)
        np.testing.assert_array_equal(result.lower[i], bins[i][:-1])
        np.testing.assert_array_equal(result.upper[i], bins[i][1:])
        np.testing.assert_array_almost_equal(result.weight[i], binned_weight)
    np.testing.assert_array_almost_equal(result, binned)


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


def trunc1(data, ell_max, axis):
    """truncate data over a single axis"""
    ell = np.arange(data.shape[axis])
    mask = ell <= ell_max
    out = np.take(data, np.where(mask)[0], axis=axis)
    return out, ell[mask], np.ones_like(ell[mask])


@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (3, 1)])
def test_truncated(ndim, axis, rng):
    shape = rng.integers(5, 50, ndim)
    lmax = shape[axis] - 1
    ell_max = rng.integers(0, lmax)

    data = heracles.Result(rng.standard_normal(shape), axis=axis)
    result = heracles.truncated(data, ell_max)

    trunc_data, trunc_ell, trunc_weight = trunc1(data, ell_max, axis)

    np.testing.assert_array_almost_equal(result, trunc_data)
    np.testing.assert_array_equal(result.ell, trunc_ell)
    np.testing.assert_array_equal(result.weight, trunc_weight)


def test_truncated_2d(rng):
    ndim = 3
    axes = (0, 2)
    shape = rng.integers(5, 50, ndim)
    data = heracles.Result(rng.standard_normal(shape), axis=axes)

    ell_max = tuple(rng.integers(0, shape[ax]) for ax in axes)
    result = heracles.truncated(data, ell_max)

    trunc = data.array
    for i, axis in enumerate(axes):
        trunc, trunc_ell, trunc_weight = trunc1(trunc, ell_max[i], axis)
        np.testing.assert_array_equal(result.ell[i], trunc_ell)
        np.testing.assert_array_equal(result.weight[i], trunc_weight)
    np.testing.assert_array_almost_equal(result, trunc)


def test_truncated_mapping():
    result = {
        object(): object(),
        object(): object(),
        object(): object(),
    }
    ell_max = object()

    with patch("heracles.result.truncated") as mock:
        out = heracles.truncated(result, ell_max)
        assert mock.call_count == len(out) == len(result)

    for i, key in enumerate(result):
        assert mock.call_args_list[i] == call(result[key], ell_max)
        assert out[key] is mock.return_value


def test_truncated_metadata():
    from heracles.result import Result

    md = {"test": object()}
    result = np.zeros(3, dtype=np.dtype(float, metadata=md))
    result = Result(result, spin=0)
    assert result.dtype.metadata == md
    truncated = heracles.truncated(result, 1)
    assert truncated.dtype.metadata == md
