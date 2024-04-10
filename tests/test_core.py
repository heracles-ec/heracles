import numpy as np
import pytest


def test_toc_match():
    from heracles.core import toc_match

    assert toc_match(("a",))
    assert toc_match(("a",), None, [])
    assert not toc_match(("a",), [], None)

    assert toc_match(("aa", 1, 2), [("aa", 1, 2)], None)
    assert toc_match(("aa", 1, 2), [("aa", 1, 2)], [])
    assert toc_match(("aa", 1, 2), [("aa", 1, 2)], [("ab", 1, 2)])
    assert toc_match(("aa", 1, 2), [("aa",)], None)
    assert toc_match(("aa", 1, 2), [(..., 1)], None)
    assert toc_match(("aa", 1, 2), [(..., ..., 2)], None)

    assert not toc_match(("aa", 1, 2), None, [("aa", 1, 2)])
    assert not toc_match(("aa", 1, 2), [], [("aa", 1, 2)])
    assert not toc_match(("aa", 1, 2), [("aa", 1, 2)], [("aa", 1, 2)])
    assert not toc_match(("aa", 1, 2), None, [("aa",)])
    assert not toc_match(("aa", 1, 2), None, [(..., 1)])
    assert not toc_match(("aa", 1, 2), None, [(..., ..., 2)])


def test_toc_filter():
    from heracles.core import toc_filter

    full = {("a", "b"): 1, ("c", "d"): 2}

    assert toc_filter(full, [("a",)]) == {("a", "b"): 1}
    assert toc_filter(full, [(..., "b")]) == {("a", "b"): 1}
    assert toc_filter(full, [("a",), (..., "d")]) == full
    assert toc_filter([full] * 2, [("a",)]) == [{("a", "b"): 1}] * 2

    with pytest.raises(TypeError):
        toc_filter(object())


def test_tocdict():
    from copy import copy, deepcopy

    from heracles.core import TocDict

    d = TocDict(
        {
            ("a", "b", 1): "ab1",
            ("a", "c", 1): "ac1",
            ("b", "c", 2): "bc2",
        },
    )

    assert d["a", "b", 1] == "ab1"
    assert d["a", "c", 1] == "ac1"
    assert d["b", "c", 2] == "bc2"
    with pytest.raises(KeyError):
        d["b", "c", 1]

    assert d["a"] == {("a", "b", 1): "ab1", ("a", "c", 1): "ac1"}
    assert d["a", ..., 1] == {("a", "b", 1): "ab1", ("a", "c", 1): "ac1"}
    assert d[..., ..., 1] == {("a", "b", 1): "ab1", ("a", "c", 1): "ac1"}
    assert d[..., "c", 1] == {("a", "c", 1): "ac1"}
    assert d[..., "c"] == {("a", "c", 1): "ac1", ("b", "c", 2): "bc2"}
    assert d[..., ..., 2] == {("b", "c", 2): "bc2"}
    with pytest.raises(KeyError):
        d["c"]

    d = TocDict(a=1, b=2)
    assert d["a"] == 1
    assert d["b"] == 2
    assert d[...] == d
    assert d[()] == d

    assert type(d.copy()) == type(d)
    assert type(copy(d)) == type(d)
    assert type(deepcopy(d)) == type(d)

    d = TocDict(a=1) | TocDict(b=2)
    assert type(d) is TocDict
    assert d == {"a": 1, "b": 2}


def test_update_metadata():
    from heracles.core import update_metadata

    other = np.dtype(float, metadata={"a": 0})

    a = np.empty(0)

    assert a.dtype.metadata is None

    update_metadata(a, other, x=1)

    assert a.dtype.metadata == {"a": 0, "x": 1}

    update_metadata(a, y=2)

    assert a.dtype.metadata == {"a": 0, "x": 1, "y": 2}

    update_metadata(a, x=3)

    assert a.dtype.metadata == {"a": 0, "x": 3, "y": 2}

    # check dtype fields are preserved

    a = np.array(
        [("Alice", 37, 56.0), ("Bob", 25, 73.0)],
        dtype=[("f0", "U10"), ("f1", "i4"), ("f2", "f4")],
    )

    a_fields_original = np.copy(a.dtype.fields)

    update_metadata(a, x=1)

    assert a.dtype.fields == a_fields_original
    assert a.dtype.metadata == {"x": 1}

    update_metadata(a, y=2)

    assert a.dtype.fields == a_fields_original
    assert a.dtype.metadata == {"x": 1, "y": 2}
