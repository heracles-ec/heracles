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
