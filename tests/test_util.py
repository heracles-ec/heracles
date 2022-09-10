def test_toc_match():
    from le3_pk_wl.util import toc_match

    assert toc_match(('a',))
    assert toc_match(('a',), None, [])
    assert not toc_match(('a',), [], None)

    assert toc_match(('aa', 1, 2), [('aa', 1, 2)], None)
    assert toc_match(('aa', 1, 2), [('aa', 1, 2)], [])
    assert toc_match(('aa', 1, 2), [('aa', 1, 2)], [('ab', 1, 2)])
    assert toc_match(('aa', 1, 2), [('aa',)], None)
    assert toc_match(('aa', 1, 2), [(..., 1)], None)
    assert toc_match(('aa', 1, 2), [(..., ..., 2)], None)

    assert not toc_match(('aa', 1, 2), None, [('aa', 1, 2)])
    assert not toc_match(('aa', 1, 2), [], [('aa', 1, 2)])
    assert not toc_match(('aa', 1, 2), [('aa', 1, 2)], [('aa', 1, 2)])
    assert not toc_match(('aa', 1, 2), None, [('aa',)])
    assert not toc_match(('aa', 1, 2), None, [(..., 1)])
    assert not toc_match(('aa', 1, 2), None, [(..., ..., 2)])
