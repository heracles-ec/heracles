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


def test_progress():

    from io import StringIO
    from le3_pk_wl.util import Progress

    f = StringIO()
    prog = Progress(f)
    prog.start(10, 'my title')
    s = f.getvalue()
    assert s.count('\r') == 1
    assert s.count('\n') == 0
    assert 'my title' in s
    assert '0/10' in s
    prog.update()
    s = f.getvalue()
    assert s.count('\r') == 2
    assert s.count('\n') == 0
    assert '1/10' in s
    prog.update(5)
    s = f.getvalue()
    assert s.count('\r') == 3
    assert s.count('\n') == 0
    assert '6/10' in s
    prog.stop()
    s = f.getvalue()
    assert s.count('\r') == 4
    assert s.count('\n') == 1
    assert '10/10' in s