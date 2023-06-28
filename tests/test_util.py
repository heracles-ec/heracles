import pytest


def test_toc_match():
    from heracles.util import toc_match

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


def test_toc_filter():

    from heracles.util import toc_filter

    full = {('a', 'b'): 1, ('c', 'd'): 2}

    assert toc_filter(full, [('a',)]) == {('a', 'b'): 1}
    assert toc_filter(full, [(..., 'b')]) == {('a', 'b'): 1}
    assert toc_filter(full, [('a',), (..., 'd')]) == full
    assert toc_filter([full]*2, [('a',)]) == [{('a', 'b'): 1}]*2

    with pytest.raises(TypeError):
        toc_filter(object())


def test_progress():

    from io import StringIO
    from heracles.util import Progress

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
