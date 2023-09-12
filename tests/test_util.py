def test_progress():
    from io import StringIO

    from heracles.util import Progress

    f = StringIO()
    prog = Progress(f)
    prog.start(10, "my title")
    s = f.getvalue()
    assert s.count("\r") == 1
    assert s.count("\n") == 0
    assert "my title" in s
    assert "0/10" in s
    prog.update()
    s = f.getvalue()
    assert s.count("\r") == 2
    assert s.count("\n") == 0
    assert "1/10" in s
    prog.update(5)
    s = f.getvalue()
    assert s.count("\r") == 3
    assert s.count("\n") == 0
    assert "6/10" in s
    prog.stop()
    s = f.getvalue()
    assert s.count("\r") == 4
    assert s.count("\n") == 1
    assert "10/10" in s
