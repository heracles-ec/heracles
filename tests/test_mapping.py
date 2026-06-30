import unittest.mock

import pytest


class MockCatalog:
    size = 10
    page_size = 1

    def __iter__(self):
        for i in range(0, self.size, self.page_size):
            yield {}


@pytest.mark.parametrize("parallel", [False, True])
def test_map_catalogs(parallel):
    from heracles.mapping import map_catalogs

    fields = {
        "a": unittest.mock.AsyncMock(),
        "b": unittest.mock.AsyncMock(),
        "c": unittest.mock.AsyncMock(),
    }
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    mapper = unittest.mock.Mock()

    maps = map_catalogs(fields, catalogs, mapper=mapper, parallel=parallel)

    for k in fields:
        for i in catalogs:
            fields[k].assert_any_call(
                catalogs[i], progress=unittest.mock.ANY, mapper=mapper
            )
            assert maps[k, i] is fields[k].return_value


def test_map_catalogs_match():
    from heracles.mapping import map_catalogs

    fields = {
        "a": unittest.mock.AsyncMock(),
        "b": unittest.mock.AsyncMock(),
        "c": unittest.mock.AsyncMock(),
    }
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}
    mapper = unittest.mock.Mock()

    maps = map_catalogs(fields, catalogs, mapper=mapper, include=[(..., "y")])

    assert set(maps.keys()) == {("a", "y"), ("b", "y"), ("c", "y")}

    maps = map_catalogs(fields, catalogs, mapper=mapper, exclude=[("a", ...)])

    assert set(maps.keys()) == {("b", "x"), ("b", "y"), ("c", "x"), ("c", "y")}


def test_transform(rng):
    from heracles.mapping import transform

    x = unittest.mock.Mock()
    y = unittest.mock.Mock()
    mapper = unittest.mock.Mock()

    fields = {"X": x, "Y": y}
    maps = {("X", 0): unittest.mock.Mock(), ("Y", 1): unittest.mock.Mock()}

    alms = transform(fields, maps, mapper=mapper)

    assert len(alms) == 2
    assert alms.keys() == {("X", 0), ("Y", 1)}
    assert alms["X", 0] is mapper.transform.return_value
    assert alms["Y", 1] is mapper.transform.return_value
    mapper.transform.assert_any_call(maps["X", 0], spin=x.spin)
    mapper.transform.assert_any_call(maps["Y", 1], spin=y.spin)
