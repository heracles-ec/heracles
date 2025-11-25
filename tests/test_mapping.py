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

    maps = map_catalogs(fields, catalogs, parallel=parallel)

    for k in fields:
        for i in catalogs:
            fields[k].assert_any_call(catalogs[i], progress=unittest.mock.ANY)
            assert maps[k, i] is fields[k].return_value


def test_map_catalogs_match():
    from heracles.mapping import map_catalogs

    fields = {
        "a": unittest.mock.AsyncMock(),
        "b": unittest.mock.AsyncMock(),
        "c": unittest.mock.AsyncMock(),
    }
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    maps = map_catalogs(fields, catalogs, include=[(..., "y")])

    assert set(maps.keys()) == {("a", "y"), ("b", "y"), ("c", "y")}

    maps = map_catalogs(fields, catalogs, exclude=[("a", ...)])

    assert set(maps.keys()) == {("b", "x"), ("b", "y"), ("c", "x"), ("c", "y")}


def test_add_metadata_to_external_map():
    import heracles.healpy
    import numpy as np

    nside = 16
    lmax = 10
    mapper = heracles.healpy.HealpixMapper(nside, lmax)

    # spin = 0 case
    m = np.ones(12 * nside**2)
    m = mapper.update_metadata(m, spin=0)

    assert m.dtype.metadata is not None
    assert m.dtype.metadata["spin"] == 0
    assert m.dtype.metadata["geometry"] == "healpix"
    assert m.dtype.metadata["kernel"] == "healpix"
    assert m.dtype.metadata["deconv"] is True
    assert m.dtype.metadata["nside"] == nside
    assert m.dtype.metadata["lmax"] == lmax

    m2 = np.ones((2, 12 * nside**2))
    m2 = mapper.update_metadata(m2, spin=2)
    assert m2.dtype.metadata is not None
    assert m2.dtype.metadata["spin"] == 2
    assert m2.dtype.metadata["geometry"] == "healpix"
    assert m2.dtype.metadata["kernel"] == "healpix"
    assert m2.dtype.metadata["deconv"] is True
    assert m2.dtype.metadata["nside"] == nside
    assert m2.dtype.metadata["lmax"] == lmax


def test_transform(rng):
    from heracles.mapping import transform

    x = unittest.mock.Mock()
    y = unittest.mock.Mock()

    fields = {"X": x, "Y": y}
    maps = {("X", 0): unittest.mock.Mock(), ("Y", 1): unittest.mock.Mock()}

    alms = transform(fields, maps)

    assert len(alms) == 2
    assert alms.keys() == {("X", 0), ("Y", 1)}
    assert alms["X", 0] is x.mapper_or_error.transform.return_value
    assert alms["Y", 1] is y.mapper_or_error.transform.return_value
