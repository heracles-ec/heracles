import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def catalog(rng):
    from heracles.catalog import CatalogBase, CatalogPage

    # fix a set of rows to be returned for testing
    size = 100
    x = rng.random(size)
    y = rng.random(size)
    z = rng.random(size)

    class TestCatalog(CatalogBase):
        SIZE = size
        DATA = dict(x=x, y=y, z=z)

        def __init__(self):
            super().__init__()

        def _names(self):
            return list(self.DATA.keys())

        def _size(self, selection):
            return self.SIZE

        def _join(self, *where):
            return where

        # implement abstract method
        def _pages(self, selection):
            size = self.SIZE
            page_size = self.page_size
            for i in range(0, size, page_size):
                page = {k: v[i : i + page_size] for k, v in self.DATA.items()}
                yield CatalogPage(page)

    return TestCatalog()


def test_catalog_page():
    from heracles.catalog import CatalogPage

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])

    page = CatalogPage({"a": a, "b": b})

    # test basic behaviour
    assert len(page) == 2
    npt.assert_array_equal(page["a"], a)
    npt.assert_array_equal(page["b"], b)
    npt.assert_array_equal(page["a", "b"], [a, b])
    npt.assert_array_equal(page[["a", "b"]], [a, b])
    npt.assert_array_equal(page[["a", "-b"]], [a, -b])

    # test names attribute
    assert page.names == ["a", "b"]

    # test size attribue
    assert page.size == 4

    # test data attribute, which is a readonly view
    data = page.data
    assert list(data.keys()) == ["a", "b"]
    npt.assert_array_equal(list(data.values()), [a, b])
    with pytest.raises(TypeError):
        data["a"] = b

    # test iterator
    assert [_ for _ in page] == ["a", "b"]

    # test copy method
    copy = page.copy()
    assert copy is not page
    assert copy.size == page.size
    assert copy.data == page.data

    # test copy magic
    copy = page.__copy__()
    assert copy is not page
    assert copy.size == page.size
    assert copy.data == page.data

    # test delete method
    page.delete([1, 2])
    assert len(page) == 2
    np.testing.assert_array_equal(page["a"], [1.0, 4.0])
    np.testing.assert_array_equal(page["b"], [5.0, 8.0])
    assert page.size == 2

    # test exception if column does not exist
    with pytest.raises(KeyError):
        page["c"]
    with pytest.raises(KeyError):
        page["a", "b", "c"]

    # test exception if rows have unequal size
    with pytest.raises(ValueError):
        CatalogPage({"a": [1, 2, 3], "b": [1, 2]})


def test_catalog_page_get():
    from heracles.catalog import CatalogPage

    a = [np.nan, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]

    page = CatalogPage({"a": a, "b": b})
    with pytest.raises(ValueError, match='column "a"'):
        page.get("a")
    npt.assert_array_equal(page.get("b"), b)
    with pytest.raises(ValueError, match='column "a"'):
        page.get("a", "b")

    a[0] = 1.0
    b[1] = np.nan

    page = CatalogPage({"a": a, "b": b})
    npt.assert_array_equal(page.get("a"), a)
    with pytest.raises(ValueError, match='column "b"'):
        page.get("b")
    with pytest.raises(ValueError, match='column "b"'):
        page.get("a", "b")

    b[1] = 6.0

    page = CatalogPage({"a": a, "b": b})
    npt.assert_array_equal(page.get("a"), a)
    npt.assert_array_equal(page.get("b"), b)
    npt.assert_array_equal(page.get("a", "b"), [a, b])


def test_catalog_page_immutable():
    from heracles.catalog import CatalogPage

    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]

    page = CatalogPage({"a": a, "b": b})

    with pytest.raises(ValueError):
        page["a"][0] = 0.0


def test_catalog_base(catalog):
    from heracles.catalog import Catalog, CatalogBase

    # ABC cannot be instantiated directly
    with pytest.raises(TypeError):
        CatalogBase()

    # fixture has tested concrete implementation
    assert isinstance(catalog, CatalogBase)

    # check that CatalogBase implements the Catalog protocol
    assert isinstance(catalog, Catalog)


def test_catalog_base_properties(catalog):
    from heracles.catalog import CatalogBase

    assert catalog.size == catalog.SIZE
    assert catalog.names == list(catalog.DATA.keys())

    assert catalog.base is None
    assert catalog.selection is None

    assert catalog.page_size == CatalogBase.default_page_size
    catalog.page_size = 1
    assert catalog.page_size == 1
    catalog.page_size = CatalogBase.default_page_size
    assert catalog.page_size == CatalogBase.default_page_size

    filt = object()
    assert catalog.filters == []
    catalog.add_filter(filt)
    assert catalog.filters == [filt]
    catalog.filters = []
    assert catalog.filters == []

    assert catalog.label is None
    catalog.label = "label 123"
    assert catalog.label == "label 123"

    assert catalog.metadata == {"catalog": catalog.label}

    v = object()
    assert catalog.visibility is None
    catalog.visibility = v
    assert catalog.visibility is v
    catalog.visibility = None
    assert catalog.visibility is None


def test_catalog_base_pagination(catalog):
    size = catalog.SIZE

    for page_size in [size, size // 2]:
        catalog.page_size = page_size
        for i, page in enumerate(catalog):
            assert page.size == page_size
            for k, v in catalog.DATA.items():
                vp = v[i * page_size : (i + 1) * page_size]
                npt.assert_array_equal(page[k], vp)
        assert i * page_size + page.size == size


def test_catalog_base_copy():
    from heracles.catalog import CatalogBase

    class TestCatalog(CatalogBase):
        def __init__(self):
            super().__init__()
            self._visibility = object()

        def _names(self):
            return []

        def _size(self, selection):
            return 0

        def _join(self, *where):
            return where

        def _pages(self, selection):
            return iter([])

    catalog = TestCatalog()

    copied = catalog.__copy__()

    assert isinstance(copied, TestCatalog)
    assert copied is not catalog
    assert copied.__dict__ == catalog.__dict__
    assert copied.visibility is catalog.visibility
    assert copied.filters is not catalog.filters


def test_catalog_view(catalog):
    from heracles.catalog import Catalog

    catalog.label = "label 123"
    catalog.visibility = cvis = object()

    where = object()

    view = catalog[where]

    assert isinstance(view, Catalog)

    assert view is not catalog
    assert catalog.base is None
    assert catalog.selection is None
    assert view.base is catalog
    assert view.metadata == {"catalog": view.label}
    assert view.label == "label 123"
    assert view.selection is where
    assert view.visibility is catalog.visibility

    with pytest.raises(AttributeError):
        view.label = "different label"

    view.visibility = vvis = object()

    assert view.visibility is not catalog.visibility
    assert view.visibility is vvis
    assert catalog.visibility is cvis

    view = catalog.where(where, vvis)

    assert view is not catalog
    assert catalog.base is None
    assert catalog.selection is None
    assert view.base is catalog
    assert view.selection is where
    assert view.visibility is not catalog.visibility
    assert view.visibility is vvis
    assert catalog.visibility is cvis

    sub = object()

    subview = view[sub]

    assert subview.selection == (where, sub)


def test_invalid_value_filter(catalog):
    from heracles.catalog import InvalidValueFilter

    catalog.DATA["x"][0] = np.nan
    catalog.DATA["y"][1] = np.nan

    page = next(iter(catalog))
    with pytest.raises(ValueError):
        page.get("x")
    with pytest.raises(ValueError):
        page.get("y")

    filt = InvalidValueFilter("x", "y")

    assert repr(filt) == "InvalidValueFilter('x', 'y', weight=None, warn=True)"

    catalog.add_filter(filt)

    with pytest.warns(UserWarning):
        page = next(iter(catalog))
    assert page.size == catalog.SIZE - 2
    for k, v in catalog.DATA.items():
        npt.assert_array_equal(page.get(k), v[2:])


def test_footprint_filter(catalog, rng):
    from healpy import ang2pix

    from heracles.catalog import FootprintFilter

    # footprint for northern hemisphere
    nside = 8
    m = np.round(rng.random(12 * nside**2))

    # replace x and y in catalog with lon and lat
    catalog.DATA["x"] = lon = rng.uniform(-180, 180, size=catalog.SIZE)
    catalog.DATA["y"] = lat = np.degrees(
        np.arcsin(rng.uniform(-1, 1, size=catalog.SIZE)),
    )

    filt = FootprintFilter(m, "x", "y")

    assert repr(filt) == "FootprintFilter(..., 'x', 'y')"

    catalog.add_filter(filt)

    good = m[ang2pix(nside, lon, lat, lonlat=True)] != 0
    assert good.sum() != good.size

    page = next(iter(catalog))
    assert page.size == good.sum()
    for k, v in catalog.DATA.items():
        np.testing.assert_array_equal(page[k], v[good])


def test_array_catalog(rng):
    from heracles.catalog import ArrayCatalog, Catalog

    arr = np.empty(100, [("lon", float), ("lat", float), ("x", float), ("y", float)])
    for name in arr.dtype.names:
        arr[name] = rng.random(len(arr))

    catalog = ArrayCatalog(arr)

    assert isinstance(catalog, Catalog)

    assert catalog.size == len(arr)
    assert catalog.names == arr.dtype.names

    catalog.page_size = len(arr)

    for i, page in enumerate(catalog):
        assert page.size == 100
        assert len(page) == 4
        assert page.names == list(arr.dtype.names)
        for k in arr.dtype.names:
            npt.assert_array_equal(page[k], arr[k])
    assert i == 0

    sel1 = arr["x"] > 0.5
    sel2 = arr["y"] < 0.5
    view = catalog[sel1, sel2]

    for i, page in enumerate(view):
        assert page.size == len(arr[sel1 & sel2])
        assert len(page) == 4
        assert page.names == list(arr.dtype.names)
        for k in arr.dtype.names:
            npt.assert_array_equal(page[k], arr[sel1 & sel2][k])
    assert i == 0

    copied = catalog.__copy__()

    assert isinstance(copied, ArrayCatalog)
    assert copied is not catalog
    assert copied.__dict__ == catalog.__dict__


def test_fits_catalog(rng, tmp_path):
    import fitsio

    from heracles.catalog import Catalog
    from heracles.catalog.fits import FitsCatalog

    size = 100
    ra = rng.uniform(-180, 180, size=size)
    dec = rng.uniform(-90, 90, size=size)

    path = tmp_path / "catalog.fits"

    with fitsio.FITS(path, "rw") as fits:
        fits.write(None)
        fits.write_table([ra, dec], names=["RA", "DEC"], extname="MYEXT")

    catalog = FitsCatalog(path)

    assert isinstance(catalog, Catalog)

    assert catalog.size == size
    assert catalog.names == ["RA", "DEC"]

    page = next(iter(catalog))
    assert page.size == size
    assert len(page) == 2
    np.testing.assert_array_equal(page["RA"], ra)
    np.testing.assert_array_equal(page["DEC"], dec)

    view = catalog["RA > 0"]

    sel = np.where(ra > 0)[0]

    assert view.size == catalog.size
    assert view.names == ["RA", "DEC"]

    page = next(iter(view))
    assert page.size == len(sel)
    assert len(page) == 2
    np.testing.assert_array_equal(page["RA"], ra[sel])
    np.testing.assert_array_equal(page["DEC"], dec[sel])

    vview = view["DEC < 0"]

    sel = np.where((ra > 0) & (dec < 0))[0]

    assert vview.size == catalog.size
    assert vview.names == ["RA", "DEC"]

    page = next(iter(vview))
    assert page.size == len(sel)
    assert len(page) == 2
    np.testing.assert_array_equal(page["RA"], ra[sel])
    np.testing.assert_array_equal(page["DEC"], dec[sel])

    copied = catalog.__copy__()

    assert isinstance(copied, FitsCatalog)
    assert copied is not catalog
    assert copied._path == catalog._path
    assert copied._columns == catalog._columns
    assert copied._ext == catalog._ext


def test_fits_catalog_caching(rng, tmp_path):
    import gc

    import fitsio

    from heracles.catalog.fits import FitsCatalog

    size = 100
    ra = rng.uniform(-180, 180, size=size)
    dec = rng.uniform(-90, 90, size=size)

    path = tmp_path / "cached.fits"

    with fitsio.FITS(path, "rw") as fits:
        fits.write(None)
        fits.write_table([ra, dec], names=["RA", "DEC"], extname="MYEXT")

    catalog = FitsCatalog(path)

    hdu = catalog.hdu()
    assert catalog.hdu() is hdu

    assert catalog._hdu() is not None

    _fits = hdu._FITS

    assert _fits.filename() == str(path)

    del hdu
    gc.collect()

    assert catalog._hdu() is None

    with pytest.raises(ValueError):
        _fits.filename()

    catalog.page_size = size // 2

    it1 = iter(catalog)
    it2 = iter(catalog)

    page1 = next(it1)
    page2 = next(it2)
    assert page1["RA"].base is page2["RA"].base

    page1 = next(it1)
    assert page1["RA"].base is not page2["RA"].base

    page2 = next(it2)
    assert page1["RA"].base is page2["RA"].base
