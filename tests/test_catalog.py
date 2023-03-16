import pytest
import numpy as np
import numpy.testing as npt


@pytest.fixture
def catalog():
    from le3_pk_wl.catalog import Catalog, CatalogPage

    # fix a set of rows to be returned for testing
    size = 100
    x = np.random.rand(size)
    y = np.random.rand(size)
    z = np.random.rand(size)

    class TestCatalog(Catalog):
        SIZE = size
        DATA = dict(x=x, y=y, z=z)

        def __init__(self):
            super().__init__()
            self._size = self.SIZE
            self._names = list(self.DATA.keys())

        # implement abstract method
        def _pages(self):
            size = self.SIZE
            page_size = self.page_size
            for i in range(0, size, page_size):
                page = {k: v[i:i+page_size] for k, v in self.DATA.items()}
                yield CatalogPage(page)

    return TestCatalog()


def test_catalog_page():

    from le3_pk_wl.catalog import CatalogPage

    a = [1., 2., 3., 4.]
    b = [5., 6., 7., 8.]

    page = CatalogPage({'a': a, 'b': b})

    # test basic behaviour
    assert len(page) == 2
    npt.assert_array_equal(page['a'], a)
    npt.assert_array_equal(page['b'], b)
    npt.assert_array_equal(page['a', 'b'], [a, b])
    npt.assert_array_equal(page[['a', 'b']], [a, b])

    # test names attribute
    assert page.names == ['a', 'b']

    # test size attribue
    assert page.size == 4

    # test data attribute, which is a readonly view
    data = page.data
    assert list(data.keys()) == ['a', 'b']
    npt.assert_array_equal(list(data.values()), [a, b])
    with pytest.raises(TypeError):
        data['a'] = b

    # test iterator
    assert [_ for _ in page] == ['a', 'b']

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
    np.testing.assert_array_equal(page['a'], [1., 4.])
    np.testing.assert_array_equal(page['b'], [5., 8.])
    assert page.size == 2

    # test exception if column does not exist
    with pytest.raises(KeyError):
        page['c']
    with pytest.raises(KeyError):
        page['a', 'b', 'c']

    # test exception if rows have unequal size
    with pytest.raises(ValueError):
        CatalogPage({'a': [1, 2, 3], 'b': [1, 2]})


def test_catalog_page_get():

    from le3_pk_wl.catalog import CatalogPage

    a = [np.nan, 2., 3., 4.]
    b = [5., 6., 7., 8.]

    page = CatalogPage({'a': a, 'b': b})
    with pytest.raises(ValueError, match='column "a"'):
        page.get('a')
    npt.assert_array_equal(page.get('b'), b)
    with pytest.raises(ValueError, match='column "a"'):
        page.get('a', 'b')

    a[0] = 1.
    b[1] = np.nan

    page = CatalogPage({'a': a, 'b': b})
    npt.assert_array_equal(page.get('a'), a)
    with pytest.raises(ValueError, match='column "b"'):
        page.get('b')
    with pytest.raises(ValueError, match='column "b"'):
        page.get('a', 'b')

    b[1] = 6.

    page = CatalogPage({'a': a, 'b': b})
    npt.assert_array_equal(page.get('a'), a)
    npt.assert_array_equal(page.get('b'), b)
    npt.assert_array_equal(page.get('a', 'b'), [a, b])


def test_catalog_abc(catalog):

    from le3_pk_wl.catalog import Catalog

    # ABC cannot be instantiated directly
    with pytest.raises(TypeError):
        Catalog()

    # fixture has tested concrete implementation
    assert isinstance(catalog, Catalog)


def test_catalog_properties(catalog):

    from le3_pk_wl.catalog import Catalog

    assert catalog.size == catalog.SIZE
    assert catalog.names == list(catalog.DATA.keys())

    assert catalog.page_size == Catalog.default_page_size
    catalog.page_size = 1
    assert catalog.page_size == 1
    catalog.page_size = Catalog.default_page_size
    assert catalog.page_size == Catalog.default_page_size

    filt = object()
    assert catalog.filters == []
    catalog.add_filter(filt)
    assert catalog.filters == [filt]
    catalog.filters = []
    assert catalog.filters == []

    v = object()
    assert catalog.visibility is None
    catalog.visibility = v
    assert catalog.visibility is v
    catalog.visibility = None
    assert catalog.visibility is None


def test_catalog_pagination(catalog):

    size = catalog.SIZE

    for page_size in [size, size//2]:
        catalog.page_size = page_size
        for i, page in enumerate(catalog):
            assert page.size == page_size
            for k, v in catalog.DATA.items():
                vp = v[i*page_size:(i+1)*page_size]
                npt.assert_array_equal(page[k], vp)
        assert i*page_size + page.size == size


def test_catalog_empty_page():

    from le3_pk_wl.catalog import Catalog, CatalogPage

    class TestCatalogEmpty(Catalog):
        def _pages(self):
            yield CatalogPage({'lon': [], 'lat': []})

    c = TestCatalogEmpty()

    with pytest.raises(StopIteration):
        next(iter(c))


def test_catalog_copy():

    from le3_pk_wl.catalog import Catalog

    class TestCatalog(Catalog):
        def __init__(self):
            super().__init__()
            self._visibility = object()
            self._names = object()
            self._size = object()

        def _pages(self):
            return iter([])

    catalog = TestCatalog()

    copied = catalog.__copy__()

    assert isinstance(copied, TestCatalog)
    assert copied is not catalog
    assert copied.__dict__ == catalog.__dict__
    assert copied.visibility is catalog.visibility
    assert copied.names is catalog.names
    assert copied.size is catalog.size
    assert copied.filters is not catalog.filters


def test_invalid_value_filter(catalog):

    from le3_pk_wl.catalog import InvalidValueFilter

    catalog.DATA['x'][0] = np.nan
    catalog.DATA['y'][1] = np.nan

    page = next(iter(catalog))
    with pytest.raises(ValueError):
        page.get('x')
    with pytest.raises(ValueError):
        page.get('y')

    filt = InvalidValueFilter('x', 'y')

    assert repr(filt) == "InvalidValueFilter('x', 'y', weight=None, warn=True)"

    catalog.add_filter(filt)

    with pytest.warns(UserWarning):
        page = next(iter(catalog))
    assert page.size == catalog.SIZE - 2
    for k, v in catalog.DATA.items():
        npt.assert_array_equal(page.get(k), v[2:])


def test_footprint_filter(catalog):

    from le3_pk_wl.catalog import FootprintFilter
    from healpy import ang2pix

    # footprint for northern hemisphere
    nside = 8
    m = np.round(np.random.rand(12*nside**2))

    # replace x and y in catalog with lon and lat
    catalog.DATA['x'] = lon = np.random.uniform(-180, 180, size=catalog.SIZE)
    catalog.DATA['y'] = lat = np.degrees(np.arcsin(np.random.uniform(-1, 1, size=catalog.SIZE)))

    filt = FootprintFilter(m, 'x', 'y')

    assert repr(filt) == "FootprintFilter(..., 'x', 'y')"

    catalog.add_filter(filt)

    good = (m[ang2pix(nside, lon, lat, lonlat=True)] != 0)
    assert good.sum() != good.size

    page = next(iter(catalog))
    assert page.size == good.sum()
    for k, v in catalog.DATA.items():
        np.testing.assert_array_equal(page[k], v[good])


def test_array_catalog():

    from le3_pk_wl.catalog import ArrayCatalog

    arr = np.empty(100, [('lon', float), ('lat', float),
                         ('x', float), ('y', float)])
    for name in arr.dtype.names:
        arr[name] = np.random.rand(len(arr))

    catalog = ArrayCatalog(arr)

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

    copied = catalog.__copy__()

    assert isinstance(copied, ArrayCatalog)
    assert copied is not catalog
    assert copied.__dict__ == catalog.__dict__


def test_fits_catalog(tmp_path):

    import fitsio
    from le3_pk_wl.catalog import FitsCatalog

    size = 100
    ra = np.random.uniform(-180, 180, size=size)
    dec = np.random.uniform(-90, 90, size=size)

    filename = str(tmp_path / 'catalog.fits')

    with fitsio.FITS(filename, 'rw') as fits:
        fits.write(None)
        fits.write_table([ra, dec], names=['RA', 'DEC'], extname='MYEXT')

    catalog = FitsCatalog(filename)

    assert catalog.size is None
    assert catalog.names is None

    catalog.peek()

    assert catalog.size == size
    assert catalog.names == ['RA', 'DEC']

    page = next(iter(catalog))
    assert page.size == size
    assert len(page) == 2
    np.testing.assert_array_equal(page['RA'], ra)
    np.testing.assert_array_equal(page['DEC'], dec)

    catalog = FitsCatalog(filename, query='RA > 0')

    sel = np.where(ra > 0)[0]

    page = next(iter(catalog))
    assert page.size == len(sel)
    assert len(page) == 2
    np.testing.assert_array_equal(page['RA'], ra[sel])
    np.testing.assert_array_equal(page['DEC'], dec[sel])

    catalog = FitsCatalog(filename).query('RA > 0')

    sel = np.where(ra > 0)[0]

    page = next(iter(catalog))
    assert page.size == len(sel)
    assert len(page) == 2
    np.testing.assert_array_equal(page['RA'], ra[sel])
    np.testing.assert_array_equal(page['DEC'], dec[sel])

    catalog = FitsCatalog(filename).query('RA > 0').query('DEC < 0')

    sel = np.where((ra > 0) & (dec < 0))[0]

    page = next(iter(catalog))
    assert page.size == len(sel)
    assert len(page) == 2
    np.testing.assert_array_equal(page['RA'], ra[sel])
    np.testing.assert_array_equal(page['DEC'], dec[sel])

    copied = catalog.__copy__()

    assert isinstance(copied, FitsCatalog)
    assert copied is not catalog
    assert copied.__dict__ == catalog.__dict__
