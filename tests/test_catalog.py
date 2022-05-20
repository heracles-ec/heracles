import pytest
import numpy as np


def test_delete_catalog_rows():

    from le3_pk_wl.catalog import CatalogRows, delete_catalog_rows

    ra = [1., 2., 3., 4.]

    cat = CatalogRows(size=4, ra=ra, dec=None, g1=None, g2=None, w=None)

    cat = delete_catalog_rows(cat, [1, 2])

    assert cat.size == 2
    np.testing.assert_array_equal(cat.ra, [1., 4.])
    assert cat.dec is None
    assert cat.g1 is None
    assert cat.g2 is None
    assert cat.w is None


def test_catalog_abc():

    from le3_pk_wl.catalog import Catalog, CatalogRows

    # footprint for northern hemisphere
    m = np.empty(192)
    m[:m.size//2] = 1
    m[m.size//2:] = 0

    # ABC cannot be instantiated directly
    with pytest.raises(TypeError):
        Catalog()

    # fix a set of rows to be returned for testing
    size = 100
    ra = np.random.uniform(-180, 180, size=size)
    dec = np.random.choice([+1, -1], size=size)*np.random.uniform(30, 90, size=size)
    g = np.random.uniform(0, 1, size=size)*np.exp(1j*np.random.uniform(0, 2*np.pi, size=size))
    w = np.random.uniform(0, 1, size=size)

    # concrete subclass
    # only override abstract method to test ABC interface
    class TestCatalog(Catalog):
        def _rows(self):
            batch_size = self.batch_size
            for i in range(0, size, batch_size):
                yield CatalogRows(size=len(ra[i:i+batch_size]),
                                  ra=ra[i:i+batch_size],
                                  dec=dec[i:i+batch_size],
                                  g1=g.real[i:i+batch_size],
                                  g2=g.imag[i:i+batch_size],
                                  w=w[i:i+batch_size])

    # instantiate concrete subclass for testing
    c = TestCatalog(footprint=m)

    # ---

    assert c.footprint is m
    c.footprint = None
    assert c.footprint is None

    assert c.batch_size == Catalog.default_batch_size
    c.batch_size = 1
    assert c.batch_size == 1
    c.batch_size = Catalog.default_batch_size
    assert c.batch_size == Catalog.default_batch_size

    assert c.conjugate_shear is False
    c.conjugate_shear = True
    assert c.conjugate_shear is True
    c.conjugate_shear = False
    assert c.conjugate_shear is False

    assert c.allow_invalid_positions is False
    c.allow_invalid_positions = True
    assert c.allow_invalid_positions is True
    c.allow_invalid_positions = False
    assert c.allow_invalid_positions is False

    assert c.allow_invalid_shears is False
    c.allow_invalid_shears = True
    assert c.allow_invalid_shears is True
    c.allow_invalid_shears = False
    assert c.allow_invalid_shears is False

    # ---

    c.batch_size = size

    for i, rows in enumerate(c):
        assert rows.size == size
        np.testing.assert_array_equal(rows.ra, ra)
        np.testing.assert_array_equal(rows.dec, dec)
        np.testing.assert_array_equal(rows.g1, g.real)
        np.testing.assert_array_equal(rows.g2, g.imag)
        np.testing.assert_array_equal(rows.w, w)
    assert i == 0

    c.batch_size = size//2

    for i, rows in enumerate(c):
        assert rows.size == size//2
        np.testing.assert_array_equal(rows.ra, ra[i*size//2:(i+1)*size//2])
        np.testing.assert_array_equal(rows.dec, dec[i*size//2:(i+1)*size//2])
        np.testing.assert_array_equal(rows.g1, g.real[i*size//2:(i+1)*size//2])
        np.testing.assert_array_equal(rows.g2, g.imag[i*size//2:(i+1)*size//2])
        np.testing.assert_array_equal(rows.w, w[i*size//2:(i+1)*size//2])
    assert i == 1

    c.batch_size = size

    # ---

    c.conjugate_shear = True

    rows = next(iter(c))
    assert rows.size == size
    np.testing.assert_array_equal(rows.ra, ra)
    np.testing.assert_array_equal(rows.dec, dec)
    np.testing.assert_array_equal(rows.g1, g.real)
    np.testing.assert_array_equal(rows.g2, -g.imag)
    np.testing.assert_array_equal(rows.w, w)

    c.conjugate_shear = False

    # ---

    ra[0] = np.nan
    dec[1] = np.nan

    with pytest.raises(ValueError):
        next(iter(c))

    c.allow_invalid_positions = True

    rows = next(iter(c))
    assert rows.size == size - 2
    np.testing.assert_array_equal(rows.ra, ra[2:])
    np.testing.assert_array_equal(rows.dec, dec[2:])
    np.testing.assert_array_equal(rows.g1, g.real[2:])
    np.testing.assert_array_equal(rows.g2, g.imag[2:])
    np.testing.assert_array_equal(rows.w, w[2:])

    ra[0] = 0.
    dec[1] = 45.

    c.allow_invalid_positions = False

    # ---

    g[-2] = complex(np.nan, 0.)
    g[-1] = complex(0., np.nan)

    with pytest.raises(ValueError):
        next(iter(c))

    c.allow_invalid_shears = True

    rows = next(iter(c))
    assert rows.size == size - 2
    np.testing.assert_array_equal(rows.ra, ra[:-2])
    np.testing.assert_array_equal(rows.dec, dec[:-2])
    np.testing.assert_array_equal(rows.g1, g.real[:-2])
    np.testing.assert_array_equal(rows.g2, g.imag[:-2])
    np.testing.assert_array_equal(rows.w, w[:-2])

    c.allow_invalid_shears = False

    w[-2] = w[-1] = 0.

    rows = next(iter(c))
    assert rows.size == size - 2
    np.testing.assert_array_equal(rows.ra, ra[:-2])
    np.testing.assert_array_equal(rows.dec, dec[:-2])
    np.testing.assert_array_equal(rows.g1, g.real[:-2])
    np.testing.assert_array_equal(rows.g2, g.imag[:-2])
    np.testing.assert_array_equal(rows.w, w[:-2])

    g[-2] = g[-1] = 0.
    w[-2] = w[-1] = 1.

    # ---

    c.footprint = m

    northern = np.where(dec > 0)[0]

    rows = next(iter(c))
    assert rows.size == len(northern)
    np.testing.assert_array_equal(rows.ra, ra[northern])
    np.testing.assert_array_equal(rows.dec, dec[northern])
    np.testing.assert_array_equal(rows.g1, g.real[northern])
    np.testing.assert_array_equal(rows.g2, g.imag[northern])
    np.testing.assert_array_equal(rows.w, w[northern])

    c.footprint = None


def test_catalog_empty_rows():

    from le3_pk_wl.catalog import Catalog, CatalogRows

    class TestCatalogEmpty(Catalog):
        def _rows(self):
            yield CatalogRows(size=0, ra=None, dec=None, g1=None, g2=None, w=None)

    c = TestCatalogEmpty()

    with pytest.raises(StopIteration):
        next(iter(c))


def test_catalog_missing_positions():

    from le3_pk_wl.catalog import Catalog, CatalogRows

    size = 10
    ra = np.random.uniform(-180, 180, size=size)
    dec = np.random.uniform(-90, 90, size=size)
    g1 = np.random.uniform(-1, 1, size=size)
    g2 = np.random.uniform(-1, 1, size=size)
    w = np.random.uniform(0, 1, size=size)

    class TestCatalogMissingRaDec(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=None, dec=None, g1=g1, g2=g2, w=w)

    c = TestCatalogMissingRaDec()

    with pytest.raises(TypeError):
        next(iter(c))

    class TestCatalogMissingRa(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=None, dec=dec, g1=g1, g2=g2, w=w)

    c = TestCatalogMissingRa()

    with pytest.raises(TypeError):
        next(iter(c))

    class TestCatalogMissingDec(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=ra, dec=None, g1=g1, g2=g2, w=w)

    c = TestCatalogMissingDec()

    with pytest.raises(TypeError):
        next(iter(c))


def test_catalog_missing_shears():

    from le3_pk_wl.catalog import Catalog, CatalogRows

    size = 10
    ra = np.random.uniform(-180, 180, size=size)
    dec = np.random.uniform(-90, 90, size=size)
    g1 = np.random.uniform(-1, 1, size=size)
    g2 = np.random.uniform(-1, 1, size=size)
    w = np.random.uniform(0, 1, size=size)

    class TestCatalogMissingShears(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=ra, dec=dec, g1=None, g2=None, w=None)

    c = TestCatalogMissingShears()

    rows = next(iter(c))
    assert rows.size == size
    assert rows.ra is ra
    assert rows.dec is dec
    assert rows.g1 is None
    assert rows.g2 is None
    assert rows.w is None

    class TestCatalogMissingWeights(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=ra, dec=dec, g1=g1, g2=g2, w=None)

    c = TestCatalogMissingWeights()

    rows = next(iter(c))
    assert rows.size == size
    assert rows.ra is ra
    assert rows.dec is dec
    assert rows.g1 is g1
    assert rows.g2 is g2
    assert rows.w is None

    class TestCatalogMissingG1(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=ra, dec=dec, g1=None, g2=g2, w=w)

    c = TestCatalogMissingG1()

    with pytest.raises(TypeError):
        next(iter(c))

    class TestCatalogMissingG2(Catalog):
        def _rows(self):
            yield CatalogRows(size=size, ra=ra, dec=dec, g1=g1, g2=None, w=w)

    c = TestCatalogMissingG2()

    with pytest.raises(TypeError):
        next(iter(c))


def test_column_reader():

    from le3_pk_wl.catalog import column_reader

    rows = np.empty(100, [('a', float), ('b', float), ('c', float)])
    rows['a'] = 1
    rows['b'] = 2
    rows['c'] = 3

    read = column_reader(rows)
    assert read.size == len(rows)
    assert read.ra is None
    assert read.dec is None
    assert read.g1 is None
    assert read.g2 is None
    assert read.w is None

    read = column_reader(rows, ra='a', g1='b', w='c')
    assert read.size == len(rows)
    np.testing.assert_array_equal(read.ra, rows['a'])
    assert read.dec is None
    np.testing.assert_array_equal(read.g1, rows['b'])
    assert read.g2 is None
    np.testing.assert_array_equal(read.w, rows['c'])


def test_array_catalog_unstructured():

    from le3_pk_wl.catalog import ArrayCatalog

    arr = np.random.rand(100, 4)

    cat = ArrayCatalog(arr)

    for i, rows in enumerate(cat):
        assert rows.size == 100
        np.testing.assert_array_equal(rows.ra, arr[:, 0])
        np.testing.assert_array_equal(rows.dec, arr[:, 1])
        np.testing.assert_array_equal(rows.g1, arr[:, 2])
        np.testing.assert_array_equal(rows.g2, arr[:, 3])
        assert rows.w is None
    assert i == 0


def test_array_catalog_structured():

    from le3_pk_wl.catalog import ArrayCatalog, CatalogRows, CatalogColumns

    def columns_function(rows):
        return CatalogRows(size=len(rows), ra=rows['a'], dec=rows['b'], g1=None, g2=None, w=rows['c'])

    columns_namedtuple = CatalogColumns(ra='a', dec='b', g1=None, g2=None, w='c')

    columns_tuple = ('a', 'b', None, None, 'c')

    for columns in columns_function, columns_namedtuple, columns_tuple:

        arr = np.empty(100, [('a', float), ('b', float), ('c', float)])
        for name in arr.dtype.names:
            arr[name] = np.random.rand(len(arr))

        cat = ArrayCatalog(arr, columns)

        for i, rows in enumerate(cat):
            assert rows.size == 100
            np.testing.assert_array_equal(rows.ra, arr['a'])
            np.testing.assert_array_equal(rows.dec, arr['b'])
            assert rows.g1 is None
            assert rows.g2 is None
            np.testing.assert_array_equal(rows.w, arr['c'])
        assert i == 0


def test_fits_catalog(tmp_path):

    import fitsio
    from le3_pk_wl.catalog import FitsCatalog, CatalogRows

    size = 100
    ra = np.random.uniform(-180, 180, size=size)
    dec = np.random.uniform(-90, 90, size=size)

    filename = str(tmp_path / 'catalog.fits')

    with fitsio.FITS(filename, 'rw') as fits:
        fits.write(None)
        fits.write_table([ra, dec], names=['COL1', 'COL2'], extname='MYEXT')

    def columns(rows):
        ra = rows['COL1']
        dec = rows['COL2']
        return CatalogRows(size=len(rows), ra=ra, dec=dec, g1=None, g2=None, w=None)

    c = FitsCatalog(filename, columns)

    rows = next(iter(c))
    assert rows.size == size
    np.testing.assert_array_equal(rows.ra, ra)
    np.testing.assert_array_equal(rows.dec, dec)
    assert rows.g1 is None
    assert rows.g2 is None
    assert rows.w is None

    c = FitsCatalog(filename, columns, ext='MYEXT')

    rows = next(iter(c))
    assert rows.size == size
    np.testing.assert_array_equal(rows.ra, ra)
    np.testing.assert_array_equal(rows.dec, dec)
    assert rows.g1 is None
    assert rows.g2 is None
    assert rows.w is None

    c = FitsCatalog(filename, columns, query='COL1 > 0')

    sel = np.where(ra > 0)[0]

    rows = next(iter(c))
    assert rows.size == len(sel)
    np.testing.assert_array_equal(rows.ra, ra[sel])
    np.testing.assert_array_equal(rows.dec, dec[sel])
    assert rows.g1 is None
    assert rows.g2 is None
    assert rows.w is None

    c = FitsCatalog(filename, columns).query('COL1 > 0')

    sel = np.where(ra > 0)[0]

    rows = next(iter(c))
    assert rows.size == len(sel)
    np.testing.assert_array_equal(rows.ra, ra[sel])
    np.testing.assert_array_equal(rows.dec, dec[sel])
    assert rows.g1 is None
    assert rows.g2 is None
    assert rows.w is None

    c = FitsCatalog(filename, columns).query('COL1 > 0').query('COL2 < 0')

    sel = np.where((ra > 0) & (dec < 0))[0]

    rows = next(iter(c))
    assert rows.size == len(sel)
    np.testing.assert_array_equal(rows.ra, ra[sel])
    np.testing.assert_array_equal(rows.dec, dec[sel])
    assert rows.g1 is None
    assert rows.g2 is None
    assert rows.w is None
