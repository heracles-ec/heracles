import unittest.mock
import numpy as np
import pytest

import heracles
import heracles.config


def test_field_type():
    assert heracles.config.FieldType["positions"].value is heracles.Positions
    assert heracles.config.FieldType["shears"].value is heracles.Shears
    assert heracles.config.FieldType["visibility"].value is heracles.Visibility
    assert heracles.config.FieldType["weights"].value is heracles.Weights
    assert str(heracles.config.FieldType.positions) == "positions"


def test_mapper_type():
    assert heracles.config.MapperType["none"]
    assert heracles.config.MapperType["healpix"]
    assert heracles.config.MapperType["discrete"]
    assert str(heracles.config.MapperType.none) == "none"


def test_selection_config():
    config = heracles.config._load_selection({"key": 1, "selection": "abc"})

    assert isinstance(config, heracles.config.SelectionConfig)
    assert config.key == 1
    assert config.selection == "abc"
    assert config.visibility is None

    config = heracles.config._load_selection(
        {
            "key": 1,
            "selection": "abc",
            "visibility": "def",
        }
    )

    assert isinstance(config, heracles.config.SelectionConfig)
    assert config.key == 1
    assert config.selection == "abc"
    assert config.visibility == "def"

    with pytest.raises(ValueError, match="missing option"):
        heracles.config._load_selection({"key": 1})

    with pytest.raises(ValueError, match="unknown option"):
        heracles.config._load_selection({"key": 1, "selection": "abc", "bad": 0})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_selection({"key": "1", "selection": "abc"})

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_selection({"key": 1, "selection": 10})

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_selection(
            {"key": 1, "selection": "abc", "visibility": 10}
        )


def test_catalog_config():
    config = heracles.config._load_catalog(
        {
            "source": "catalog.fits",
            "fields": ["A", "B"],
        },
    )

    assert isinstance(config, heracles.config.CatalogConfig)
    assert config.source == "catalog.fits"
    assert config.fields == ["A", "B"]
    assert config.label is None
    assert config.selections == []
    assert config.visibility is None

    # test with invalid source

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_catalog({"source": object()})

    # test with invalid fields

    with pytest.raises(ValueError, match="expected list of str"):
        heracles.config._load_catalog({"source": "s", "fields": [1, 2]})

    # test with label

    config = heracles.config._load_catalog(
        {
            "source": "catalog.fits",
            "fields": ["A", "B"],
            "label": "my label",
        },
    )

    assert isinstance(config, heracles.config.CatalogConfig)
    assert config.source == "catalog.fits"
    assert config.fields == ["A", "B"]
    assert config.label == "my label"
    assert config.selections == []
    assert config.visibility is None

    # test with invalid label

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_catalog({"source": "s", "fields": [], "label": 0})

    # test with selections

    selections = [object(), object()]

    with unittest.mock.patch("heracles.config._load_selection") as mock:
        config = heracles.config._load_catalog(
            {
                "source": "catalog.fits",
                "fields": ["A", "B"],
                "selections": selections,
            },
        )

    assert isinstance(config, heracles.config.CatalogConfig)
    assert config.source == "catalog.fits"
    assert config.fields == ["A", "B"]
    assert config.label is None
    assert config.selections == [mock.return_value, mock.return_value]
    assert config.visibility is None

    assert mock.call_args_list == [unittest.mock.call(_) for _ in selections]

    # test with invalid selections

    with pytest.raises(ValueError, match="expected list, got"):
        heracles.config._load_catalog(
            {
                "source": "catalog.fits",
                "fields": ["A", "B"],
                "selections": object(),
            },
        )

    # test with visibility

    config = heracles.config._load_catalog(
        {
            "source": "catalog.fits",
            "fields": ["A", "B"],
            "visibility": "vmap.fits",
        },
    )

    assert isinstance(config, heracles.config.CatalogConfig)
    assert config.source == "catalog.fits"
    assert config.fields == ["A", "B"]
    assert config.label is None
    assert config.selections == []
    assert config.visibility == "vmap.fits"

    # test with invalid visibility

    with pytest.raises(ValueError, match="expected str, got"):
        heracles.config._load_catalog(
            {
                "source": "catalog.fits",
                "fields": ["A", "B"],
                "visibility": object(),
            },
        )

    # test with invalid options

    with pytest.raises(ValueError, match="unknown options: 'a', 'b'"):
        heracles.config._load_catalog(
            {
                "source": "catalog.fits",
                "fields": ["A", "B"],
                "a": object(),
                "b": object(),
            },
        )


def test_load_mapper():
    with unittest.mock.patch("heracles.healpy.HealpixMapper") as mock:
        heracles.config._load_mapper({"nside": 1})
    mock.assert_called_once_with(1, None)

    with unittest.mock.patch("heracles.healpy.HealpixMapper") as mock:
        heracles.config._load_mapper({"mapper": "healpix", "nside": 1, "lmax": 2})
    mock.assert_called_once_with(1, 2)

    with pytest.raises(ValueError, match="missing option"):
        heracles.config._load_mapper({"mapper": "healpix", "lmax": 2})

    with unittest.mock.patch("heracles.ducc.DiscreteMapper") as mock:
        heracles.config._load_mapper({"mapper": "discrete", "lmax": 1})
    mock.assert_called_once_with(1)

    with pytest.raises(ValueError, match="missing option"):
        heracles.config._load_mapper({"mapper": "discrete"})

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_mapper({"mapper": None})

    with pytest.raises(ValueError, match="invalid value"):
        heracles.config._load_mapper({"mapper": "invalid"})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_mapper({"nside": "1"})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_mapper({"lmax": "1"})


def test_load_field():
    key, field = heracles.config._load_field(
        {
            "key": "POS",
            "type": "positions",
            "mapper": "none",
        }
    )
    assert key == "POS"
    assert isinstance(field, heracles.Positions)
    assert field.mapper is None
    assert field.columns is None
    assert field.mask is None

    key, field = heracles.config._load_field(
        {
            "key": "POS",
            "type": "positions",
            "mapper": "none",
            "columns": ["RA", "Dec"],
            "mask": "MASK",
        }
    )
    assert key == "POS"
    assert isinstance(field, heracles.Positions)
    assert field.mapper is None
    assert field.columns == ("RA", "Dec", None)
    assert field.mask == "MASK"

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_field({"key": 1, "type": "positions"})

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_field({"key": "POS", "type": object()})

    with pytest.raises(ValueError, match="invalid value"):
        heracles.config._load_field({"key": "POS", "type": "invalid"})

    with pytest.raises(ValueError, match="expected list of str"):
        heracles.config._load_field({"key": "POS", "type": "positions", "columns": [1]})

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_field(
            {"key": "POS", "type": "positions", "mapper": "none", "mask": object()}
        )

    with pytest.raises(ValueError, match="POS: mapper: invalid value"):
        heracles.config._load_field(
            {"key": "POS", "type": "positions", "mapper": "invalid"}
        )

    with pytest.raises(ValueError, match="POS: unknown options"):
        heracles.config._load_field(
            {"key": "POS", "type": "positions", "mapper": "none", "invalid": 0}
        )


def test_load_bins():
    bins, weights = heracles.config._load_bins({"n": 10}, 0, 10)
    np.testing.assert_array_equal(bins, np.linspace(0, 11, 11))
    assert weights is None

    bins, weights = heracles.config._load_bins({"n": 4, "spacing": "log"}, 1, 11)
    np.testing.assert_array_almost_equal_nulp(bins, np.geomspace(1, 12, 5))
    assert weights is None

    bins, weights = heracles.config._load_bins({"n": 2, "weights": "2l+1"}, 0, 2)
    np.testing.assert_array_equal(bins, np.linspace(0, 3, 3))
    assert weights == "2l+1"

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_bins({"n": "a"}, 0, 1)

    with pytest.raises(ValueError, match="invalid number of bins"):
        heracles.config._load_bins({"n": 1}, 0, 1)

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_bins({"n": 2, "spacing": 1}, 0, 1)

    with pytest.raises(ValueError, match="invalid value"):
        heracles.config._load_bins({"n": 2, "spacing": "invalid"}, 0, 1)

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_bins({"n": 2, "weights": 1}, 0, 1)


def test_load_twopoint():
    key, twopoint = heracles.config._load_twopoint({"key": "POS-POS"})
    assert key == ("POS", "POS")
    assert twopoint.lmin is None
    assert twopoint.lmax is None
    assert twopoint.l2max is None
    assert twopoint.l3max is None
    assert twopoint.debias
    assert twopoint.bins is None

    key, twopoint = heracles.config._load_twopoint({"key": ["POS", "POS"]})
    assert key == ("POS", "POS")

    key, twopoint = heracles.config._load_twopoint({"key": "x", "lmin": 0})
    assert twopoint.lmin == 0

    key, twopoint = heracles.config._load_twopoint({"key": "x", "lmax": 0})
    assert twopoint.lmax == 0

    key, twopoint = heracles.config._load_twopoint({"key": "x", "l2max": 0})
    assert twopoint.l2max == 0

    key, twopoint = heracles.config._load_twopoint({"key": "x", "l3max": 0})
    assert twopoint.l3max == 0

    key, twopoint = heracles.config._load_twopoint({"key": "x", "debias": False})
    assert not twopoint.debias

    key, twopoint = heracles.config._load_twopoint(
        {"key": "x", "lmin": 0, "lmax": 1, "bins": {"n": 2}}
    )
    np.testing.assert_array_equal(twopoint.bins, [0, 1, 2])

    with pytest.raises(ValueError, match="expected str"):
        heracles.config._load_twopoint({"key": 0})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_twopoint({"key": "x", "lmin": "0"})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_twopoint({"key": "x", "lmax": "0"})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_twopoint({"key": "x", "l2max": "0"})

    with pytest.raises(ValueError, match="expected int"):
        heracles.config._load_twopoint({"key": "x", "l3max": "0"})

    with pytest.raises(ValueError, match="expected bool"):
        heracles.config._load_twopoint({"key": "x", "debias": 0})

    with pytest.raises(ValueError, match="expected dict"):
        heracles.config._load_twopoint({"key": "x", "bins": 0})

    with pytest.raises(ValueError, match="requires lmin and lmax"):
        heracles.config._load_twopoint({"key": "x", "bins": {}})

    with pytest.raises(ValueError, match="bins: n: expected int"):
        heracles.config._load_twopoint(
            {"key": "x", "lmin": 0, "lmax": 1, "bins": {"n": "0"}}
        )

    with pytest.raises(ValueError, match="unknown options"):
        heracles.config._load_twopoint({"key": "x", "unknown": 0})


def test_load():
    config = heracles.config.load({})
    assert config.catalogs == []
    assert config.fields == {}
    assert config.spectra == {}
