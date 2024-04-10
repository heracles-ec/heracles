from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest


def test_getlist():
    from heracles.cli import getlist

    assert (
        getlist(
            """
                x
                y
                z
            """,
        )
        == ["x", "y", "z"]
    )
    assert getlist("xyz") == ["xyz"]


def test_getdict():
    from heracles.cli import getdict

    assert (
        getdict(
            """
                x=1
                y = 2
                z= 3
            """,
        )
        == {"x": "1", "y": "2", "z": "3"}
    )

    with pytest.raises(ValueError, match="Invalid value"):
        getdict(
            """
                1
                2
                3
            """,
        )


def test_getchoice():
    from heracles.cli import getchoice

    choices = {
        "first": 1,
        "second": 2,
    }

    assert getchoice("first", choices) == 1
    with pytest.raises(ValueError, match="Invalid value"):
        getchoice("third", choices)


@patch.dict("os.environ", {"HOME": "/home/user", "TEST": "folder"})
def test_getpath():
    from heracles.cli import getpath

    assert getpath("~/${TEST}/file.txt") == "/home/user/folder/file.txt"


def test_getfilter():
    from heracles.cli import getfilter

    assert getfilter("a") == [("a",)]
    assert getfilter("a, ..., 1, 2") == [("a", ..., 1, 2)]
    assert (
        getfilter(
            """
                a, 1
                b, 2
            """,
        )
        == [("a", 1), ("b", 2)]
    )


def test_subsections():
    from heracles.cli import ConfigParser

    config = ConfigParser()
    config.read_dict(
        {
            "getme:a": {},
            "getme: b ": {},
            "getmenot:c": {},
        },
    )

    assert config.subsections("getme") == {"a": "getme:a", "b": "getme: b "}


def test_field_from_config():
    from unittest.mock import Mock

    from heracles.cli import ConfigParser, field_from_config

    mock, other_mock = Mock(), Mock()

    mock_field_types = {
        "test": mock,
        "other_test": other_mock,
        "error": "<invalid>",
    }

    config = ConfigParser()
    config[config.default_section]["mapper"] = "none"
    config.read_dict(
        {
            "a": {
                "type": "test",
                "columns": """
                    COL1
                    -COL2
                """,
                "mask": "x",
            },
            "b": {
                "type": "other_test",
            },
            "c": {
                "type": "error",
            },
        },
    )

    with patch.dict("heracles.cli.FIELD_TYPES", mock_field_types):
        a = field_from_config(config, "a")
        b = field_from_config(config, "b")
        with pytest.raises(RuntimeError, match="Internal error"):
            field_from_config(config, "c")

    mock.assert_called_once_with(None, "COL1", "-COL2", mask="x")
    assert mock.return_value is a
    other_mock.assert_called_once_with(None, mask=None)
    assert other_mock.return_value is b


@patch("heracles.cli.field_from_config")
def test_fields_from_config(mock):
    from heracles.cli import ConfigParser, fields_from_config

    config = ConfigParser()
    config.read_dict(
        {
            "fields:a": {},
            "fields:b": {},
            "fields:c": {},
        },
    )

    m = fields_from_config(config)

    assert m == {
        "a": mock.return_value,
        "b": mock.return_value,
        "c": mock.return_value,
    }
    assert mock.call_args_list == [
        ((config, "fields:a"),),
        ((config, "fields:b"),),
        ((config, "fields:c"),),
    ]


@patch("heracles.io.read_vmap")
def test_catalog_from_config(mock):
    from heracles.cli import ConfigParser, catalog_from_config

    # single visibility

    config = ConfigParser()
    config.read_dict(
        {
            "test_with_single_visibility": {
                "source": "catalog.fits",
                "selections": """
                    0 = TOM_BIN_ID==0
                    1 = TOM_BIN_ID==1
                    2 = TOM_BIN_ID==2
                """,
                "visibility": "vmap.fits",
            },
            "test_with_many_visibilities": {
                "source": "catalog.fits",
                "selections": """
                    0 = TOM_BIN_ID==0
                    1 = TOM_BIN_ID==1
                    2 = TOM_BIN_ID==2
                """,
                "visibility": """
                    0 = vmap.0.fits
                    2 = vmap.2.fits
                """,
            },
        },
    )

    catalog = catalog_from_config(config, "test_with_single_visibility", "label 1")

    assert catalog.keys() == {0, 1, 2}
    assert catalog[0].base.__class__.__name__ == "FitsCatalog"
    assert catalog[0].base.path == "catalog.fits"
    assert catalog[0].base.visibility is mock.return_value
    assert catalog[0].base.label == "label 1"
    assert catalog[1].base is catalog[0].base
    assert catalog[2].base is catalog[0].base
    assert catalog[0].label is catalog[0].base.label
    assert catalog[1].label is catalog[0].base.label
    assert catalog[2].label is catalog[0].base.label
    assert catalog[0].selection == "TOM_BIN_ID==0"
    assert catalog[1].selection == "TOM_BIN_ID==1"
    assert catalog[2].selection == "TOM_BIN_ID==2"
    assert catalog[0].visibility is catalog[0].base.visibility
    assert catalog[1].visibility is catalog[0].base.visibility
    assert catalog[2].visibility is catalog[0].base.visibility
    assert mock.call_args_list == [(("vmap.fits",),)]

    mock.reset_mock()

    catalog = catalog_from_config(config, "test_with_many_visibilities", "label 2")

    assert catalog.keys() == {0, 1, 2}
    assert catalog[0].base.__class__.__name__ == "FitsCatalog"
    assert catalog[0].base.path == "catalog.fits"
    assert catalog[0].base.visibility is None
    assert catalog[0].base.label == "label 2"
    assert catalog[1].base is catalog[0].base
    assert catalog[2].base is catalog[0].base
    assert catalog[0].label is catalog[0].base.label
    assert catalog[1].label is catalog[0].base.label
    assert catalog[2].label is catalog[0].base.label
    assert catalog[0].selection == "TOM_BIN_ID==0"
    assert catalog[1].selection == "TOM_BIN_ID==1"
    assert catalog[2].selection == "TOM_BIN_ID==2"
    assert catalog[0].visibility is mock.return_value
    assert catalog[1].visibility is None
    assert catalog[2].visibility is mock.return_value
    assert mock.call_args_list == [
        (("vmap.0.fits",),),
        (("vmap.2.fits",),),
    ]

    with pytest.raises(ValueError, match="Duplicate selection"):
        catalog_from_config(config, "test_with_single_visibility", out=catalog)


@patch("heracles.cli.catalog_from_config")
def test_catalogs_from_config(mock):
    from heracles.cli import ConfigParser, catalogs_from_config

    config = ConfigParser()
    config.read_dict(
        {
            "catalogs:a": {},
            "catalogs:b": {},
            "catalogs:c": {},
        },
    )

    c = catalogs_from_config(config)

    assert mock.call_args_list == [
        ((config, "catalogs:a", "a"), {"out": c}),
        ((config, "catalogs:b", "b"), {"out": c}),
        ((config, "catalogs:c", "c"), {"out": c}),
    ]


def test_bins_from_config():
    from heracles.cli import ConfigParser, bins_from_config

    config = ConfigParser()
    config.read_dict(
        {
            "linear_bins": {
                "lmin": "0",
                "lmax": "4",
                "bins": "5 linear",
            },
            "log_bins": {
                "lmin": "1",
                "lmax": "999_999",
                "bins": "6 log",
            },
            "sqrt_bins": {
                "lmin": "1",
                "lmax": "35",
                "bins": "5 sqrt",
            },
            "log1p_bins": {
                "lmin": "0",
                "lmax": "9",
                "bins": "5 log1p",
            },
        },
    )

    npt.assert_array_equal(
        bins_from_config(config, "linear_bins")[0],
        [0, 1, 2, 3, 4, 5],
    )

    npt.assert_array_equal(
        bins_from_config(config, "log_bins")[0],
        [1, 10, 100, 1000, 10000, 100000, 1000000],
    )

    npt.assert_array_equal(
        bins_from_config(config, "sqrt_bins")[0],
        [1, 4, 9, 16, 25, 36],
    )

    npt.assert_allclose(
        bins_from_config(config, "log1p_bins")[0],
        np.expm1(np.linspace(np.log1p(0), np.log1p(10), 6)),
    )


@patch("heracles.cli.bins_from_config")
def test_spectrum_from_config(mock):
    from heracles.cli import ConfigParser, spectrum_from_config

    config = ConfigParser()
    config.read_dict(
        {
            "a": {
                "lmax": 10,
                "l2max": 12,
                "l3max": 20,
                "include": "x",
                "exclude": "y",
                "bins": "...",
            },
        },
    )

    assert spectrum_from_config(config, "a") == {
        "lmax": 10,
        "l2max": 12,
        "l3max": 20,
        "include": [("x",)],
        "exclude": [("y",)],
        "bins": mock.return_value,
    }
