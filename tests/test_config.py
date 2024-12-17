import io
import textwrap
import unittest.mock

import pytest

import heracles
import heracles.config


def load(text):
    with io.StringIO(textwrap.dedent(text)) as stream:
        loader = heracles.config.YamlLoader(stream)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


@unittest.mock.patch("heracles.Positions")
def test_construct_field(mock):
    field = load(
        """
        !positions
        """
    )
    assert field is mock.return_value
    mock.assert_called_with(None)

    field = load(
        """
        !positions lon lat weight
        """
    )
    assert field is mock.return_value
    mock.assert_called_with(None, "lon", "lat", "weight")

    field = load(
        """
        !positions
          - lon
          - lat
        """
    )
    assert field is mock.return_value
    mock.assert_called_with(None, "lon", "lat")

    field = load(
        """
        !positions
          mapper: xyz
        """
    )
    assert field is mock.return_value
    mock.assert_called_with("xyz")

    field = load(
        """
        !positions
          columns: lon lat
        """
    )
    assert field is mock.return_value
    mock.assert_called_with(None, "lon", "lat")

    field = load(
        """
        !positions
          columns:
            - lon
            - lat
            - weight
        """
    )
    assert field is mock.return_value
    mock.assert_called_with(None, "lon", "lat", "weight")

    field = load(
        """
        !positions
          arg1: abc
          arg2: 2
          arg3:
            - x
            - y
            - z
        """
    )
    assert field is mock.return_value
    mock.assert_called_with(None, arg1="abc", arg2=2, arg3=["x", "y", "z"])


@pytest.mark.parametrize(
    "tag, field",
    [
        ("!positions", "heracles.Positions"),
        ("!shears", "heracles.Shears"),
        ("!visibility", "heracles.Visibility"),
        ("!weights", "heracles.Weights"),
    ],
)
def test_field(tag, field):
    with unittest.mock.patch(field) as mock:
        obj = load(tag)
        assert obj is mock.return_value
        mock.assert_called_with(None)


@unittest.mock.patch("heracles.config.YamlLoader")
def test_load_config(mock, tmp_path):
    path = tmp_path / "test.yaml"
    path.touch()

    values = {"a": 1, "b": 2, "c": 3}

    mock.return_value.get_single_data.return_value = values

    config = heracles.load_config(path)

    assert mock.call_count == 1
    assert mock.call_args[0][0].name == str(path)

    loader = mock.return_value
    loader.get_single_data.assert_called_once_with()
    loader.dispose.assert_called_once_with()

    assert config.__dict__ == values
