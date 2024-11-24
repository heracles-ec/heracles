# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of Heracles.
#
# Heracles is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Heracles is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Heracles. If not, see <https://www.gnu.org/licenses/>.
"""
Module for loading YAML configuration files.
"""

from __future__ import annotations

from types import SimpleNamespace

import heracles

from heracles.core import external_dependency_explainer

with external_dependency_explainer:
    from yaml.loader import SafeLoader
    from yaml import ScalarNode, SequenceNode, MappingNode


class YamlLoader(SafeLoader):
    """
    Loader for Heracles configuration in YAML format.
    """

    @classmethod
    def tag(cls, tag):
        """
        Decorator for custom constructors.
        """

        def decorator(ctor):
            cls.add_constructor(tag, ctor)
            return ctor

        return decorator


def _construct_field(cls, loader, node):
    """
    Construct a Field instance.
    """
    if isinstance(node, MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
        mapper = kwargs.pop("mapper", None)
        columns = kwargs.pop("columns", "")
    elif isinstance(node, SequenceNode):
        mapper = None
        columns = loader.construct_sequence(node, deep=True)
        kwargs = {}
    elif isinstance(node, ScalarNode):
        mapper = None
        columns = node.value
        kwargs = {}

    if columns is None:
        columns = []
    elif isinstance(columns, str):
        columns = columns.split()

    return cls(mapper, *columns, **kwargs)


@YamlLoader.tag("!heracles.Positions")
def construct_positions(loader, node):
    """
    Construct :class:`heracles.Positions`.
    """
    return _construct_field(heracles.Positions, loader, node)


@YamlLoader.tag("!heracles.Shears")
def construct_shears(loader, node):
    """
    Construct :class:`heracles.Shears`.
    """
    return _construct_field(heracles.Shears, loader, node)


@YamlLoader.tag("!heracles.Visibility")
def construct_visibility(loader, node):
    """
    Construct :class:`heracles.Visibility`.
    """
    return _construct_field(heracles.Visibility, loader, node)


@YamlLoader.tag("!heracles.Weights")
def construct_weights(loader, node):
    """
    Construct :class:`heracles.Weights`.
    """
    return _construct_field(heracles.Weights, loader, node)


def load_config(path):
    """
    Load YAML configuration from *path*.
    """
    with open(path) as fp:
        loader = YamlLoader(fp)
        try:
            config = loader.get_single_data()
        finally:
            loader.dispose()
    return SimpleNamespace(**config)
