# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023 Euclid Science Ground Segment
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
"""The Heracles command line interface."""

from __future__ import annotations

import argparse
import configparser
import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .fields import Field

# valid option keys
FIELD_TYPES = {
    "positions": "heracles.fields:Positions",
    "shears": "heracles.fields:Shears",
    "visibility": "heracles.fields:Visibility",
    "weights": "heracles.fields:Weights",
}


def getlist(value: str) -> list[str]:
    """Convert to list."""
    return list(filter(None, map(str.strip, value.splitlines())))


def getdict(value: str) -> dict[str, str]:
    """Convert to dictionary."""
    out = {}
    for line in map(str.strip, value.splitlines()):
        if not line:
            continue
        key, sep, val = line.partition("=")
        if sep != "=":
            msg = f"Invalid value: {line!r} (expected 'KEY = VALUE')"
            raise ValueError(msg)
        out[key.rstrip()] = val.lstrip()
    return out


def getchoice(value: str, choices: dict[str, Any]) -> Any:
    """Get choice from a fixed set of option values."""
    try:
        return choices[value]
    except KeyError:
        expected = ", ".join(map(repr, choices))
        msg = f"Invalid value: {value!r} (expected {expected})"
        raise ValueError(msg) from None


def getpath(value: str) -> str:
    "Convert to path, expanding environment variables."
    return os.path.expanduser(os.path.expandvars(value))


def getfilter(value: str) -> list[tuple[Any, ...]]:
    """Convert to list of include or exclude filters."""
    filt = []
    for row in getlist(value):
        item = []
        for part in map(str.strip, row.split(",")):
            if part == "...":
                item.append(...)
            elif part.isdigit():
                item.append(int(part))
            else:
                item.append(part)
        filt.append(tuple(item))
    return filt


class ConfigParser(configparser.ConfigParser):
    """ConfigParser with additional getters."""

    _UNSET = configparser._UNSET

    def __init__(self) -> None:
        # fully specify parent class
        super().__init__(
            defaults={
                "mapper": "healpix",
            },
            dict_type=dict,
            allow_no_value=False,
            delimiters=("=",),
            comment_prefixes=("#",),
            inline_comment_prefixes=("#",),
            strict=True,
            empty_lines_in_values=False,
            default_section="defaults",
            interpolation=None,
            converters={
                "list": getlist,
                "dict": getdict,
                "path": getpath,
                "filter": getfilter,
            },
        )

    def getchoice(
        self,
        section,
        option,
        choices,
        *,
        raw=False,
        vars=None,  # noqa: A002
        fallback=_UNSET,
    ):
        """Get choice from a fixed set of option values."""
        try:
            value = self.get(section, option, raw=False, vars=None)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not self._UNSET:
                return fallback
            raise
        return getchoice(value, choices)

    def sections(self, prefix: str | None = None) -> list[str]:
        """
        Return all the configuration section names.  If given, only
        sections starting with *prefix* are returned.
        """

        sections = super().sections()
        if prefix is not None:
            sections = [s for s in sections if s.startswith(prefix)]
        return sections

    def subsections(self, group: str) -> dict[str, str]:
        """
        Return a mapping of subsections in *group*.
        """
        sections = self.sections(f"{group}:")
        return {s.rpartition(":")[-1].strip(): s for s in sections}


def mapper_from_config(config, section):
    """Construct a mapper instance from config."""

    choices = {
        "none": "none",
        "healpix": "healpix",
    }

    mapper = config.getchoice(section, "mapper", choices)
    if mapper == "none":
        return None
    if mapper == "healpix":
        from .maps import Healpix

        nside = config.getint(section, "nside")
        lmax = config.getint(section, "lmax", fallback=None)
        deconvolve = config.getint(section, "deconvolve", fallback=None)
        return Healpix(nside, lmax, deconvolve=deconvolve)
    return None


def field_from_config(config, section):
    """Construct a field instance from config."""

    from pkgutil import resolve_name

    _type = config.getchoice(section, "type", FIELD_TYPES)
    if isinstance(_type, str):
        try:
            cls = resolve_name(_type)
        except (ValueError, ImportError, AttributeError) as exc:
            value = config.get(section, "type")
            msg = (
                f"Internal error: field type {value!r} maps to type {_type!r}, "
                f"which raised the following error: {exc!s}"
            )
            raise RuntimeError(msg) from None
    else:
        cls = _type
    mapper = mapper_from_config(config, section)
    columns = config.getlist(section, "columns", fallback=())
    mask = config.get(section, "mask", fallback=None)
    return cls(mapper, *columns, mask=mask)


def fields_from_config(config):
    """Construct all field instances from config."""
    sections = config.subsections("fields")
    return {
        name: field_from_config(config, section) for name, section in sections.items()
    }


def catalog_from_config(config, section, label=None, *, out=None):
    """Construct a catalogue instance from config."""

    from .catalog import FitsCatalog
    from .io import read_vmap

    # TODO support non-FITS catalogue sources
    source = config.getpath(section, "source")
    # check if visibility is per catalogue or per selection
    visibility: str | Mapping[str, str]
    visibility = config.get(section, "visibility", fallback=None)
    # check if visibility is a mapping
    if visibility and "\n" in visibility:
        visibility = config.getdict(section, "visibility")
    selections = config.getdict(section, "selections")
    # build the base catalogue
    base_catalog = FitsCatalog(source)
    base_catalog.label = label
    # set base catalogue's visibility if just one was given
    if isinstance(visibility, str):
        try:
            base_catalog.visibility = read_vmap(getpath(visibility))
        except (TypeError, ValueError, OSError) as exc:
            msg = f"Cannot load visibility: {exc!s}"
            raise ValueError(msg)
    # create a view of the base catalogue for each selection
    # since `out` can be given, also keep track of selections added here
    if out is None:
        out = {}
    added = set()
    for key, where in selections.items():
        # convert key to number and make sure it wasn't used before
        num = int(key)
        if out and num in out:
            msg = f"Duplicate selection: {num}"
            raise ValueError(msg)
        # create view from selection string, if present
        # otherwise, give the base catalog itself
        if where:
            catalog = base_catalog.where(where)
        else:
            catalog = base_catalog
        # store the selection
        out[num] = catalog
        added.add(num)
    # assign visibilities to individual selections if a mapping was given
    # only allow visibilities for selections added here
    if isinstance(visibility, Mapping):
        for key, vmap in visibility.items():
            num = int(key)
            if num not in added:
                msg = f"Invalid value: unknown selection '{num}'"
                raise ValueError(msg)
            try:
                out[num].visibility = read_vmap(getpath(vmap))
            except (TypeError, ValueError, OSError) as exc:
                msg = f"Cannot load visibility: {exc!s}"
                raise ValueError(msg)
    # all done, return `out` unconditionally
    return out


def catalogs_from_config(config):
    """Construct all catalog instances from config."""
    sections = config.subsections("catalogs")
    catalogs = {}
    for label, section in sections.items():
        catalog_from_config(config, section, label, out=catalogs)
    return catalogs


def bins_from_config(config, section):
    """Construct angular bins from config."""

    # dictionary of {spacing: (op, invop)}
    spacings = {
        "linear": (lambda x: x, lambda x: x),
        "log": (np.log10, lambda x: 10**x),
        "sqrt": (np.sqrt, np.square),
        "log1p": (np.log1p, np.expm1),
    }

    # dictionary of known weights
    weights = {
        None,
        "2l+1",
        "l(l+1)",
    }

    bins = config.get(section, "bins", fallback="none")

    if bins == "none":
        return None, None

    binopts = bins.split()

    if not 2 <= len(binopts) <= 3:
        msg = f"{section}: bins should be of the form '<size> <spacing> [<weights>]'"
        raise ValueError(msg)

    n = int(binopts[0])
    s = binopts[1]
    w = binopts[2] if len(binopts) > 2 else None

    if n < 2:
        msg = f"Invalid bin size '{n}' in section {section}"
        raise ValueError(msg)
    if s not in spacings:
        msg = f"Invalid bin spacing '{s}' in section {section}"
        raise ValueError(msg)
    if w is not None and w not in weights:
        msg = f"Invalid bin weights '{w}' in section {section}"
        raise ValueError(msg)

    lmin = config.getint(section, "lmin", fallback=1)
    lmax = config.getint(section, "lmax")

    op, inv = spacings[s]
    arr = inv(np.linspace(op(lmin), op(lmax + 1), n + 1))
    # fix first and last array element to be exact
    arr[0], arr[-1] = lmin, lmax + 1

    return arr, w


def spectrum_from_config(config, section):
    """Construct info dict for angular power spectra from config."""

    options = config[section]

    info = {}
    if "lmax" in options:
        info["lmax"] = options.getint("lmax")
    if "l2max" in options:
        info["l2max"] = options.getint("l2max")
    if "l3max" in options:
        info["l3max"] = options.getint("l3max")
    if "include" in options:
        info["include"] = options.getfilter("include")
    if "exclude" in options:
        info["exclude"] = options.getfilter("exclude")
    if "debias" in options:
        info["debias"] = options.getboolean("debias")
    if "bins" in options:
        info["bins"] = bins_from_config(config, section)

    return info


def spectra_from_config(config):
    """Construct pairs of labels and *kwargs* for angular power spectra."""
    sections = config.subsections("spectra")
    spectra = []
    for label, section in sections.items():
        spectra += [(label, spectrum_from_config(config, section))]
    if not spectra:
        spectra += [(None, {})]
    return spectra


# the type of a single path
Path = Union[str, os.PathLike]

# the type of one or multiple paths
Paths = Union[Path, Iterable[Path]]

# the type of loader functions for load_xyz()
ConfigLoader = Callable[[Paths], ConfigParser]


def configloader(path: Paths) -> ConfigParser:
    """Load a config file using configparser."""

    if isinstance(path, (str, os.PathLike)):
        path = (path,)

    config = ConfigParser()
    for p in path:
        with open(p) as fp:
            config.read_file(fp)
    return config


# this constant sets the default loader
DEFAULT_LOADER = configloader


def map_all_selections(
    fields: Mapping[str, Field],
    config: ConfigParser,
    logger: logging.Logger,
    progress: bool,
) -> Iterator:
    """Iteratively map the catalogues defined in config."""

    from .maps import map_catalogs

    # load catalogues to process
    catalogs = catalogs_from_config(config)

    logger.info("fields %s", ", ".join(map(repr, fields)))

    # process each catalogue separately into maps
    for key, catalog in catalogs.items():
        logger.info(
            "%s%s",
            f"catalog {catalog.label!r}, " if catalog.label else "",
            f"selection {key}",
        )

        # maps for single catalogue
        yield map_catalogs(
            fields,
            {key: catalog},
            parallel=True,  # process everything at this level in one go
            progress=progress,
        )


def load_all_maps(paths: Paths, logger: logging.Logger) -> Iterator:
    """Iterate over MapFits from a path or list of paths."""

    from .io import MapFits

    # make iterable if single path is given
    if isinstance(paths, (str, os.PathLike)):
        paths = (paths,)

    for path in paths:
        logger.info("reading maps from %s", path)
        yield MapFits(path, clobber=False)


def maps(
    path: Path,
    *,
    files: Paths,
    logger: logging.Logger,
    loader: ConfigLoader = DEFAULT_LOADER,
    progress: bool,
) -> None:
    """compute maps"""

    from .io import MapFits

    # load the config file, this contains the maps definition
    logger.info("reading configuration from %s", files)
    config = loader(files)

    # construct fields for mapping
    fields = fields_from_config(config)

    # iterator over the individual maps
    # this generates maps on the fly
    itermaps = map_all_selections(fields, config, logger, progress)

    # output goes into a FITS-backed tocdict so we don't fill memory up
    out = MapFits(path, clobber=True)

    # iterate over maps, keeping only one in memory at a time
    for maps in itermaps:
        # write to disk
        logger.info("writing maps to %s", path)
        out.update(maps)
        # forget maps before next turn to free some memory
        del maps


def alms(
    path: Path,
    *,
    files: Paths | None,
    maps: Paths | None,
    healpix_datapath: Path | None = None,
    logger: logging.Logger,
    loader: ConfigLoader = DEFAULT_LOADER,
    progress: bool,
) -> None:
    """compute spherical harmonic coefficients

    Compute spherical harmonic coefficients (alms) from catalogues or
    maps.  For catalogue input, the maps for each selection are created
    in memory and discarded after its alms have been computed.

    """

    from .io import AlmFits
    from .maps import Healpix, transform_maps

    # load the config file, this contains alms setting and maps definition
    logger.info("reading configuration from %s", files)
    config = loader(files)

    # set the HEALPix datapath
    if healpix_datapath is not None:
        Healpix.DATAPATH = healpix_datapath

    # construct fields to get mappers for transform
    fields = fields_from_config(config)

    # process either catalogues or maps
    # everything is loaded via iterators to keep memory use low
    itermaps: Iterator
    if maps:
        itermaps = load_all_maps(maps, logger)
    else:
        itermaps = map_all_selections(fields, config, logger, progress)

    # output goes into a FITS-backed tocdict so we don't fill up memory
    logger.info("writing alms to %s", path)
    out = AlmFits(path, clobber=True)

    # iterate over maps and transform each
    for maps in itermaps:
        logger.info("transforming %d maps", len(maps))
        transform_maps(
            fields,
            maps,
            progress=progress,
            out=out,
        )
        del maps


def chained_alms(alms: Paths | None) -> Mapping[Any, NDArray] | None:
    """Return a ChainMap of AlmFits from all input alms, or None."""
    from collections import ChainMap

    from .io import AlmFits

    if alms is None:
        return None
    return ChainMap(*(AlmFits(alm) for alm in reversed(alms)))


def spectra(
    path: Path,
    *,
    files: Paths,
    alms: Paths,
    alms2: Paths | None,
    logger: logging.Logger,
    loader: ConfigLoader = DEFAULT_LOADER,
    progress: bool,
) -> None:
    """compute angular power spectra"""

    from .io import ClsFits
    from .twopoint import angular_power_spectra

    # load the config file, this contains angular binning settings
    logger.info("reading configuration from %s", files)
    config = loader(files)

    # collect angular power spectra settings from config
    spectra = spectra_from_config(config)

    # link all alms together
    all_alms, all_alms2 = chained_alms(alms), chained_alms(alms2)

    # create an empty cls file, then fill it iteratively with alm combinations
    out = ClsFits(path, clobber=True)

    total = 0
    logger.info("using %d set(s) of alms", len(all_alms))
    if all_alms2 is not None:
        logger.info("using %d set(s) of cross-alms", len(all_alms2))
    for label, info in spectra:
        logger.info(
            "computing %s spectra",
            repr(label) if label is not None else "all",
        )
        # angular binning
        if info.get("bins") is not None:
            bins, weights = info["bins"]
        else:
            bins, weights = None, None
        # compute spectra
        angular_power_spectra(
            all_alms,
            all_alms2,
            lmax=info.get("lmax"),
            debias=info.get("debias", True),
            bins=bins,
            weights=weights,
            include=info.get("include"),
            exclude=info.get("exclude"),
            out=out,
        )
        logger.info("-> added %d spectra, total is now %d", len(out) - total, len(out))
        total = len(out)
    logger.info("finished computing %d spectra", total)


def mixmats(
    path: Path,
    *,
    files: Paths,
    alms: Paths,
    alms2: Paths | None,
    logger: logging.Logger,
    loader: ConfigLoader = DEFAULT_LOADER,
    progress: bool,
) -> None:
    """compute mixing matrices"""

    from .fields import get_masks
    from .io import MmsFits
    from .twopoint import angular_power_spectra, mixing_matrices

    # load the config file, this contains angular binning settings
    logger.info("reading configuration from %s", files)
    config = loader(files)

    # collect the defined fields from config
    fields = fields_from_config(config)

    # collect angular power spectra settings from config
    spectra = spectra_from_config(config)

    # link all alms together
    all_alms, all_alms2 = chained_alms(alms), chained_alms(alms2)

    # create an empty mms file, then fill it iteratively
    out = MmsFits(path, clobber=True)

    total = 0
    logger.info("using %d set(s) of alms", len(all_alms))
    if all_alms2 is not None:
        logger.info("using %d set(s) of cross-alms", len(all_alms2))
    for label, info in spectra:
        # get mask combinations for fields included in these spectra
        include, exclude = info.get("include"), info.get("exclude")
        include_masks = get_masks(
            fields,
            comb=2,
            include=include,
            exclude=exclude,
            append_eb=True,
        )
        if not include_masks:
            logger.info(
                "missing masks for %s spectra, skipping...",
                repr(label) if label is not None else "all",
            )
            continue
        logger.info(
            "computing %s mask spectra for %s",
            repr(label) if label is not None else "all",
            ", ".join(map(str, include_masks)),
        )
        # determine the various lmax values
        lmax, l2max, l3max = info.get("lmax"), info.get("l2max"), info.get("l3max")
        # angular binning, to be applied to rows of mixing matrices
        if info.get("bins") is not None:
            bins, weights = info["bins"]
        else:
            bins, weights = None, None
        # compute spectra of masks
        mask_cls = angular_power_spectra(
            all_alms,
            all_alms2,
            lmax=l3max,
            debias=info.get("debias", True),
            include=include_masks,
        )
        # now compute the mixing matrices from these spectra
        logger.info(
            "computing %s mixing matrices from %d spectra",
            repr(label) if label is not None else "all",
            len(mask_cls),
        )
        mixing_matrices(
            fields,
            mask_cls,
            l1max=lmax,
            l2max=l2max,
            l3max=l3max,
            bins=bins,
            weights=weights,
            progress=progress,
            out=out,
        )
        logger.info("-> added %d mixmats, total is now %d", len(out) - total, len(out))
        total = len(out)
        del mask_cls
    logger.info("finished computing %d mixing matrices", total)


class MainFormatter(argparse.RawDescriptionHelpFormatter):
    """Formatter that keeps order of arguments for usage."""

    def add_usage(self, usage, actions, groups, prefix=None):
        self.actions = actions
        super().add_usage(usage, actions, groups, prefix)

    def _format_actions_usage(self, actions, groups):
        return super()._format_actions_usage(self.actions, groups)


def main():
    """Main method of the `heracles` command.

    Parses arguments and calls the appropriate subcommand.

    """

    def add_command(func):
        """Create a subparser for a command given by a function."""

        name = func.__name__
        doc = func.__doc__.strip()
        help_, _, description = doc.partition("\n")

        parser = commands.add_parser(
            name,
            help=help_,
            description=description,
            parents=[cmd_parser],
            formatter_class=MainFormatter,
        )
        parser.set_defaults(cmd=func)
        return parser

    # common parser for all subcommands
    cmd_parser = argparse.ArgumentParser(
        add_help=False,
    )
    cmd_parser.add_argument(
        "-c",
        "--config",
        help="configuration file (can be repeated)",
        metavar="<config>",
        action="append",
        dest="files",
    )
    cmd_parser.add_argument(
        "--no-progress",
        help="do not show progress bars",
        action="store_false",
        dest="progress",
    )

    # main parser for CLI invokation
    main_parser = argparse.ArgumentParser(
        prog="heracles",
        epilog="Made in the Euclid Science Ground Segment",
        formatter_class=MainFormatter,
    )
    main_parser.set_defaults(cmd=None)

    commands = main_parser.add_subparsers(
        title="commands",
        metavar="<command>",
        help="the processing step to carry out",
    )

    ########
    # maps #
    ########

    parser = add_command(maps)
    group = parser.add_argument_group("output")
    group.add_argument(
        "path",
        help="output FITS file for maps",
        metavar="<maps>",
    )

    ########
    # alms #
    ########

    parser = add_command(alms)
    parser.add_argument(
        "--healpix-datapath",
        help="path to HEALPix data files",
        metavar="<path>",
    )
    group = parser.add_argument_group("output")
    group.add_argument(
        "path",
        help="output FITS file for alms",
        metavar="<alms>",
    )
    group = parser.add_argument_group("inputs")
    group.add_argument(
        "maps",
        nargs="*",
        default=None,
        help="input FITS file(s) for maps",
        metavar="<maps>",
    )

    ###########
    # spectra #
    ###########

    parser = add_command(spectra)
    group = parser.add_argument_group("output")
    group.add_argument(
        "path",
        help="output FITS file for spectra",
        metavar="<spectra>",
    )
    group = parser.add_argument_group("inputs")
    group.add_argument(
        "alms",
        nargs="+",
        help="input FITS file(s) for alms",
        metavar="<alms>",
    )
    group.add_argument(
        "-X",
        nargs="+",
        help="input FITS file(s) for cross-spectra",
        metavar="<alms>",
        dest="alms2",
    )

    ###########
    # mixmats #
    ###########

    parser = add_command(mixmats)
    group = parser.add_argument_group("output")
    group.add_argument(
        "path",
        help="output FITS file for mixing matrices",
        metavar="<mixmats>",
    )
    group = parser.add_argument_group("inputs")
    group.add_argument(
        "alms",
        nargs="+",
        help="input FITS file(s) for alms",
        metavar="<alms>",
    )
    group.add_argument(
        "-X",
        nargs="+",
        help="input FITS file(s) for cross-spectra",
        metavar="<alms>",
        dest="alms2",
    )

    #######
    # run #
    #######

    args = main_parser.parse_args()

    # show full help if no command is given
    if args.cmd is None:
        main_parser.print_help()
        return 1

    # fix default config
    if not args.files:
        args.files = ["heracles.cfg"]

    # get keyword args
    kwargs = vars(args)
    cmd = kwargs.pop("cmd")

    # set up logger for CLI output
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    try:
        cmd(**kwargs, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Exception", exc_info=exc)
        logger.error(f"ERROR: {exc!s}")
        return 1
    else:
        return 0
