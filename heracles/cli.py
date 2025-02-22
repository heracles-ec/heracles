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
"""The Heracles command line interface."""

from __future__ import annotations

import argparse
import gc
import sys
import textwrap
import traceback

import heracles.config
import heracles.io

try:
    import tomllib
except ModuleNotFoundError:
    from heracles.core import external_dependency_explainer

    with external_dependency_explainer:
        import tomli as tomllib  # type: ignore[no-redef]

TYPE_CHECKING = False
if TYPE_CHECKING:
    from heracles.catalog import CatalogView
    from heracles.config import Config


class MainFormatter(argparse.RawDescriptionHelpFormatter):
    """Formatter that keeps order of arguments for usage."""

    def add_usage(self, usage, actions, groups, prefix=None):
        self.actions = actions
        super().add_usage(usage, actions, groups, prefix)

    def _format_actions_usage(self, actions, groups):
        return super()._format_actions_usage(self.actions, groups)


def read_config(path: str | None = None) -> Config:
    """
    Read a config file.
    """
    if path is None:
        path = "heracles.toml"
        use_default = True
    else:
        use_default = False

    try:
        with open(path, "rb") as fp:
            config = tomllib.load(fp)
    except FileNotFoundError as exc:
        if use_default and hasattr(exc, "add_note"):
            exc.add_note(
                "(Hint: it looks like Heracles cannot load its default "
                "configuration file. Did you forget to create one?)"
            )
        raise

    return heracles.config.load(config)


def _maps_alms_internal(
    *,
    maps_path: str | None = None,
    alms_path: str | None = None,
    config: str | None = None,
    parallel: bool = False,
) -> None:
    """
    Compute maps and/or alms from catalogues.
    """

    config_loaded = read_config(config)

    if maps_path is not None:
        maps_out = heracles.io.MapFits(maps_path, clobber=True)
    else:
        maps_out = None

    if alms_path is not None:
        alms_out = heracles.io.AlmFits(alms_path, clobber=True)
    else:
        alms_out = None

    for catalog_config in config_loaded.catalogs:
        base_catalog = heracles.FitsCatalog(catalog_config.source)
        base_catalog.label = catalog_config.label

        if catalog_config.visibility is not None:
            base_catalog.visibility = heracles.read_vmap(catalog_config.visibility)

        catalogs: dict[int, CatalogView] = {}
        visibilities: dict[int, str | None] = {}
        for selection in catalog_config.selections:
            catalogs[selection.key] = base_catalog[selection.selection]
            visibilities[selection.key] = selection.visibility

        fields = {key: config_loaded.fields[key] for key in catalog_config.fields}

        for key, catalog in catalogs.items():
            if visibilities[key] is not None:
                catalogs[key].visibility = heracles.read_vmap(visibilities[key])

            # this split will no longer be neccessary when visibilities are
            # lazy-loaded
            if not parallel:
                # process one catalogue
                data = heracles.map_catalogs(
                    fields,
                    {key: catalog},
                    parallel=False,
                )

                # write if asked to
                if maps_out is not None:
                    maps_out.update(data)

                # this catalogue is done, clean up
                catalogs[key].visibility = None
                gc.collect()

                # compute alms if asked to
                if alms_out is not None:
                    heracles.transform(fields, data, out=alms_out)

                # done with data
                del data
                gc.collect()

        if parallel:
            # process all catalogues
            data = heracles.map_catalogs(
                fields,
                catalogs,
                out=maps_out,
                parallel=True,
            )

            # compute alms if asked to
            if alms_out is not None:
                heracles.transform(fields, data, out=alms_out)

        # clean up before next catalogue is processed
        del base_catalog, catalogs, visibilities
        gc.collect()


def maps(
    path: str,
    config: str | None = None,
    parallel: bool = False,
) -> None:
    """map catalogues

    Create maps from input catalogues.

    """
    _maps_alms_internal(maps_path=path, config=config, parallel=parallel)


def alms(
    path: str,
    maps: list[str] | None = None,
    config: str | None = None,
    parallel: bool = False,
) -> None:
    """compute alms from catalogues or maps

    Compute alms from input catalogues or pre-computed maps.

    """
    # if no maps are given, process directly from catalogues
    if maps is None:
        _maps_alms_internal(alms_path=path, config=config, parallel=parallel)
        return

    # load configuration to get fields
    config_loaded = read_config(config)

    # open output FITS
    alms_out = heracles.io.AlmFits(path, clobber=True)

    for maps_path in maps:
        # quick check to see if file is readable
        with open(maps_path) as _fp:
            pass

        # lazy-load the file
        data = heracles.io.MapFits(maps_path)

        # transform this fits
        heracles.transform(config_loaded.fields, data, out=alms_out)

        # clean up before next iteration
        del data


def main() -> int:
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
            description=textwrap.dedent(description),
            parents=[cmd_parser],
            formatter_class=MainFormatter,
        )
        parser.set_defaults(cmd=func)
        return parser

    # common parser for all subcommands
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument("-c", "--config", help="configuration file")

    # main parser for CLI invokation
    main_parser = argparse.ArgumentParser(
        prog="heracles",
        description=textwrap.dedent(
            """
            This is Heracles — Harmonic-space statistics on the sphere.

            To run a command, use `heracles <cmd>`.
            To show help for a command, use `heracles <cmd> --help`.
            """
        ),
        epilog="Made in the Euclid Science Ground Segment",
        formatter_class=MainFormatter,
    )
    main_parser.set_defaults(cmd=None)

    commands = main_parser.add_subparsers(
        title="commands",
        metavar="<cmd>",
        help="command to run",
    )

    ########
    # maps #
    ########

    maps_parser = add_command(maps)
    maps_parser.add_argument(
        "--parallel",
        action="store_true",
        help="process all maps of a catalogue in parallel",
    )
    maps_parser.add_argument(
        "path",
        help="output FITS file for maps",
    )

    ########
    # alms #
    ########

    alms_parser = add_command(alms)
    alms_parser.add_argument(
        "--parallel",
        action="store_true",
        help="process all maps of a catalogue in parallel",
    )
    alms_parser.add_argument(
        "path",
        help="output FITS file for alms",
    )
    alms_parser.add_argument(
        "--maps",
        action="append",
        help="transform pre-computed maps",
    )

    #######
    # run #
    #######

    args = main_parser.parse_args()

    # show full help if no command is given
    if args.cmd is None:
        main_parser.print_help()
        return 1

    # get keyword args
    kwargs = dict(args.__dict__)
    cmd = kwargs.pop("cmd")

    try:
        cmd(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(
            textwrap.dedent(
                """
                === ERROR ===

                Heracles crashed with an uncaught exception.

                If you suspect this is a bug, please open an issue at
                https://github.com/heracles-ec/heracles/issues.
                """
            ),
            file=sys.stderr,
            flush=True,
        )
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        print("\n".join(tb), file=sys.stderr, flush=True)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
