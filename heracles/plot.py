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
"""utility functions for plotting"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import chain, count, cycle

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

DEFAULT_CYCLER = cycler(linestyle=["-", "--", ":", "-."])


def _dont_draw_zero_tick(tick):
    """custom draw function for ticks that does not draw zero"""
    draw = tick.draw

    def wrap(*args, **kwargs):
        if tick.get_loc() == 0.0:
            tick.set_label1("")
            tick.set_label2("")
        draw(*args, **kwargs)

    return wrap


def _pad_xlim(xmin, xmax, margin=0.05):
    """pad the x axis range depending on signs"""
    f = (xmax / xmin) ** margin
    return xmin / f, xmax * f


def _pad_ylim(ymin, ymax):
    """pad the y axis range depending on signs"""
    return (ymin * 10 ** (-np.sign(ymin) / 2), ymax * 10 ** (np.sign(ymax) / 2))


def postage_stamps(
    plot=None,
    transpose=None,
    *,
    scale=None,
    trxshift=0,
    tryshift=0,
    stampsize=1.0,
    space=0.05,
    hatch_empty=False,
    linscale=0.01,
    cycler=None,
    group=True,
):
    """create a postage stamp plot for cls"""

    if cycler is None:
        cycler = DEFAULT_CYCLER

    if plot is None and transpose is None:
        msg = "missing plot data"
        raise ValueError(msg)

    if isinstance(plot, Mapping):
        plot = [plot]
    if isinstance(transpose, Mapping):
        transpose = [transpose]

    if plot is not None:
        keys = {k: None for x in plot for k in x}
    else:
        keys = {}
    if transpose is not None:
        trkeys = {k: None for x in transpose for k in x}
    else:
        trkeys = {}

    # either group the plots by last two indices or not
    # uses Counter as a set that remembers order
    if group:
        si = list(Counter(key[-2] for key in keys))
        sj = list(Counter(key[-1] for key in keys))
        ti = list(Counter(key[-2] for key in trkeys))
        tj = list(Counter(key[-1] for key in trkeys))
    else:
        si = sj = list(Counter(key[k::2] for key in keys for k in (0, 1)))
        ti = tj = list(Counter(key[k::2] for key in trkeys for k in (0, 1)))

    nx = max(len(si), len(tj))
    ny = max(len(sj), len(ti))

    if trkeys:
        nx += trxshift
        ny += tryshift

    figw = (nx + (nx - 1) * space) * stampsize
    figh = (ny + (ny - 1) * space) * stampsize

    fig, axes = plt.subplots(
        ny,
        nx,
        figsize=(figw, figh),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    prop = defaultdict(lambda: cycle(prop_cycle))

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = 0, 0
    trymin, trymax = 0, 0

    axidx = set()
    traxidx = set()

    for n, key in enumerate(chain(keys, trkeys)):
        ki, kj, i, j = key

        if n < len(keys):
            if group:
                idx_y, idx_x = sj.index(j), si.index(i)
            else:
                idx_x, idx_y = sorted([si.index((ki, i)), sj.index((kj, j))])
            idx = (idx_y + tryshift, idx_x)
            cls = (x.get(key) for x in plot)
            axidx.add(idx)
        else:
            if group:
                idx_y, idx_x = ti.index(i), tj.index(j)
            else:
                idx_y, idx_x = sorted([ti.index((ki, i)), tj.index((kj, j))])
            idx = (idx_y, idx_x + trxshift)
            cls = (x.get(key) for x in transpose)
            traxidx.add(idx)

        # axis for plotting this key
        ax = axes[idx]

        # outer property cycle for this axis
        oprop = next(prop[idx])

        # label for first plot only, set to None after
        label = f"${ki}^{{{i}}} \\times {kj}^{{{j}}}$"

        for m, cl, iprop in zip(count(), cls, cycle(cycler)):
            if cl is None:
                continue

            cl = np.asanyarray(cl)

            if cl.dtype.names is None:
                ell = np.arange(len(cl), dtype=float)
                err = None
            else:
                ell = cl["L"].astype(float)
                err = cl["ERR"] if "ERR" in cl.dtype.names else None
                cl = cl["CL"]

            if scale is None:
                pass
            elif scale == "2l+1":
                cl = (2 * ell + 1) * cl
            else:
                cl = scale * cl

            xmin = np.nanmin(ell[ell != 0], initial=xmin)
            xmax = np.nanmax(ell[ell != 0], initial=xmax)
            if n < len(keys):
                if err is None:
                    ymin = np.nanmin(cl, initial=ymin)
                    ymax = np.nanmax(cl, initial=ymax)
                else:
                    ymin = np.nanmin(cl - err, initial=ymin)
                    ymax = np.nanmax(cl + err, initial=ymax)
            else:
                if err is None:
                    trymin = np.nanmin(cl, initial=trymin)
                    trymax = np.nanmax(cl, initial=trymax)
                else:
                    trymin = np.nanmin(cl - err, initial=trymin)
                    trymax = np.nanmax(cl + err, initial=trymax)

            lines = ax.plot(
                ell,
                cl,
                label=label,
                zorder=2 + 0.4 / (n + 1),
                **{**oprop, **iprop},
            )

            if err is not None:
                ax.fill_between(
                    ell,
                    cl - err,
                    cl + err,
                    ec="none",
                    fc=lines[-1].get_color(),
                    alpha=0.2,
                )

            # prevent multiple labels with same colour
            label = None

    xmin, xmax = _pad_xlim(xmin, xmax)
    if keys:
        ymin, ymax = _pad_ylim(ymin, ymax)
        ylin = 10 ** np.ceil(np.log10(max(abs(ymin), abs(ymax)) * linscale))
    if trkeys:
        trymin, trymax = _pad_ylim(trymin, trymax)
        trylin = 10 ** np.ceil(np.log10(max(abs(trymin), abs(trymax)) * linscale))

    # scale the axes and transpose axes
    for n, idx in enumerate(chain(axidx, traxidx)):
        if n < len(axidx):
            ymin_, ymax_, ylin_ = ymin, ymax, ylin
        else:
            ymin_, ymax_, ylin_ = trymin, trymax, trylin

        ax = axes[idx]

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            labeltop=(idx[0] == 0),
            labelbottom=(idx[0] == ny - 1),
            labelleft=(idx[1] == 0),
            labelright=(idx[1] == nx - 1),
        )
        ax.set_axisbelow(False)

        leg = ax.legend(
            frameon=True,
            edgecolor="none",
            framealpha=0.8,
            fontsize=8,
            labelcolor="linecolor",
            handlelength=0,
            handletextpad=0,
            borderpad=0,
            borderaxespad=0.5,
            labelspacing=0,
        )
        try:
            hnds = leg.legend_handles
        except AttributeError:
            hnds = leg.legendHandles
        for hnd in hnds:
            hnd.set_visible(False)
        leg.set_zorder(3)

        ax.set_xlim(xmin, xmax)
        ax.set_xscale("log")
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99)

        ax.set_ylim(ymin_, ymax_)
        ax.set_yscale(
            "symlog",
            linthresh=ylin_,
            linscale=0.45,
            subs=[2, 3, 4, 5, 6, 7, 8, 9],
        )

        for tick in ax.xaxis.get_major_ticks():
            tick.draw = _dont_draw_zero_tick(tick)
        for tick in ax.yaxis.get_major_ticks():
            tick.draw = _dont_draw_zero_tick(tick)

    # fill empty axes
    for i, ax in enumerate(axes.ravel()):
        if not ax.has_data():
            if hatch_empty:
                if isinstance(hatch_empty, str):
                    hatch = hatch_empty
                else:
                    hatch = "/////"

                ax.patch.set_facecolor("none")
                ax.patch.set_edgecolor("k")
                ax.patch.set_hatch(hatch)
                ax.patch.set_alpha(0.8)
                ax.tick_params(
                    axis="both",
                    which="both",
                    top=False,
                    bottom=False,
                    left=False,
                    right=False,
                    labeltop=False,
                    labelbottom=False,
                    labelleft=False,
                    labelright=False,
                )
            else:
                ax.axis("off")

    fig.tight_layout(pad=0.0)
    fig.subplots_adjust(wspace=space, hspace=space)

    # adjust figure size for spacing
    fig.set_size_inches(
        figw / (fig.subplotpars.right - fig.subplotpars.left),
        figh / (fig.subplotpars.top - fig.subplotpars.bottom),
    )

    return fig
