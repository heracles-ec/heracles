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
'''utility functions for plotting'''

from collections import defaultdict
from collections.abc import Mapping
from itertools import chain, count, cycle
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


DEFAULT_CYCLER = cycler(linestyle=['-', '--', ':', '-.'])


def _dont_draw_zero_tick(tick):
    '''custom draw function for ticks that does not draw zero'''
    draw = tick.draw

    def wrap(*args, **kwargs):
        if tick.get_loc() == 0.:
            tick.set_label1('')
            tick.set_label2('')
        draw(*args, **kwargs)

    return wrap


def _pad_ylim(ymin, ymax):
    '''pad the y axis range depending on signs'''
    return (ymin * 10**(-np.sign(ymin)/2), ymax * 10**(np.sign(ymax)/2))


def postage_stamps(plot=None, transpose=None, *, scale=None,
                   shift_transpose=0, stampsize=1.0, hatch_empty=False,
                   linscale=0.01, cycler=None):
    '''create a postage stamp plot for cls'''

    if cycler is None:
        cycler = DEFAULT_CYCLER

    if plot is None and transpose is None:
        raise ValueError('missing plot data')

    if isinstance(plot, Mapping):
        plot = [plot]
    if isinstance(transpose, Mapping):
        transpose = [transpose]

    keys = {k: None for x in plot for k in x} if plot is not None else {}
    trkeys = {} if transpose is None else {k: None for x in transpose for k in x}
    stamps = sorted(
        ({key[-2:] for key in keys} | {key[-2:][::-1] for key in trkeys})
    )

    sx = list({i for i, _ in stamps})
    sy = list({j for _, j in stamps})

    nx = len(sx)
    ny = len(sy)

    if trkeys:
        ny += shift_transpose

    fig, axes = plt.subplots(nx, ny, figsize=(ny*stampsize, nx*stampsize),
                             squeeze=False, sharex=False, sharey=False)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    prop = defaultdict(lambda: cycle(prop_cycle))

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = 0, 0
    trymin, trymax = 0, 0

    axidx = set()
    traxidx = set()

    for n, key in enumerate(chain(keys, trkeys)):

        ki, kj, i, j = key

        if n < len(keys):
            idx = (sx.index(j), sy.index(i))
            cls = (x.get(key) for x in plot)
            axidx.add(idx)
        else:
            idx = (sx.index(i), sy.index(j)+shift_transpose)
            cls = (x.get(key) for x in transpose)
            traxidx.add(idx)

        # axis for plotting this key
        ax = axes[idx]

        # outer property cycle for this axis
        oprop = next(prop[idx])

        # label for first plot only, set to None after
        label = f'${ki}^{{{i}}} \\times {kj}^{{{j}}}$'

        for m, cl, iprop in zip(count(), cls, cycle(cycler)):
            if cl is None:
                continue

            cl = np.asanyarray(cl)

            if cl.dtype.names is None:
                ell = np.arange(len(cl), dtype=float)
            else:
                ell, cl = cl['L'].astype(float), cl['CL']

            if scale is None:
                pass
            elif scale == '2l+1':
                cl = (2*ell+1)*cl
            else:
                cl = scale*cl

            xmin = np.nanmin(ell, initial=xmin)
            xmax = np.nanmax(ell, initial=xmax)
            if n < len(keys):
                ymin = np.nanmin(cl, initial=ymin)
                ymax = np.nanmax(cl, initial=ymax)
            else:
                trymin = np.nanmin(cl, initial=trymin)
                trymax = np.nanmax(cl, initial=trymax)

            ax.plot(ell, cl, lw=1.5, label=label, **{**oprop, **iprop})

            # prevent multiple labels with same colour
            label = None

    ymin, ymax = _pad_ylim(ymin, ymax)
    ylin = 10**np.ceil(np.log10(max(abs(ymin), abs(ymax))*linscale))

    if trkeys:
        trymin, trymax = _pad_ylim(trymin, trymax)
        trylin = 10**np.ceil(np.log10(max(abs(trymin), abs(trymax))*linscale))

    # scale the axes and transpose axes
    for n, idx in enumerate(chain(axidx, traxidx)):

        if n < len(axidx):
            ymin_, ymax_, ylin_ = ymin, ymax, ylin
        else:
            ymin_, ymax_, ylin_ = trymin, trymax, trylin

        ax = axes[idx]

        ax.axhline(0., c='k', lw=0.8, ls='--')

        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=True,
            bottom=True,
            left=True,
            right=True,
            labeltop=idx in [(0, 0), (0, ny - 1)],
            labelbottom=idx in [(nx - 1, 0), (nx - 1, ny - 1)],
            labelleft=idx in [(0, 0), (nx - 1, 0)],
            labelright=idx in [(0, ny - 1), (nx - 1, ny - 1)],
        )
        ax.set_axisbelow(False)

        leg = ax.legend(frameon=True, edgecolor='none', framealpha=0.8,
                        fontsize=8, labelcolor='linecolor', handlelength=0,
                        handletextpad=0, borderpad=0, borderaxespad=0.5,
                        labelspacing=0)
        try:
            hnds = leg.legend_handles
        except AttributeError:
            hnds = leg.legendHandles
        for hnd in hnds:
            hnd.set_visible(False)
        leg.set_zorder(2)

        ax.set_xlim(xmin, xmax)
        ax.set_xscale('symlog', linthresh=10, linscale=0.45,
                      subs=[2, 3, 4, 5, 6, 7, 8, 9])

        ax.set_ylim(ymin_, ymax_)
        ax.set_yscale('symlog', linthresh=ylin_, linscale=0.45,
                      subs=[2, 3, 4, 5, 6, 7, 8, 9])

        for tick in ax.xaxis.get_major_ticks():
            tick.draw = _dont_draw_zero_tick(tick)
        for tick in ax.yaxis.get_major_ticks():
            tick.draw = _dont_draw_zero_tick(tick)

    # fill empty axes
    for ax in axes.ravel():
        if not ax.has_data():
            if hatch_empty:

                hatch = hatch_empty if isinstance(hatch_empty, str) else '/////'
                ax.patch.set_facecolor('none')
                ax.patch.set_edgecolor('k')
                ax.patch.set_hatch(hatch)
                ax.patch.set_alpha(0.8)
                ax.tick_params(axis='both', which='both',
                               top=False, bottom=False,
                               left=False, right=False,
                               labeltop=False, labelbottom=False,
                               labelleft=False, labelright=False)
            else:
                ax.axis('off')

    fig.tight_layout(pad=0.)
    fig.subplots_adjust(wspace=0, hspace=0)

    # adjust figure size for spacing
    fig.set_size_inches(
        ny*stampsize/(fig.subplotpars.right - fig.subplotpars.left),
        nx*stampsize/(fig.subplotpars.top - fig.subplotpars.bottom),
    )

    return fig
