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

    if plot is not None:
        keys = {k: None for x in plot for k in x}
    else:
        keys = {}
    if transpose is not None:
        trkeys = {k: None for x in transpose for k in x}
    else:
        trkeys = {}

    stamps = sorted(set(key[-2:] for key in keys)
                    | set(key[-2:][::-1] for key in trkeys))

    sx = list(set(i for i, _ in stamps))
    sy = list(set(j for _, j in stamps))

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
                ell = np.arange(len(cl))
            else:
                ell, cl = cl['L'], cl['CL']

            if scale is None:
                pass
            elif scale == '2l+1':
                cl = (2*ell+1)*cl
            else:
                cl = scale*cl

            xmin, xmax = min(xmin, np.min(ell)), max(xmax, np.max(ell))
            clmin, clmax = np.min(cl), np.max(cl)
            if n < len(keys):
                ymin, ymax = min(ymin, clmin), max(ymax, clmax)
            else:
                trymin, trymax = min(trymin, clmin), max(trymax, clmax)

            ax.plot(ell, cl, lw=1.5, label=label, **{**oprop, **iprop})

            # prevent multiple labels with same colour
            label = None

    ymin, ymax = _pad_ylim(ymin, ymax)
    trymin, trymax = _pad_ylim(trymin, trymax)

    ylin = 10**np.ceil(np.log10(max(abs(ymin), abs(ymax)) * linscale))
    trylin = 10**np.ceil(np.log10(max(abs(trymin), abs(trymax)) * linscale))

    # scale the axes and transpose axes
    for n, idx in enumerate(chain(axidx, traxidx)):

        if n < len(axidx):
            ymin_, ymax_, ylin_ = ymin, ymax, ylin
        else:
            ymin_, ymax_, ylin_ = trymin, trymax, trylin

        ax = axes[idx]

        ax.axhline(0., c='k', lw=0.8, ls='--')

        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True,
                       labeltop=(idx == (0, 0) or idx == (0, ny-1)),
                       labelbottom=(idx == (nx-1, 0) or idx == (nx-1, ny-1)),
                       labelleft=(idx == (0, 0) or idx == (nx-1, 0)),
                       labelright=(idx == (0, ny-1) or idx == (nx-1, ny-1)))
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
    for i, ax in enumerate(axes.ravel()):

        if not ax.has_data():
            if hatch_empty:

                if isinstance(hatch_empty, str):
                    hatch = hatch_empty
                else:
                    hatch = '/////'

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

    return fig
