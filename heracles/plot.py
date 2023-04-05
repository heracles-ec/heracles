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
            tick.set_label('')
        draw(*args, **kwargs)

    return wrap


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
                             squeeze=False, sharex=True, sharey=True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    prop = defaultdict(lambda: cycle(prop_cycle))

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = 0, 0

    for n, key in enumerate(chain(keys, trkeys)):

        ki, kj, i, j = key

        if n < len(keys):
            idx = (sx.index(j), sy.index(i))
            cls = (x.get(key) for x in plot)
        else:
            idx = (sx.index(i), sy.index(j)+shift_transpose)
            cls = (x.get(key) for x in transpose)

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
            ymin, ymax = min(ymin, np.min(cl)), max(ymax, np.max(cl))

            ax.plot(ell, cl, lw=1.5, label=label, **oprop, **iprop)

            # prevent multiple labels with same colour
            label = None

    ylin = 10**np.ceil(np.log10(max(abs(ymin), abs(ymax)) * linscale))

    for i, ax in enumerate(axes.ravel()):

        if ax.has_data():

            ax.axhline(0., c='k', lw=0.8)

            ax.tick_params(axis='both', which='both', direction='in',
                           top=True, bottom=True, left=True, right=True,
                           labeltop=(i == 0), labelbottom=False,
                           labelleft=(i == 0), labelright=False)

            ax.set_axisbelow(False)

            leg = ax.legend(frameon=True, edgecolor='none', framealpha=0.8,
                            fontsize=8, labelcolor='linecolor', handlelength=0,
                            borderpad=0, borderaxespad=0.5, labelspacing=0)
            leg.set_zorder(2)

            ax.set_xlim(xmin, xmax)
            ax.set_xscale('symlog', linthresh=10, linscale=0.45,
                          subs=[2, 3, 4, 5, 6, 7, 8, 9])

            ax.set_yscale('symlog', linthresh=ylin, linscale=0.45,
                          subs=[2, 3, 4, 5, 6, 7, 8, 9])

            for tick in ax.yaxis.get_major_ticks():
                tick.draw = _dont_draw_zero_tick(tick)

        else:

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
