import numpy as np


def legendre_p_all(n, x, *, diff_n=0):
    """
    All Legendre polynomials of the first kind up to the specified degree n,
    evaluated at a scalar x, optionally with their first derivatives.
    """
    allP = np.empty(n + 1)
    allP[0] = 1.0
    if n >= 1:
        allP[1] = x
    for l in range(1, n):
        allP[l + 1] = ((2 * l + 1) * x * allP[l] - l * allP[l - 1]) / (l + 1)
    if diff_n == 0:
        return allP
    assert diff_n == 1, "only diff_n=1 is supported"
    ls = np.arange(1, n + 1)
    alldP = np.zeros_like(allP)
    alldP[1:] = ls * (allP[:-1] - x * allP[1:]) / (1 - x**2)
    return allP, alldP


try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace

from .progress import NoProgress, Progress
from .result import get_result_array

gauss_legendre = None
_gauss_legendre_cache = {}


def _cached_gauss_legendre(npoints, cache=True):
    """
    Gauss-Legendre quadrature nodes/weights for `npoints` points on [-1, 1].
    """
    key = npoints
    if cache and key in _gauss_legendre_cache:
        return _gauss_legendre_cache[key]
    else:
        if gauss_legendre is not None:
            xvals = np.empty(npoints)
            weights = np.empty(npoints)
            gauss_legendre(xvals, weights, npoints)
            xvals.flags.writeable = False
            weights.flags.writeable = False
        else:
            xvals, weights = np.polynomial.legendre.leggauss(npoints)
            xvals.flags.writeable = False
            weights.flags.writeable = False
        if cache:
            _gauss_legendre_cache[key] = xvals, weights
        return xvals, weights


def legendre_funcs(lmax, x, spin, lfacs=None, lfacs2=None, lrootfacs=None):
    """
    Utility function to return the Legendre/Wigner-d functions needed to
    transform a Cl of the given `spin` to/from a correlation function, for
    all :math:`\ell` up to lmax. Note that the spin functions start at
    :math:`\ell=2`, so are shorter than the spin (0, 0) case.

    Only spin (0, 0), (0, 2)/(2, 0), and (2, 2) are supported.

    :param lmax: maximum :math:`\ell`
    :param x: scalar value of :math:`\cos(\theta)` at which to evaluate
    :param spin: (s1, s2) spin of the field pair -- selects which functions
        are computed
    :param lfacs: optional pre-computed :math:`\ell(\ell+1)` float array
        (ignored for spin (0, 0))
    :param lfacs2: optional pre-computed :math:`(\ell+2)*(\ell-1)` float array
    :param lrootfacs: optional pre-computed sqrt(lfacs*lfacs2) array
    :return: `P`, starting at :math:`\ell=0`, for spin (0, 0); otherwise
        `(d_{20}, d_{22}, d_{2,-2})`, starting at :math:`\ell=2`
    """
    s1, s2 = spin
    if s1 == 0 and s2 == 0:
        return legendre_p_all(lmax, x)
    elif not ({s1, s2} == {0, 2} or (s1 == 2 and s2 == 2)):
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )

    allP, alldP = legendre_p_all(lmax, x, diff_n=1)
    fac1 = 1 - x
    fac2 = 1 + x

    if lfacs is None:
        ls = np.arange(2, lmax + 1, dtype=np.float64)
        lfacs = ls * (ls + 1)
        lfacs2 = (ls + 2) * (ls - 1)
        lrootfacs = np.sqrt(lfacs * lfacs2)
    P = allP[2:]
    dP = alldP[2:]

    fac = fac1 / fac2
    d22 = (
        ((4 * x - 8) / fac2 + lfacs) * P + 4 * fac * (fac2 + (x - 2) / lfacs) * dP
    ) / lfacs2
    if x > 0.998:
        # for stability use series at small angles (thanks Pavel Motloch)
        d2m2 = np.empty(lmax - 1)
        indser = int(np.sqrt((400.0 + 3 / (1 - x**2)) / 150)) - 1
        d2m2[indser:] = (
            (lfacs[indser:] - (4 * x + 8) / fac1) * P[indser:]
            + 4 / fac * (-fac1 + (x + 2) / lfacs[indser:]) * dP[indser:]
        ) / lfacs2[indser:]
        sin2 = 1 - x**2
        d2m2[:indser] = (
            lfacs[:indser]
            * lfacs2[:indser]
            * sin2**2
            / 7680
            * (20 + sin2 * (16 - lfacs[:indser]))
        )
    else:
        d2m2 = (
            (lfacs - (4 * x + 8) / fac1) * P
            + 4 / fac * (-fac1 + (x + 2) / lfacs) * dP
        ) / lfacs2
    d20 = (2 * x * dP - lfacs * P) / lrootfacs
    return d20, d22, d2m2


def rotate(cl, spin):
    """
    Rotate a Cl (or correlation function) array of the given spin into the
    "+/-" basis used for the real-space transforms in `_cl2corr`/`_corr2cl`.

    spin (2, 2): `cl` is `[[EE, EB], [BE, BB]]`, returns

        [[EE+BB, EE-BB],
         [EB+BE, EB-BE]]

    i.e. sum/difference applied to the diagonal pair (EE, BB) for row 0, and
    to the anti-diagonal pair (EB, BE) for row 1.

    spin (0, 2) or (2, 0): `cl` is `[Ta, Tb]` (e.g. T x E and T x B),
    returns `[Ta+Tb, Ta-Tb]`.

    Not defined for spin (0, 0), since there is nothing to combine.
    """
    if spin == (0, 0):
        raise ValueError("rotate is not defined for spin (0, 0)")
    elif spin in ((0, 2), (2, 0)):
        return np.array([cl[0] + cl[1], cl[0] - cl[1]])
    elif spin == (2, 2):
        EE, EB = cl[0, 0], cl[0, 1]
        BE, BB = cl[1, 0], cl[1, 1]
        return np.array(
            [
                [EE + BB, EE - BB],
                [EB + BE, EB - BE],
            ]
        )
    else:
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )


def unrotate(cl, spin):
    """
    Inverse of `rotate` for the given spin.

    For spin (2, 2), given `[[p_diag, m_diag], [p_anti, m_anti]]` as
    produced by `rotate`, recovers the original `[[EE, EB], [BE, BB]]`. Note
    this is *not* the same as calling `rotate` a second time: `rotate`
    pairs the diagonal and anti-diagonal entries of its input, whereas
    undoing it means un-pairing its own *rows* instead -- a different
    grouping (`rotate(rotate(cl, (2, 2)), (2, 2))` is generally not `2*cl`).

    For spin (0, 2)/(2, 0), `rotate`'s sum/difference of a plain pair *is*
    self-inverse up to a factor of 2, so this is just that same formula,
    halved.

    Not defined for spin (0, 0), since there is nothing to un-combine.
    """
    if spin == (0, 0):
        raise ValueError("unrotate is not defined for spin (0, 0)")
    elif spin in ((0, 2), (2, 0)):
        return np.array([(cl[0] + cl[1]) / 2, (cl[0] - cl[1]) / 2])
    elif spin == (2, 2):
        p_diag, m_diag = cl[0, 0], cl[0, 1]
        p_anti, m_anti = cl[1, 0], cl[1, 1]
        return np.array(
            [
                [(p_diag + m_diag) / 2, (p_anti + m_anti) / 2],
                [(p_anti - m_anti) / 2, (p_diag - m_diag) / 2],
            ]
        )
    else:
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )


def _cl2corr(cl, spin, lmax=None, sampling_factor=1, xvals=None):
    """
    Get the correlation function from the power spectra, evaluated at points
    cos(theta) = xvals, dispatching directly on the spin of `cl` instead of
    always going through a fixed [T, Q+U, Q-U, cross] layout.
    Use roots of Legendre polynomials (np.polynomial.legendre.leggauss) for accurate back integration with corr2cl.
    Note currently does not work at xvals=1 (can easily calculate that as special case!).

    :param cl: Cl array, shape depending on spin: 1D `cl[l]` for spin (0, 0);
        2D `cl[a, l]` (a in 0, 1) for spin (0, 2)/(2, 0); 3D
        `cl[[EE, EB], [BE, BB]][l]` for spin (2, 2). Should include
        :math:`\ell(\ell+1)/2\pi` factors.
    :param spin: (s1, s2) spin of the field pair; only (0, 0), (0, 2)/(2, 0),
        and (2, 2) are supported
    :param lmax: optional maximum L to use from the cl array
    :param sampling_factor: oversampling factor for the quadrature grid,
        ignored if `xvals` is given
    :param xvals: if given, evaluate the correlation function at these
        cos(theta) points directly instead of the Gauss-Legendre quadrature
        grid -- e.g. to evaluate at arbitrary angles, not just quadrature
        nodes (`corr2cl` then cannot be used to transform the result back)
    :return: correlation function array with the same leading shape as `cl`,
        but with the l axis replaced by the quadrature (theta) axis
    """
    cl = np.asarray(cl, dtype=np.float64)

    if lmax is None:
        lmax = cl.shape[-1] - 1

    if xvals is None:
        xvals, _ = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)
    else:
        xvals = np.asarray(xvals, dtype=np.float64)
    ls = np.arange(0, lmax + 1, dtype=np.float64)
    facs = (2 * ls + 1) / (4 * np.pi)

    if spin == (0, 0):
        ct = facs * cl[: lmax + 1]
        corr = np.empty(len(xvals))
        for i, x in enumerate(xvals):
            P = legendre_funcs(lmax, x, spin)
            corr[i] = np.dot(ct, P)
        return corr
    elif spin not in ((0, 2), (2, 0), (2, 2)):
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )

    # For polarization, all arrays start at 2
    ls2 = ls[2:]
    lfacs = ls2 * (ls2 + 1)
    lfacs2 = (ls2 + 2) * (ls2 - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)

    if spin in ((0, 2), (2, 0)):
        # T x spin-2 cross correlation: both combinations use the same d20
        cp, cm = facs[2:] * rotate(cl[:, 2 : lmax + 1], spin)
        corr = np.empty((2, len(xvals)))
        for i, x in enumerate(xvals):
            d20, _, _ = legendre_funcs(lmax, x, spin, lfacs, lfacs2, lrootfacs)
            corr[0, i] = np.dot(cp, d20)
            corr[1, i] = np.dot(cm, d20)
        return corr

    # spin (2, 2): EE/BB use d22/d2m2 on the rotated diagonal pair,
    # EB/BE use d22/d2m2 (negated) on the rotated anti-diagonal pair
    r = rotate(cl, spin)
    cp = facs[2:] * r[0, 0, 2 : lmax + 1]
    cm = facs[2:] * r[0, 1, 2 : lmax + 1]
    icp = facs[2:] * r[1, 1, 2 : lmax + 1]
    icm = facs[2:] * r[1, 0, 2 : lmax + 1]
    corr = np.zeros((2, 2, len(xvals)))
    for i, x in enumerate(xvals):
        _, d22, d2m2 = legendre_funcs(lmax, x, spin, lfacs, lfacs2, lrootfacs)
        corr[0, 0, i] = np.dot(cp, d22)  # EE-like
        corr[1, 1, i] = np.dot(cm, d2m2)  # BB-like
        corr[0, 1, i] = -np.dot(icp, d22)  # EB-like
        corr[1, 0, i] = -np.dot(icm, d2m2)  # BE-like
    return corr


def _corr2cl(corr, spin, lmax=None, sampling_factor=1):
    """
    Transform from correlation functions to power spectra, dispatching
    directly on the spin of `corr` instead of always going through a fixed
    [T, Q+U, Q-U, cross] layout.
    Note that using cl2corr followed by corr2cl is generally very accurate (< 1e-5 relative error) if
    xvals, weights = np.polynomial.legendre.leggauss(lmax+1)

    :param corr: correlation array, mirroring `_cl2corr`'s output shape for
        the given spin (1D, 2D, or 3D -- see `_cl2corr`)
    :param spin: (s1, s2) spin of the field pair; only (0, 0), (0, 2)/(2, 0),
        and (2, 2) are supported
    :param lmax: maximum :math:`\ell` to calculate :math:`C_\ell`
    :param sampling_factor: oversampling factor for the quadrature grid
    :return: Cl array with the same leading shape as `corr`, but with the
        theta axis replaced by the l axis. Includes
        :math:`\ell(\ell+1)/2\pi` factors.
    """
    corr = np.asarray(corr, dtype=np.float64)

    if lmax is None:
        lmax = corr.shape[-1] - 1

    xvals, weights = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)

    if spin == (0, 0):
        cl = np.zeros(lmax + 1)
        for x, weight, c in zip(xvals, weights, corr):
            P = legendre_funcs(lmax, x, spin)
            cl += (weight * c) * P
        return 2 * np.pi * cl
    elif spin not in ((0, 2), (2, 0), (2, 2)):
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )

    # For polarization, all arrays start at 2
    ls = np.arange(2, lmax + 1, dtype=np.float64)
    lfacs = ls * (ls + 1)
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)

    if spin in ((0, 2), (2, 0)):
        clp = np.zeros(lmax + 1)
        clm = np.zeros(lmax + 1)
        for i, (x, weight) in enumerate(zip(xvals, weights)):
            d20, _, _ = legendre_funcs(lmax, x, spin, lfacs, lfacs2, lrootfacs)
            clp[2:] += (weight * corr[0, i]) * d20
            clm[2:] += (weight * corr[1, i]) * d20
        return 2 * np.pi * unrotate(np.array([clp, clm]), spin)

    # spin (2, 2): undo each slot's own kernel (matching how _cl2corr
    # produced it) to recover rotate(cl) exactly, then unrotate
    r = np.zeros((2, 2, lmax + 1))
    for i, (x, weight) in enumerate(zip(xvals, weights)):
        _, d22, d2m2 = legendre_funcs(lmax, x, spin, lfacs, lfacs2, lrootfacs)
        r[0, 0, 2:] += (weight * corr[0, 0, i]) * d22
        r[0, 1, 2:] += (weight * corr[1, 1, i]) * d2m2
        r[1, 0, 2:] += -(weight * corr[1, 0, i]) * d2m2
        r[1, 1, 2:] += -(weight * corr[0, 1, i]) * d22
    return 2 * np.pi * unrotate(r, spin)


def cl2corr(cls, progress: Progress | None = None):
    """
    Transforms cls to correlation functions
    Args:
        cls: Data Cl
        progress: optional progress reporter
    Returns:
        corr: correlation function
    """
    if progress is None:
        progress = NoProgress()

    wds = {}
    current, total = 0, len(cls)
    for key in cls.keys():
        current += 1
        progress.update(current, total)
        with progress.task(f"{key}"):
            cl = cls[key]
            spin = cl.spin
            # Grab metadata
            dtype = cl.array.dtype
            # Determine lmax from ell field or shape along ell axis
            lmax = len(get_result_array(cl, "ell")[0]) - 1
            xvals, _ = _cached_gauss_legendre(lmax + 1)
            # transform to corrs, dispatching directly on spin
            wd = _cl2corr(cl.array, spin, lmax=lmax)
            # Add metadata back
            wd = np.array(list(wd), dtype=dtype)
            wds[key] = replace(
                cls[key],
                ell=xvals,
                array=wd,
            )
    return wds


def corr2cl(wds, progress: Progress | None = None):
    """
    Transforms correlation functions to cls
    Args:
        wds: data correlation functions
        progress: optional progress reporter
    Returns:
        corr: correlation function
    """
    if progress is None:
        progress = NoProgress()

    cls = {}
    current, total = 0, len(wds)
    for key in wds.keys():
        current += 1
        progress.update(current, total)
        with progress.task(f"{key}"):
            wd = wds[key]
            spin = wd.spin
            # Grab metadata
            dtype = wd.array.dtype
            # Derive lmax from xvals stored in the correlation's ell field
            xvals = get_result_array(wd, "ell")[0]
            lmax = len(xvals) - 1
            # transform to cl, dispatching directly on spin
            cl = _corr2cl(wd.array, spin, lmax=lmax)
            # Add metadata back
            cl = np.array(list(cl), dtype=dtype)
            cls[key] = replace(
                wds[key],
                ell=np.arange(lmax + 1),
                array=cl,
            )
    return cls
