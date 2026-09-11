from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre
from .progress import NoProgress, Progress
from .result import get_result_array


def legendre_p_all_vec(n, xvals, *, diff_n=0):
    """
    All Legendre polynomials of the first kind up to degree n, evaluated
    at every point in the 1D array `xvals` at once, optionally with
    their first derivatives, via the standard three-term recurrence.

    The walk over `l` is inherently sequential and stays a Python loop,
    but each step is a single vectorized numpy expression over all
    points at once -- this is the actual hot path in `_cl2corr`/
    `_corr2cl`, so it matters that it's O(lmax) Python-level iterations,
    each O(npoints) C-level work, rather than O(npoints) Python-level
    calls each doing O(lmax) more Python-level work (the O(lmax^2)
    interpreter overhead an earlier, per-point scalar version had at
    npoints ~ lmax).

    :param n: maximum degree
    :param xvals: 1D array of x values to evaluate at
    :param diff_n: if 1, also return the first derivatives
    :return: `allP` (or `(allP, alldP)`), shape `(n+1, len(xvals))`
    """
    xvals = np.asarray(xvals, dtype=np.float64)
    allP = np.empty((n + 1, len(xvals)))
    allP[0] = 1.0
    if n >= 1:
        allP[1] = xvals
    for ell in range(1, n):
        allP[ell + 1] = ((2 * ell + 1) * xvals * allP[ell] - ell * allP[ell - 1]) / (
            ell + 1
        )
    if diff_n == 0:
        return allP
    assert diff_n == 1, "only diff_n=1 is supported"
    ls = np.arange(1, n + 1)[:, None]
    alldP = np.zeros_like(allP)
    alldP[1:] = ls * (allP[:-1] - xvals * allP[1:]) / (1 - xvals**2)
    return allP, alldP


try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace

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
            # scipy's roots_legendre (Newton iteration off asymptotic
            # initial guesses) rather than np.polynomial.legendre.leggauss
            # (dense eigvalsh of the n x n Jacobi matrix, O(n^3)) -- both
            # give the same nodes/weights to ~1e-14, but leggauss is the
            # dominant cost of a cold call at the lmax this module is used
            # at (e.g. ~13s at lmax=4000 for leggauss alone, vs ~0.6s for
            # roots_legendre) since it's recomputed once per distinct
            # npoints, not amortized like the cache below implies for
            # repeat calls at the same npoints.
            xvals, weights = roots_legendre(npoints)
            xvals.flags.writeable = False
            weights.flags.writeable = False
        if cache:
            _gauss_legendre_cache[key] = xvals, weights
        return xvals, weights


def _cached_shifted_gauss_legendre(npoints, a, b):
    """
    Gauss-Legendre quadrature nodes/weights for `npoints` points on
    `[a, b]` -- an affine shift of the standard `[-1, 1]` rule
    (`_cached_gauss_legendre`), exact for the same polynomial degree as
    the standard rule, just on this sub-interval. Cheap regardless of
    `(a, b)`: reuses the cached standard-rule nodes, no new expensive
    computation and no dedicated cache for the shift itself.
    """
    xvals, weights = _cached_gauss_legendre(npoints)
    scale = (b - a) / 2
    xvals_shifted = a + (xvals + 1) * scale
    weights_shifted = weights * scale
    return xvals_shifted, weights_shifted


def legendre_funcs_vec(lmax, xvals, spin, lfacs=None, lfacs2=None, lrootfacs=None):
    """
    Utility function to return the Legendre/Wigner-d functions needed to
    transform a Cl of the given `spin` to/from a correlation function, for
    all :math:`\ell` up to lmax, evaluated at every point in the 1D array
    `xvals` at once (shape `(nl, len(xvals))`). Note that the spin
    functions start at :math:`\ell=2`, so are shorter than the spin (0, 0)
    case.

    Only spin (0, 0), (0, 2)/(2, 0), and (2, 2) are supported.

    :param lmax: maximum :math:`\ell`
    :param xvals: 1D array of :math:`\cos(\theta)` values to evaluate at
    :param spin: (s1, s2) spin of the field pair -- selects which functions
        are computed
    :param lfacs: optional pre-computed :math:`\ell(\ell+1)` float array
        (ignored for spin (0, 0))
    :param lfacs2: optional pre-computed :math:`(\ell+2)*(\ell-1)` float array
    :param lrootfacs: optional pre-computed sqrt(lfacs*lfacs2) array
    :return: `P` with shape `(lmax+1, len(xvals))` for spin (0, 0);
        otherwise `(d_{20}, d_{22}, d_{2,-2})`, each shape
        `(lmax-1, len(xvals))`
    """
    s1, s2 = spin
    if s1 == 0 and s2 == 0:
        return legendre_p_all_vec(lmax, xvals)
    elif not ({s1, s2} == {0, 2} or (s1 == 2 and s2 == 2)):
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )

    xvals = np.asarray(xvals, dtype=np.float64)
    allP, alldP = legendre_p_all_vec(lmax, xvals, diff_n=1)
    fac1 = 1 - xvals
    fac2 = 1 + xvals

    if lfacs is None:
        ls = np.arange(2, lmax + 1, dtype=np.float64)
        lfacs = ls * (ls + 1)
        lfacs2 = (ls + 2) * (ls - 1)
        lrootfacs = np.sqrt(lfacs * lfacs2)
    lfacs_c = lfacs[:, None]
    lfacs2_c = lfacs2[:, None]
    lrootfacs_c = lrootfacs[:, None]
    P = allP[2:]
    dP = alldP[2:]

    fac = fac1 / fac2
    d22 = (
        ((4 * xvals - 8) / fac2 + lfacs_c) * P
        + 4 * fac * (fac2 + (xvals - 2) / lfacs_c) * dP
    ) / lfacs2_c

    # general-case formula everywhere first, then overwrite the
    # small-angle points' low-l rows with the series below
    d2m2 = (
        (lfacs_c - (4 * xvals + 8) / fac1) * P
        + 4 / fac * (-fac1 + (xvals + 2) / lfacs_c) * dP
    ) / lfacs2_c

    small_angle = xvals > 0.998
    for j in np.nonzero(small_angle)[0]:
        x = xvals[j]
        # for stability use series at small angles (thanks Pavel Motloch)
        indser = int(np.sqrt((400.0 + 3 / (1 - x**2)) / 150)) - 1
        sin2 = 1 - x**2
        d2m2[:indser, j] = (
            lfacs[:indser]
            * lfacs2[:indser]
            * sin2**2
            / 7680
            * (20 + sin2 * (16 - lfacs[:indser]))
        )
    d20 = (2 * xvals * dP - lfacs_c * P) / lrootfacs_c
    return d20, d22, d2m2


def _cl2corr(cl, kernel, lmax=None, sampling_factor=1, xvals=None):
    """
    Get the correlation function of a single, already-rotated Cl COMPONENT
    (1D array `cl[l]`) from the power spectrum, evaluated at points
    cos(theta) = xvals, via the explicit kernel selected by `kernel`:
    (0, 0) -> Legendre P_l (l from 0); (2, 2) -> Wigner d^l_{2,2}; (2, -2)
    -> d^l_{2,-2}; (2, 0) (alias (0, 2)) -> d^l_{2,0} (l from 2 for the
    polarization kernels). `cl` must already be the desired linear
    combination of physical fields (e.g. Cl^EE+Cl^BB for kernel (2, 2)) --
    add/subtract the relevant pair first to build that combination from a
    full [[EE, EB], [BE, BB]] (or [Ta, Tb]) array.
    Use roots of Legendre polynomials (np.polynomial.legendre.leggauss) for accurate back integration with corr2cl.
    Note currently does not work at xvals=1 (can easily calculate that as special case!).

    :param cl: Cl component array, `cl[l]`. Should include
        :math:`\ell(\ell+1)/2\pi` factors.
    :param kernel: (s1, s2) kernel selector; only (0, 0), (2, 0)/(0, 2),
        (2, 2), and (2, -2) are supported
    :param lmax: optional maximum L to use from the cl array
    :param sampling_factor: oversampling factor for the quadrature grid,
        ignored if `xvals` is given
    :param xvals: if given, evaluate the correlation function at these
        cos(theta) points directly instead of the Gauss-Legendre quadrature
        grid -- e.g. to evaluate at arbitrary angles, not just quadrature
        nodes (`_corr2cl` then cannot be used to transform the result back)
    :return: correlation function component array, the l axis replaced by
        the quadrature (theta) axis
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

    if kernel == (0, 0):
        ct = facs * cl[: lmax + 1]
        P = legendre_funcs_vec(lmax, xvals, (0, 0))
        return np.einsum("l,lp->p", ct, P)
    elif kernel not in ((0, 2), (2, 0), (2, 2), (2, -2)):
        raise ValueError(
            f"unsupported kernel {kernel!r}: only (0, 0), (0, 2)/(2, 0), "
            "(2, 2), and (2, -2) are supported"
        )

    # For polarization, all arrays start at 2
    ls2 = ls[2:]
    lfacs = ls2 * (ls2 + 1)
    lfacs2 = (ls2 + 2) * (ls2 - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)

    ct = facs[2:] * cl[2 : lmax + 1]
    d20, d22, d2m2 = legendre_funcs_vec(lmax, xvals, (2, 2), lfacs, lfacs2, lrootfacs)
    d = d2m2 if kernel == (2, -2) else (d22 if kernel == (2, 2) else d20)
    return np.einsum("l,lp->p", ct, d)


def _corr2cl(corr, kernel, lmax=None, sampling_factor=1, xvals=None, weights=None):
    """
    Transform a single correlation-function component (1D array) back to a
    single Cl component, via the explicit kernel selected by `kernel` --
    see `_cl2corr` for the supported kernel values and conventions. This is
    the inverse of `_cl2corr`.
    Note that using _cl2corr followed by _corr2cl is generally very accurate (< 1e-5 relative error) if
    xvals, weights = np.polynomial.legendre.leggauss(lmax+1)

    :param corr: correlation function component array, mirroring
        `_cl2corr`'s output shape
    :param kernel: (s1, s2) kernel selector; only (0, 0), (2, 0)/(0, 2),
        (2, 2), and (2, -2) are supported
    :param lmax: maximum :math:`\ell` to calculate :math:`C_\ell`
    :param sampling_factor: oversampling factor for the quadrature grid,
        ignored if `xvals`/`weights` are given
    :param xvals, weights: optional explicit quadrature nodes/weights to
        use instead of the standard `[-1, 1]` Gauss-Legendre rule (e.g.
        from `_cached_shifted_gauss_legendre`, for a domain-restricted
        reconstruction) -- `corr` must already be evaluated at exactly
        these `xvals`. Either both must be given or neither.
    :return: Cl component array, the theta axis replaced by the l axis.
        Includes :math:`\ell(\ell+1)/2\pi` factors.
    """
    corr = np.asarray(corr, dtype=np.float64)

    if lmax is None:
        lmax = corr.shape[-1] - 1

    if xvals is None:
        assert weights is None, "xvals and weights must be given together"
        xvals, weights = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)
    else:
        assert weights is not None, "xvals and weights must be given together"
        xvals = np.asarray(xvals, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

    if kernel == (0, 0):
        # a node whose correlation value is exactly 0
        #  contributes exactly 0 to the sum
        nz = corr != 0.0
        if not np.any(nz):
            return np.zeros(lmax + 1)
        P = legendre_funcs_vec(lmax, xvals[nz], (0, 0))
        cl = np.einsum("p,lp->l", weights[nz] * corr[nz], P)
        return 2 * np.pi * cl
    elif kernel not in ((0, 2), (2, 0), (2, 2), (2, -2)):
        raise ValueError(
            f"unsupported kernel {kernel!r}: only (0, 0), (0, 2)/(2, 0), "
            "(2, 2), and (2, -2) are supported"
        )

    # For polarization, all arrays start at 2
    ls = np.arange(2, lmax + 1, dtype=np.float64)
    lfacs = ls * (ls + 1)
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)

    cl = np.zeros(lmax + 1)
    # same exact-zero skip as above
    nz = corr != 0.0
    if not np.any(nz):
        return cl
    d20, d22, d2m2 = legendre_funcs_vec(
        lmax, xvals[nz], (2, 2), lfacs, lfacs2, lrootfacs
    )
    d = d2m2 if kernel == (2, -2) else (d22 if kernel == (2, 2) else d20)
    cl[2:] = np.einsum("p,lp->l", weights[nz] * corr[nz], d)
    return 2 * np.pi * cl


def cl2corr(cls, domain=None, progress: Progress | None = None):
    """
    Transforms cls to correlation functions
    Args:
        cls: Data Cl
        domain: optional `(a, b)` bounds in cos(theta) to evaluate every
            key at, via a genuine (affine-shifted) Gauss-Legendre
            quadrature confined to that sub-interval
            (`_cached_shifted_gauss_legendre`) instead of the standard
            `[-1, 1]` rule -- each key gets its own correctly-sized grid
            for its own derived `lmax` (keys need not share `lmax`).
            `None` (default) is today's behavior, unchanged.
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
            (ell,) = get_result_array(cl, "ell")
            lmax = len(ell) - 1
            if domain is None:
                xvals_key, _ = _cached_gauss_legendre(lmax + 1)
            else:
                a, b = domain
                xvals_key, _ = _cached_shifted_gauss_legendre(lmax + 1, a, b)
            # transform to corrs, dispatching directly on spin: add/subtract
            # into the "+/-" basis, then transform each component with its
            # own kernel
            if spin == (0, 0):
                wd = _cl2corr(cl.array, (0, 0), lmax=lmax, xvals=xvals_key)
            elif spin in ((0, 2), (2, 0)):
                # T x spin-2 cross correlation: both combinations use the
                # same d20 kernel
                Ta, Tb = cl.array[0], cl.array[1]
                cp, cm = Ta + Tb, Ta - Tb
                wd = np.array(
                    [
                        _cl2corr(cp, (2, 0), lmax=lmax, xvals=xvals_key),
                        _cl2corr(cm, (2, 0), lmax=lmax, xvals=xvals_key),
                    ]
                )
            else:
                # spin (2, 2): EE/BB use d22/d2m2 on the diagonal pair's
                # sum/difference, EB/BE use d22/d2m2 (negated) on the
                # anti-diagonal pair's sum/difference. Kept as one shared
                # loop (rather than 4 separate _cl2corr component calls)
                # so allP/alldP are computed once per quadrature point
                # instead of 4 times.
                ls = np.arange(0, lmax + 1, dtype=np.float64)
                facs = (2 * ls + 1) / (4 * np.pi)
                ls2 = ls[2:]
                lfacs = ls2 * (ls2 + 1)
                lfacs2 = (ls2 + 2) * (ls2 - 1)
                lrootfacs = np.sqrt(lfacs * lfacs2)
                EE, EB = cl.array[0, 0], cl.array[0, 1]
                BE, BB = cl.array[1, 0], cl.array[1, 1]
                cp = facs[2:] * (EE + BB)[2 : lmax + 1]
                cm = facs[2:] * (EE - BB)[2 : lmax + 1]
                icp = facs[2:] * (EB - BE)[2 : lmax + 1]
                icm = facs[2:] * (EB + BE)[2 : lmax + 1]
                wd = np.zeros((2, 2, len(xvals_key)))
                _, d22, d2m2 = legendre_funcs_vec(
                    lmax, xvals_key, spin, lfacs, lfacs2, lrootfacs
                )
                wd[0, 0] = np.einsum("l,lp->p", cp, d22)  # EE-like
                wd[1, 1] = np.einsum("l,lp->p", cm, d2m2)  # BB-like
                wd[0, 1] = -np.einsum("l,lp->p", icp, d22)  # EB-like
                wd[1, 0] = -np.einsum("l,lp->p", icm, d2m2)  # BE-like
            # Add metadata back
            wd = np.array(list(wd), dtype=dtype)
            wds[key] = replace(
                cls[key],
                ell=xvals_key,
                array=wd,
            )
    return wds


def corr2cl(wds, domain=None, progress: Progress | None = None):
    """
    Transforms correlation functions to cls
    Args:
        wds: data correlation functions
        domain: optional `(a, b)` bounds in cos(theta), matching whatever
            `domain` `cl2corr` was given to produce `wds` (each key's own
            nodes are already stored in its `ell` field; this only
            supplies the corresponding quadrature weights, since a
            shifted rule's weights depend on `(a, b)`, not just point
            count) -- used instead of the standard `[-1, 1]`
            Gauss-Legendre weights. `None` (default) is today's
            behavior, unchanged.
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
            xvals = wd.ell
            if domain is None:
                weights_key = _cached_gauss_legendre(len(xvals))[1]
            else:
                a, b = domain
                weights_key = _cached_shifted_gauss_legendre(len(xvals), a, b)[1]
            lmax = len(xvals) - 1
            # transform to cl, dispatching directly on spin: undo each
            # component's own kernel, then add/subtract back to the
            # physical [[EE, EB], [BE, BB]] (or [Ta, Tb]) layout
            if spin == (0, 0):
                cl = _corr2cl(
                    wd.array, (0, 0), lmax=lmax, xvals=xvals, weights=weights_key
                )
            elif spin in ((0, 2), (2, 0)):
                clp = _corr2cl(
                    wd.array[0], (2, 0), lmax=lmax, xvals=xvals, weights=weights_key
                )
                clm = _corr2cl(
                    wd.array[1], (2, 0), lmax=lmax, xvals=xvals, weights=weights_key
                )
                cl = np.array([(clp + clm) / 2, (clp - clm) / 2])
            else:
                # spin (2, 2): kept as one shared loop, mirroring cl2corr's
                # (2, 2) branch -- see there for the kernel/slot mapping.
                ls = np.arange(2, lmax + 1, dtype=np.float64)
                lfacs = ls * (ls + 1)
                lfacs2 = (ls + 2) * (ls - 1)
                lrootfacs = np.sqrt(lfacs * lfacs2)
                r = np.zeros((2, 2, lmax + 1))
                _, d22, d2m2 = legendre_funcs_vec(
                    lmax, xvals, spin, lfacs, lfacs2, lrootfacs
                )
                r[0, 0, 2:] = np.einsum("p,lp->l", weights_key * wd.array[0, 0], d22)
                r[0, 1, 2:] = np.einsum("p,lp->l", weights_key * wd.array[1, 1], d2m2)
                r[1, 0, 2:] = -np.einsum("p,lp->l", weights_key * wd.array[1, 0], d2m2)
                r[1, 1, 2:] = -np.einsum("p,lp->l", weights_key * wd.array[0, 1], d22)
                p_diag, m_diag = r[0, 0], r[0, 1]
                p_anti, m_anti = r[1, 0], r[1, 1]
                EE, BB = (p_diag + m_diag) / 2, (p_diag - m_diag) / 2
                EB, BE = (p_anti + m_anti) / 2, (p_anti - m_anti) / 2
                cl = 2 * np.pi * np.array([[EE, EB], [BE, BB]])
            # Add metadata back
            cl = np.array(list(cl), dtype=dtype)
            cls[key] = replace(
                wds[key],
                ell=np.arange(lmax + 1),
                array=cl,
            )
    return cls
