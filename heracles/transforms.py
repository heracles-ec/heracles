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
    if cache and npoints in _gauss_legendre_cache:
        return _gauss_legendre_cache[npoints]
    else:
        if gauss_legendre is not None:
            xvals = np.empty(npoints)
            weights = np.empty(npoints)
            gauss_legendre(xvals, weights, npoints)
            xvals.flags.writeable = False
            weights.flags.writeable = False
        else:
            xvals, weights = np.polynomial.legendre.leggauss(npoints)
        if cache:
            _gauss_legendre_cache[npoints] = xvals, weights
        return xvals, weights


def legendre_funcs(lmax, x, m=(0, 2), lfacs=None, lfacs2=None, lrootfacs=None):
    """
    Utility function to return array of Legendre and :math:`d_{mn}` functions for all :math:`\ell` up to lmax.
    Note that :math:`d_{mn}` arrays start at :math:`\ell_{\rm min} = \max(m,n)`, so returned arrays are different sizes

    :param lmax: maximum :math:`\ell`
    :param x: scalar value of :math:`\cos(\theta)` at which to evaluate
    :param m: m values to calculate :math:`d_{m,n}`, etc. as relevant
    :param lfacs: optional pre-computed :math:`\ell(\ell+1)` float array
    :param lfacs2: optional pre-computed :math:`(\ell+2)*(\ell-1)` float array
    :param lrootfacs: optional pre-computed sqrt(lfacs*lfacs2) array
    :return: :math:`(P,P'),(d_{11},d_{-1,1}), (d_{20}, d_{22}, d_{2,-2})` as requested, where P starts
             at :math:`\ell=0`, but spin functions start at :math:`\ell=\ell_{\rm min}`
    """
    allP, alldP = legendre_p_all(lmax, x, diff_n=1)
    # Polarization functions all start at L=2
    fac1 = 1 - x
    fac2 = 1 + x
    res = []
    if 0 in m:
        res.append((allP, alldP))

    if 1 in m:
        lfacs1 = np.arange(1, lmax + 1, dtype=np.float64)
        lfacs1 *= 1 + lfacs1
        d11 = fac1 * alldP[1:] / lfacs1 + allP[1:]
        dm11 = fac2 * alldP[1:] / lfacs1 - allP[1:]
        res.append((d11, dm11))

    if 2 in m:
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
        res.append((d20, d22, d2m2))

    return res


def purify(f, theta, theta_max=None):
    r"""
    Delta-function ("T") correction operator (Chon et al. 2004) applied to a
    correlation function `f` tabulated at cos(theta) = x.

    For every input angle theta_i (x_i = cos(theta_i)), computes

        T[f](theta_i) = f(theta_i)
            + prefac1(x_i) * int_{x_i}^{x_max} (1+x') f(x') / (1-x')^2 dx'
            + prefac2(x_i) * int_{x_i}^{x_max} f(x') / (1-x')^2 dx'

    where x_max = cos(radians(theta_max)) (or 1 if theta_max is None).

    `f` is only known at the tabulated points `theta`. Every
    int_{x_i}^{x_max} is read off a *single* reverse cumulative trapezoidal
    integration over the tabulated nodes -- since trapezoidal panel
    integrals between consecutive nodes are additive over sub-intervals, a
    reverse cumulative sum of them directly gives the tail integral from
    each node up to x_max, in one O(N) pass, instead of re-interpolating
    `f` and running a fresh high-order quadrature per evaluation point
    (O(N * n_quad)).

    Args:
        f: array of function values, tabulated at the angles in `theta`.
        theta: array of angles (degrees) at which `f` is tabulated, and at
            which T[f] is evaluated.
        theta_max: small angle (in degrees) excluding a neighborhood of the
            theta'=0 singularity from the integration. The integral runs
            over theta' in [theta_max, theta_i] instead of [0, theta_i];
            for theta_i <= theta_max, the whole would-be domain is inside
            the excluded core, so T[f](theta_i) = f(theta_i) (no
            correction). If None, integrate all the way to theta'=0
            (x'=1), which is singular unless f vanishes there.

            `prefac1`/`prefac2` (below) have their own, unrelated pole at
            theta_i=180 (x_i=-1); the same `theta_max` is reused to exclude
            a mirrored neighborhood [180-theta_max, 180] of *evaluation*
            points theta_i, since the prefactors blow up there regardless
            of theta'=0 or the integration domain. theta_i in that band
            also get T[f](theta_i) = f(theta_i) (no correction).

    Returns:
        Array of T[f](theta), same shape as `theta`.

    Note:
        The kernel f(x')/(1-x')^2 is singular as x' -> 1 (theta' -> 0).
        Passing `theta_max` keeps the quadrature away from that endpoint
        (recommended for f that doesn't vanish at theta'=0); leaving it
        None integrates all the way to the singularity and, unless f
        vanishes there to at least second order, the integral is formally
        a Hadamard finite-part integral that this quadrature does not
        converge to. Separately, prefac1 = 8(2-x)/(1+x)^2 and
        prefac2 = 8/(1+x) both diverge as x -> -1 (theta -> 180); with
        theta_max=None this endpoint is *not* guarded and T[f] will blow up
        for theta_i near 180 whenever the (theta_max-truncated or not)
        integral doesn't vanish fast enough to cancel the pole.

        The trapezoidal rule is only as accurate as the tabulated grid is
        dense/well-resolved (O(h^2) in the local node spacing) -- it is not
        adaptive like the point-by-point Gauss-Legendre quadrature this
        replaced. The integral's upper limit is also only approximated by
        the nearest tabulated node <= x_max, not x_max itself exactly.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    if f.shape != theta.shape:
        raise ValueError("f and theta must have the same shape")

    n = len(theta)
    x = np.cos(np.radians(theta))
    x_max = 1.0 if theta_max is None else np.cos(np.radians(theta_max))

    order = np.argsort(x)
    xs = x[order]
    gs = f[order] / (1 - xs) ** 2
    g1 = (1 + xs) * gs
    g2 = gs

    # cumulative trapezoidal integration over the tabulated grid: panel
    # integrals between consecutive sorted nodes are additive over
    # sub-intervals, so a reverse cumulative sum of them gives the tail
    # integral from each node up to (approximately) x_max. This is *not*
    # the same as summing partial Gauss-Legendre weights, which are only
    # exact for the full-interval sum, not sub-interval slices of it.
    dx = np.diff(xs)
    panel1 = dx * (g1[:-1] + g1[1:]) / 2
    panel2 = dx * (g2[:-1] + g2[1:]) / 2

    # panels beyond x_max don't contribute (the integral's upper limit is
    # approximated by the nearest tabulated node <= x_max)
    beyond = xs[1:] > x_max
    panel1[beyond] = 0.0
    panel2[beyond] = 0.0

    int1 = np.empty(n)
    int2 = np.empty(n)
    int1[order] = np.concatenate([np.cumsum(panel1[::-1])[::-1], [0.0]])
    int2[order] = np.concatenate([np.cumsum(panel2[::-1])[::-1], [0.0]])

    prefac1 = 8 * (2 - x) / (1 + x) ** 2
    prefac2 = 8 / (1 + x)

    result = f + prefac1 * int1 + prefac2 * int2

    # x_i >= x_max, or theta_i within theta_max of 180, get no correction
    skip = x >= x_max
    if theta_max is not None:
        skip |= theta >= 180.0 - theta_max
    return np.where(skip, f, result)


def _cl2corr(cls, lmax=None, sampling_factor=1):
    """
    Get the correlation function from the power spectra, evaluated at points cos(theta) = xvals.
    Use roots of Legendre polynomials (np.polynomial.legendre.leggauss) for accurate back integration with corr2cl.
    Note currently does not work at xvals=1 (can easily calculate that as special case!).

    :param cls: 2D array cls(L,ix), with L (:math:`\equiv \ell`) starting at zero and ix-0,1,2,3 in
                order TT, EE, BB, TE. cls should include :math:`\ell(\ell+1)/2\pi` factors.
    :param xvals: array of :math:`\cos(\theta)` values at which to calculate correlation function.
    :param lmax: optional maximum L to use from the cls arrays
    :return: 2D array of corrs[i, ix], where ix=0,1,2,3 are T, Q+U, Q-U and cross
    """

    if cls.ndim == 1:
        cls = np.array(
            [cls, np.zeros_like(cls), np.zeros_like(cls), np.zeros_like(cls)]
        ).T

    if lmax is None:
        lmax = cls.shape[0] - 1

    xvals, weights = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)

    ls = np.arange(0, lmax + 1, dtype=np.float64)
    corrs = np.zeros((len(xvals), 4))
    lfacs = ls * (ls + 1)
    lfacs[0] = 1
    facs = (2 * ls + 1) / (4 * np.pi)

    ct = facs * cls[: lmax + 1, 0]
    # For polarization, all arrays start at 2
    cp = facs[2:] * (cls[2 : lmax + 1, 1] + cls[2 : lmax + 1, 2])
    cm = facs[2:] * (cls[2 : lmax + 1, 1] - cls[2 : lmax + 1, 2])
    cc = facs[2:] * cls[2 : lmax + 1, 3]
    ls = ls[2:]
    lfacs = lfacs[2:]
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)
    for i, x in enumerate(xvals):
        (P, _), (d20, d22, d2m2) = legendre_funcs(
            lmax, x, [0, 2], lfacs, lfacs2, lrootfacs
        )
        corrs[i, 0] = np.dot(ct, P)  # T
        corrs[i, 1] = np.dot(cp, d22)  # Q+U
        corrs[i, 2] = np.dot(cm, d2m2)  # Q-U
        corrs[i, 3] = np.dot(cc, d20)  # cross
    return corrs


def _corr2cl(corrs, lmax=None, sampling_factor=1):
    """
    Transform from correlation functions to power spectra.
    Note that using cl2corr followed by corr2cl is generally very accurate (< 1e-5 relative error) if
    xvals, weights = np.polynomial.legendre.leggauss(lmax+1)

    :param corrs: 2D array, corrs[i, ix], where ix=0,1,2,3 are T, Q+U, Q-U and cross
    :param xvals: values of :math:`\cos(\theta)` at which corrs stores values
    :param weights: weights for integrating each point in xvals. Typically from np.polynomial.legendre.leggauss
    :param lmax: maximum :math:`\ell` to calculate :math:`C_\ell`
    :return: array of power spectra, cl[L, ix], where L starts at zero and ix=0,1,2,3 in order TT, EE, BB, TE.
      They include :math:`\ell(\ell+1)/2\pi` factors.
    """

    if corrs.ndim == 1:
        corrs = np.array(
            [corrs, np.zeros_like(corrs), np.zeros_like(corrs), np.zeros_like(corrs)]
        ).T

    if lmax is None:
        lmax = corrs.shape[0] - 1

    xvals, weights = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)

    # For polarization, all arrays start at 2
    ls = np.arange(2, lmax + 1, dtype=np.float64)
    lfacs = ls * (ls + 1)
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)
    cls = np.zeros((lmax + 1, 4))
    for i, (x, weight) in enumerate(zip(xvals, weights)):
        (P, _), (d20, d22, d2m2) = legendre_funcs(
            lmax, x, [0, 2], lfacs, lfacs2, lrootfacs
        )
        cls[:, 0] += (weight * corrs[i, 0]) * P
        T2 = (corrs[i, 1] * weight / 2) * d22
        T4 = (corrs[i, 2] * weight / 2) * d2m2
        cls[2:, 1] += T2 + T4
        cls[2:, 2] += T2 - T4
        cls[2:, 3] += (weight * corrs[i, 3]) * d20
    return 2 * np.pi * cls


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
            s1, s2 = cl.spin
            # Grab metadata
            dtype = cl.array.dtype
            # Determine lmax from ell field or shape along ell axis
            lmax = len(get_result_array(cl, "ell")[0]) - 1
            xvals, _ = _cached_gauss_legendre(lmax + 1)
            # Initialize wd
            wd = np.zeros_like(cl)
            if (s1 != 0) and (s2 != 0):
                _cl = np.array(
                    [
                        np.zeros_like(cl[0, 0]),
                        cl[0, 0],  # EE like spin-2
                        cl[1, 1],   # BB like spin-2
                        np.zeros_like(cl[0, 0]),
                    ]
                )
                _icl = np.array(
                    [
                        np.zeros_like(cl[0, 0]),
                        -cl[0, 1],  # EB like spin-0
                        cl[1, 0],  # EB like spin-0
                        np.zeros_like(cl[0, 0]),
                    ]
                )
                # transform to corrs
                _wd = _cl2corr(_cl.T).T + 1j * _cl2corr(_icl.T).T
                _rwd = _wd.real
                _iwd = _wd.imag
                # reorder (purify reads the opposite Wigner-matrix slot)
                wd[0, 0] = _rwd[1]  # E^+ (or E^+_dec)
                wd[1, 1] = _rwd[2]  # E^- (or E^-_dec)
                wd[0, 1] = _iwd[1]  # EB like spin-0
                wd[1, 0] = _iwd[2]  # EB like spin-0
            elif (s1 != 0) or (s2 != 0):
                _clp = np.array(
                    [
                        np.zeros_like(cl[0]),
                        np.zeros_like(cl[0]),
                        np.zeros_like(cl[0]),
                        cl[0] + cl[1],  # TE like spin-2
                    ]
                )
                _clm = np.array(
                    [
                        np.zeros_like(cl[0]),
                        np.zeros_like(cl[0]),
                        np.zeros_like(cl[0]),
                        cl[0] - cl[1],  # TE like spin-2
                    ]
                )
                # trnsform to corrs
                wd[0] = _cl2corr(_clp.T).T[3]
                wd[1] = _cl2corr(_clm.T).T[3]
            elif (s1 == 0) and (s2 == 0):
                wd = _cl2corr(cl).T[0]
            else:
                raise ValueError("Invalid spin combination")
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
            s1, s2 = wd.spin
            # Grab metadata
            dtype = wd.array.dtype
            # Derive lmax from xvals stored in the correlation's ell field
            xvals = get_result_array(wd, "ell")[0]
            lmax = len(xvals) - 1
            # initialize cl
            cl = np.zeros_like(wd)
            if (s1 != 0) and (s2 != 0):
                _rwd = np.array(
                    [
                        np.zeros_like(wd[0, 0]),
                        wd[0, 0],  # EE like spin-2
                        wd[1, 1],  # BB like spin-2
                        np.zeros_like(wd[0, 0]),
                    ]
                )
                _rcl = _corr2cl(_rwd.T).T
                cl[0, 0] = _rcl[1]  # EE like spin-2
                cl[1, 1] = _rcl[2]  # BB like spin-2
            # EB cross-term: unchanged by purify
                _iwd = np.array(
                    [
                        np.zeros_like(wd[0, 0]),
                        wd[0, 1],  # EB like spin-0
                        wd[1, 0],  # EB like spin-0
                        np.zeros_like(wd[0, 0]),
                    ]
                )
                _icl = _corr2cl(_iwd.T).T
                cl[0, 1] = -_icl[1]  # EB like spin-0
                cl[1, 0] = _icl[2]  # EB like spin-0
            elif (s1 != 0) or (s2 != 0):
                _wp = np.array(
                    [
                        np.zeros_like(wd[0]),
                        np.zeros_like(wd[0]),
                        np.zeros_like(wd[0]),
                        wd[0],  # TE like spin-2
                    ]
                )
                _wm = np.array(
                    [
                        np.zeros_like(wd[0]),
                        np.zeros_like(wd[0]),
                        np.zeros_like(wd[0]),
                        wd[1],  # TE like spin-2
                    ]
                )
                _clp = _corr2cl(_wp.T).T[3]
                _clm = _corr2cl(_wm.T).T[3]
                cl[0] = (_clp + _clm) / 2
                cl[1] = (_clp - _clm) / 2
            elif (s1 == 0) and (s2 == 0):
                # Treat everything as spin-0 and preserve 1D shape.
                cl = _corr2cl(wd).T[0]
            else:
                raise ValueError("Invalid spin combination")
            # Add metadata back
            cl = np.array(list(cl), dtype=dtype)
            cls[key] = replace(
                wds[key],
                ell=np.arange(lmax + 1),
                array=cl,
            )
    return cls


def _purified_corr2cl(corr_wd, theta_max=None, progress: Progress | None = None):
    """
    Purified version of corr2cl for the natural-spice pipeline.

    For s1=s2=2 (EE/BB) keys, uses the delta-function correction `T` to turn
    the unmixed Xi^+ = corr_wd[key][0, 0] into the "dec" correlation
    Xi^+_dec = T[Xi^+], then builds the pure EE/BB correlation functions

        Xi^EE = Xi^+_dec + Xi^-
        Xi^BB = Xi^+ - Xi^+_dec

    which are transformed to Cl with the *opposite* Wigner matrix from the
    one they would normally use (Xi^EE via d^l_{2,-2}, Xi^BB via d^l_{2,2}),
    i.e.

        Cl^EE = int 1/2 (Xi^+_dec + Xi^-) d^l_{2,-2}
        Cl^BB = int 1/2 (Xi^+ - Xi^+_dec) d^l_{2,2}

    All other keys (TE, TT) are unaffected by purification and are passed
    through the ordinary corr2cl.
    Args:
        corr_wd: mask-deconvolved data correlation functions (e.g. the
            output of heracles.unmixing._naturalspice)
        theta_max: passed through to `T` as its own `theta_max`, excluding a
            neighborhood of the theta'=0 (and, symmetrically, theta=180)
            singularities from the T integral (see `T`). If None, T
            integrates all the way to theta'=0, which is singular unless
            Xi^+ vanishes there.
        progress: optional progress reporter
    Returns:
        corr_d: purified Cl
    """
    if progress is None:
        progress = NoProgress()

    spin2_keys = [key for key, wd in corr_wd.items() if wd.spin[0] != 0 and wd.spin[1] != 0]
    other = {key: wd for key, wd in corr_wd.items() if key not in spin2_keys}

    cls = corr2cl(other) if other else {}

    current, total = 0, len(spin2_keys)
    for key in spin2_keys:
        current += 1
        progress.update(current, total)

        wd = corr_wd[key]
        dtype = wd.array.dtype
        xvals = get_result_array(wd, "ell")[0]
        theta = np.degrees(np.arccos(xvals))
        lmax = len(xvals) - 1

        Xi_p, Xi_m = wd[0, 0], wd[1, 1]
        Xi_p_dec = purify(Xi_p, theta, theta_max=theta_max)

        zeros = np.zeros_like(Xi_p)
        # Xi^BB, transformed with d^l_{2,2} alone (the "+"-matrix)
        _rwd_BB = np.array([zeros, Xi_p - Xi_p_dec, zeros, zeros])
        # Xi^EE, transformed with d^l_{2,-2} alone (the "-"-matrix)
        _rwd_EE = np.array([zeros, zeros, Xi_p_dec + Xi_m, zeros])
        cl_BB = _corr2cl(_rwd_BB.T).T[1]
        cl_EE = _corr2cl(_rwd_EE.T).T[1]

        # EB cross-term: unaffected by purification
        _iwd = np.array([zeros, wd[0, 1], wd[1, 0], zeros])
        _icl = _corr2cl(_iwd.T).T

        cl = np.zeros_like(wd)
        cl[0, 0] = cl_EE
        cl[1, 1] = cl_BB
        cl[0, 1] = -_icl[1]
        cl[1, 0] = _icl[2]
        cl = np.array(list(cl), dtype=dtype)

        cls[key] = replace(wd, ell=np.arange(lmax + 1), array=cl)

    return cls
