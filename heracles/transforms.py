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

    The kernel 1/(1-x')^2 is steep just below x_max (whenever theta_max is
    small), where the tabulated grid -- spaced for the whole 0-180 degree
    range -- is too coarse for a plain trapezoidal panel to resolve well.
    The single panel that would otherwise straddle x_max (and, previously,
    was dropped from the sum entirely) is instead replaced by the closed-form
    integral of 1/(1-x')^2 and (1+x')/(1-x')^2 against a linear fit of f
    between the two tabulated nodes bracketing x_max, evaluated exactly up
    to x_max -- analogous in spirit to the small-angle series `legendre_funcs`
    uses for d2m2 near x=1, though here f is arbitrary tabulated data rather
    than a known analytic function, so only this one boundary panel can be
    handled in closed form; panels further from x_max still use the plain
    trapezoidal rule.

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
    fs = f[order]
    gs = fs / (1 - xs) ** 2
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

    if theta_max is not None and n >= 2:
        # Analytic correction for the steep sliver [xa, x_max] just below
        # x_max, which the plain trapezoidal panels above dropped entirely
        # (xa is the tabulated node closest to, but not above, x_max; xb
        # the next one, used only to get a local slope for f -- extrapolate
        # off (xa, xb) if x_max happens to fall beyond the tabulated grid).
        idx_a = int(np.clip(np.searchsorted(xs, x_max, side="right") - 1, 0, n - 2))
        xa, fa = xs[idx_a], fs[idx_a]
        xb, fb = xs[idx_a + 1], fs[idx_a + 1]
        ua, ub = 1 - xa, 1 - x_max
        if xb != xa and ub > 0:
            s = (fb - fa) / (xb - xa)
            A = fa + s * (1 - xa)
            # F2(u) is the antiderivative of f(x')/(1-x')^2 in u=1-x', with
            # f linearised as A - s*u; F1(u) likewise for (1+x')f/(1-x')^2.
            F2 = lambda u: A / u + s * np.log(u)  # noqa: E731
            F1 = lambda u: 2 * A / u + (A + 2 * s) * np.log(u) - s * u  # noqa: E731
            I2_bnd = F2(ub) - F2(ua)
            I1_bnd = F1(ub) - F1(ua)
            include = x <= xa
            int1 = np.where(include, int1 + I1_bnd, int1)
            int2 = np.where(include, int2 + I2_bnd, int2)

    prefac1 = 8 * (2 - x) / (1 + x) ** 2
    prefac2 = 8 / (1 + x)

    result = f + prefac1 * int1 + prefac2 * int2

    # x_i >= x_max, or theta_i within theta_max of 180, get no correction
    skip = x >= x_max
    if theta_max is not None:
        skip |= theta >= 180.0 - theta_max
    return np.where(skip, f, result)


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


def _cl2corr(cl, spin, lmax=None, sampling_factor=1):
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
    :param sampling_factor: oversampling factor for the quadrature grid
    :return: correlation function array with the same leading shape as `cl`,
        but with the l axis replaced by the quadrature (theta) axis
    """
    cl = np.asarray(cl, dtype=np.float64)

    if lmax is None:
        lmax = cl.shape[-1] - 1

    xvals, _ = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)
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
