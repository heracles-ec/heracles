from scipy.special import lpn as legendrep
from heracles.twopoint import truncated
import numpy as np


try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


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
    allP, alldP = legendrep(lmax, x)
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


def cl2corr(cls, lmax=None, sampling_factor=1):
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
    corrs[:, 0] += corrs[:, 0][0] / (4 * np.pi)
    return corrs


def corr2cl(corrs, lmax=None, sampling_factor=1):
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

    cls[1, :] *= 2
    cls[2:, :] = cls[2:, :]
    return 2 * np.pi * cls


def transform_cls(cls, lmax_out=None):
    """
    Natural unmixing of the data Cl.
    Args:
        cls: Data Cl
    Returns:
        corr: correlation function
    """
    wds = {}
    for key in cls.keys():
        cl = cls[key]
        s1, s2 = cl.spin
        lmax = cl.shape[-1]
        if lmax_out is None:
            lmax_out = lmax
        # Grab metadata
        dtype = cl.array.dtype
        # pad cls
        cl = np.atleast_2d(cl.array)
        pad_width = [(0, 0)] * cl.ndim  # no padding for other dims
        pad_width[-1] = (0, lmax_out - lmax)  # pad only last dim
        cl = np.pad(cl, pad_width, mode="constant", constant_values=0)
        if (s1 != 0) and (s2 != 0):
            _cl = np.array(
                [
                    np.zeros_like(cl[0, 0]),
                    cl[0, 0],  # EE like spin-2
                    cl[1, 1],  # BB like spin-2
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
            # transform to corr
            _wd = cl2corr(_cl.T).T + 1j * cl2corr(_icl.T).T
            _iwd = _wd.imag
            _wd = _wd.real
            # reorganize
            wd = np.zeros_like(cl)
            wd[0, 0] = _wd[1]  # EE like spin-2
            wd[1, 1] = _wd[2]  # BB like spin-2
            wd[0, 1] = _iwd[1]  # EB like spin-0
            wd[1, 0] = _iwd[2]  # EB like spin-0
        else:
            # Treat everything as spin-0
            wd = []
            for _cl in cl:
                _wd = cl2corr(_cl).T
                wd.append(_wd[0])
            # remove extra axis
            wd = np.squeeze(wd)
        # Add metadata back
        wd = np.array(list(wd), dtype=dtype)
        wds[key] = replace(cls[key],
                           ell=np.arange(lmax_out),
                           lower=np.arange(lmax_out)[:-1],
                           upper=np.arange(lmax_out)[1:],
                           weight=np.ones(lmax_out),
                           array=wd)
    # truncate to lmax
    wds = truncated(wds, lmax_out)
    return wds


def transform_corrs(wds, lmax_out=None):
    """
    Natural unmixing of the data Cl.
    Args:
        corrs: data corrs
    Returns:
        corr: correlation function
    """
    cls = {}
    for key in wds.keys():
        wd = wds[key]
        s1, s2 = wd.spin
        lmax = wd.shape[-1]
        if lmax_out is None:
            lmax_out = lmax
        # Grab metadata
        dtype = wd.array.dtype
        # pad cls
        wd = np.atleast_2d(wd.array)
        pad_width = [(0, 0)] * wd.ndim  # no padding for other dims
        pad_width[-1] = (0, lmax_out - lmax)  # pad only last dim
        wd = np.pad(wd, pad_width, mode="constant", constant_values=0)
        if (s1 != 0) and (s2 != 0):
            _wd = np.array(
                [
                    np.zeros_like(wd[0, 0]),
                    wd[0, 0],  # EE like spin-2
                    wd[1, 1],  # BB like spin-2
                    np.zeros_like(wd[0, 0]),
                ]
            )
            _iwd = np.array(
                [
                    np.zeros_like(wd[0, 0]),
                    wd[0, 1],  # EB like spin-0
                    wd[1, 0],  # EB like spin-0
                    np.zeros_like(wd[0, 0]),
                ]
            )
            # transform to cls
            _wd = corr2cl(_wd.T).T
            _iwd = corr2cl(_iwd.T).T
            # reorganize
            cl = np.zeros_like(wd)
            cl[0, 0] = _wd[1]  # EE like spin-2
            cl[1, 1] = _wd[2]  # BB like spin-2
            cl[0, 1] = _iwd[1]  # EB like spin-0
            cl[1, 0] = _iwd[2]  # EB like spin-0
        else:
            # Treat everything as spin-0
            cl = []
            for _wd in wd:
                _wd = corr2cl(_wd).T
                cl.append(_wd[0])
            # remove extra axis
            cl = np.squeeze(cl)
        # Add metadata back
        cl = np.array(list(cl), dtype=dtype)
        cls[key] = replace(wds[key],
                           ell=np.arange(lmax_out),
                           lower=np.arange(lmax_out)[:-1],
                           upper=np.arange(lmax_out)[1:],
                           weight=np.ones(lmax_out), 
                           array=cl)
    # truncate to lmax
    cls = truncated(cls, lmax_out)
    return cls
