import numpy as np
from scipy.special import lpn as legendrep

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
    r"""
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


def _cl2corr(cls, lmax=None, sampling_factor=1):
    r"""
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
    r"""
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


def cl2corr(cls):
    """
    Transforms cls to correlation functions
    Args:
        cls: Data Cl
    Returns:
        corr: correlation function
    """
    wds = {}
    for key in cls.keys():
        cl = cls[key]
        s1, s2 = cl.spin
        # Grab metadata
        dtype = cl.array.dtype
        # Initialize wd
        wd = np.zeros_like(cl)
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
            # transform to corrs
            _wd = _cl2corr(_cl.T).T + 1j * _cl2corr(_icl.T).T
            _rwd = _wd.real
            _iwd = _wd.imag
            # reorder
            wd[0, 0] = _rwd[1]  # EE like spin-2
            wd[1, 1] = _rwd[2]  # BB like spin-2
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
            array=wd,
        )
    return wds


def corr2cl(wds):
    """
    Transforms correlation functions to cls
    Args:
        corrs: data corrs
    Returns:
        corr: correlation function
    """
    cls = {}
    for key in wds.keys():
        wd = wds[key]
        s1, s2 = wd.spin
        # Grab metadata
        dtype = wd.array.dtype
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
            _iwd = np.array(
                [
                    np.zeros_like(wd[0, 0]),
                    wd[0, 1],  # EB like spin-0
                    wd[1, 0],  # EB like spin-0
                    np.zeros_like(wd[0, 0]),
                ]
            )
            # transform back to Cl
            _rcl = _corr2cl(_rwd.T).T
            _icl = _corr2cl(_iwd.T).T
            # reorder
            cl[0, 0] = _rcl[1]  # EE like spin-2
            cl[1, 1] = _rcl[2]  # BB like spin-2
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
            array=cl,
        )
    return cls


def _shear_tomographic_bins(cls):
    bins = set()
    for key in cls:
        s1, s2 = cls[key].spin
        if s1 != 0 and s2 != 0:
            bins.add(key[2])
            bins.add(key[3])
    return sorted(bins)


class _CloeTracerProxy:
    def __init__(self, n_z_bins):
        self.n_z_bins = n_z_bins


class _CloeClsAdapter:
    def __init__(self, cls, n_z_bins):
        self._cls = cls
        self.tracer1 = _CloeTracerProxy(n_z_bins)
        self.tracer2 = _CloeTracerProxy(n_z_bins)

    def get_Cl(self, ells, nl, ks):
        return self._cls

    def _software_tag(self, method):
        return f"{self.__class__.__name__} (heracles), `{method.__name__}` method"


def cl2cosebis(
    cls,
    n_cosebis,
    n_thread=1,
):
    """
    Transform shear-shear angular power spectra to COSEBIs.

    This function can call Cloe's ``get_cosebis`` implementation by
    injecting Heracles ``cls`` through an adapter that overrides
    ``get_Cl``.

    Args:
        cls: Dictionary of Heracles Cl results.
        n_cosebis: Number of COSEBIs modes to compute (1..n_cosebis).
        n_thread: Number of threads passed to Cloe ``get_W_ell``.

    Returns:
        Dictionary of COSEBIs stored as Heracles ``Result`` objects.
        For each tomographic pair, array shape is ``(2, 2, n_cosebis)``,
        with EE on ``[0, 0]`` and BB on ``[1, 1]``.
    """
    n_cosebis = int(n_cosebis)
    if n_cosebis < 1:
        raise ValueError("n_cosebis must be >= 1")
    ns = np.arange(1, n_cosebis + 1, dtype=int)

    _key = next(key for key in cls if cls[key].spin[0] != 0 and cls[key].spin[1] != 0)
    ells = np.asarray(cls[_key].ell, dtype=np.float64)
    theta_min = np.pi / np.max(ells)
    theta_max = np.pi / np.min(ells)
    n_theta = max(64, ells.size)
    thetagrid = np.geomspace(theta_min, theta_max, n_theta)

    from cloelib.auxiliary.cosebi_helpers import get_W_ell

    w_ell = get_W_ell(
        thetagrid,
        n_cosebis,
        ells,
        int(n_thread),
    )

    from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

    bins = _shear_tomographic_bins(cls)
    if not bins:
        raise ValueError("no ('SHE', 'SHE', i, j) entries found in cls")

    nl = None  # Not used by the adapter, but required by the interface.
    ks = None  # Not used by the adapter, but required by the interface.
    adapter = _CloeClsAdapter(cls, len(bins))
    cosebis_ee = AngularTwoPoint.get_cosebis(adapter, ells, nl, ks, w_ell, ns)

    # Fool Cloe into giving us BB COSEBIs by replacing EE with BB in the input cls.
    #  We can then repackage the output to match the original cls keys.
    bb_cls = {}
    for key, result in cls.items():
        s1, s2 = result.spin
        if s1 != 0 and s2 != 0:
            arr = np.zeros_like(result.array)
            arr[0, 0] = result.array[1, 1]
            bb_cls[key] = replace(result, array=arr)
        else:
            bb_cls[key] = result
    adapter_bb = _CloeClsAdapter(bb_cls, len(bins))
    cosebis_bb = AngularTwoPoint.get_cosebis(adapter_bb, ells, nl, ks, w_ell, ns)

    # Repackage the outputs to match the original cls keys.
    cosebis = {}
    for key, result in cls.items():
        s1, s2 = result.spin
        if s1 != 0 and s2 != 0:
            if key not in cosebis_ee or key not in cosebis_bb:
                raise KeyError(f"Missing COSEBIs output for key {key}")
            arr = np.zeros((2, 2, ns.size), dtype=np.float64)
            arr[0, 0] = np.asarray(cosebis_ee[key].array, dtype=np.float64)
            arr[1, 1] = np.asarray(cosebis_bb[key].array, dtype=np.float64)
            cosebis[key] = replace(result, array=arr, ell=ns, lower=ns, upper=ns + 1)
        else:
            cosebis[key] = result

    return cosebis
