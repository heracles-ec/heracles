from scipy.special import lpn as legendrep
import numpy as np

from .result import get_result_array

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


def _prepare_cosebis_filters(ell, wn, wn_ell=None):
    """Return W_n(ell) sampled on the target ell grid."""

    ell = np.asarray(ell, dtype=np.float64)
    wn = np.asarray(wn, dtype=np.float64)

    if wn.ndim == 1:
        wn = wn[np.newaxis, :]
    if wn.ndim != 2:
        raise ValueError("wn must be a 1D or 2D array")

    if wn_ell is None:
        if wn.shape[1] != ell.size:
            raise ValueError(
                "wn must have the same ell length as the input cl when wn_ell is not given"
            )
        return wn

    wn_ell = np.asarray(wn_ell, dtype=np.float64)
    if wn_ell.ndim != 1:
        raise ValueError("wn_ell must be a 1D array")
    if wn_ell.size != wn.shape[1]:
        raise ValueError("wn_ell and wn have incompatible sizes")

    return np.array([np.interp(ell, wn_ell, row, left=0.0, right=0.0) for row in wn])


def cl2cosebis(cls, wn, *, wn_ell=None):
    r"""
    Transform angular power spectra to COSEBIs-like filtered modes.

    This follows the CosmoSIS convention of convolving each spectrum with a set
    of filters ``W_n(ell)`` via:

    .. math::
        X_n = \int d\ell\ \frac{\ell}{2\pi}\ W_n(\ell)\ C_\ell

    Parameters
    ----------
    cls
        Mapping of spectra (typically Heracles ``Result`` objects) with ell on
        the last axis.
    wn
        Filter matrix of shape ``(n_mode, n_ell_filter)`` or a single filter of
        shape ``(n_ell_filter,)``.
    wn_ell
        Optional ell grid for ``wn``. If omitted, ``wn`` is assumed to be
        sampled on the same ell grid as each input spectrum.
    """

    out = {}

    for key, cl in cls.items():
        dtype = cl.array.dtype
        ell = get_result_array(cl, "ell")[-1]
        filters = _prepare_cosebis_filters(ell, wn, wn_ell)
        kernel = (ell[np.newaxis, :] * filters) / (2 * np.pi)

        modes = np.trapz(cl.array[..., np.newaxis, :] * kernel, x=ell, axis=-1)
        mode_index = np.arange(1, kernel.shape[0] + 1)

        out[key] = replace(
            cl,
            array=np.asarray(modes, dtype=dtype),
            ell=mode_index,
            axis=-1,
        )

    return out


def _cosmosis_bindings():
    """Import and return CosmoSIS bindings used by ``cl2cosebis_cosmosis``."""

    from .core import external_dependency_explainer

    with external_dependency_explainer:
        from cosmosis.datablock import DataBlock, names
        from cosmosis.runtime.module import Module

    return DataBlock, names.option_section, Module


def cl2cosebis_cosmosis(
    cls,
    *,
    module_path,
    theta_min,
    theta_max,
    n_modes,
    input_section="shear_cl",
    output_section="cosebis",
    output_section_b=None,
    ell_name="ell",
    cl_name_template="bin_{i}_{j}",
    options=None,
    skip_non_spin2=True,
    _bindings=None,
):
    """
    Run CosmoSIS ``cl_to_cosebis`` on Heracles spectra and return COSEBIs.

    Parameters
    ----------
    cls
        Mapping of Heracles ``Result`` spectra.
    module_path
        Path to the CosmoSIS ``cl_to_cosebis`` module shared library.
    theta_min, theta_max
        Angular limits in arcmin passed to CosmoSIS.
    n_modes
        Number of COSEBI modes.
    input_section, output_section
        CosmoSIS datablock section names for input C_ell and E-mode output.
    output_section_b
        Optional section name for B-mode output.
    ell_name
        Name of the ell array in ``input_section``.
    cl_name_template
        Format string used to map tomographic bins into datablock names.
    options
        Extra CosmoSIS module options.
    skip_non_spin2
        Ignore non-(2,2) spectra instead of raising.
    """

    if _bindings is None:
        DataBlock, option_section, Module = _cosmosis_bindings()
    else:
        DataBlock, option_section, Module = _bindings

    opt = DataBlock()
    opt[option_section, "theta_min"] = float(theta_min)
    opt[option_section, "theta_max"] = float(theta_max)
    opt[option_section, "n_max"] = int(n_modes)
    opt[option_section, "input_section_name"] = input_section
    opt[option_section, "output_section_name"] = output_section
    if output_section_b is not None:
        opt[option_section, "output_section_name_b"] = output_section_b
    if options:
        for name, value in options.items():
            opt[option_section, name] = value

    module = Module("cl_to_cosebis", module_path)
    config = module.setup(opt)

    block = DataBlock()
    out_e = {}
    out_b = {}
    ell_ref = None
    ell_dtype = None
    selected = []

    for key, value in cls.items():
        if value.spin != (2, 2):
            if skip_non_spin2:
                continue
            raise ValueError(f"cl2cosebis_cosmosis requires spin (2, 2), got {value.spin}")

        ell = np.asarray(get_result_array(value, "ell")[-1], dtype=np.float64)
        if ell_ref is None:
            ell_ref = ell
            ell_dtype = value.array.dtype
        elif not np.array_equal(ell_ref, ell):
            raise ValueError("all spin-(2,2) spectra must share the same ell grid")

        i, j = key[-2], key[-1]
        name = cl_name_template.format(i=i, j=j)
        block[input_section, name] = np.asarray(value.array[0, 0], dtype=np.float64)
        selected.append((key, value, name))

    if not selected:
        return ({}, {}) if output_section_b is not None else {}

    block[input_section, ell_name] = ell_ref

    try:
        status = module.execute(block)
    except TypeError:
        # Some wrappers require explicit config forwarding.
        status = module.execute(block, config)
    if status not in (None, 0):
        raise RuntimeError(f"CosmoSIS cl_to_cosebis failed with status {status}")

    mode_index = np.arange(1, int(n_modes) + 1)
    for key, value, name in selected:
        en = np.asarray(block[output_section, name], dtype=np.float64)
        out_e[key] = replace(
            value,
            array=en.astype(ell_dtype, copy=False),
            ell=mode_index,
            axis=-1,
        )
        if output_section_b is not None:
            bn = np.asarray(block[output_section_b, name], dtype=np.float64)
            out_b[key] = replace(
                value,
                array=bn.astype(ell_dtype, copy=False),
                ell=mode_index,
                axis=-1,
            )

    try:
        module.cleanup(config)
    except Exception:
        pass

    if output_section_b is not None:
        return out_e, out_b
    return out_e
