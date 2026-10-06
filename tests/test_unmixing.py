import numpy as np
import pytest

import heracles
from heracles.transforms import _cl2corr, _cached_shifted_gauss_legendre
from heracles.unmixing import apod_window, purify_xip, unmix


def _pure_e_cl(lmax):
    """Pure-E spectrum: EE = 1/(l(l+1)), EB = BE = BB = 0."""
    ls = np.arange(lmax + 1)
    cl = np.zeros((2, 2, lmax + 1))
    cl[0, 0, 2:] = 1.0 / (ls[2:] * (ls[2:] + 1))
    return cl


def _full_sky_mask_cls(lmax_mask):
    """Mask Cl of a unit full-sky mask: only the monopole, so w(theta) = 1."""
    ml = np.zeros(lmax_mask + 1)
    ml[0] = 4 * np.pi
    return heracles.Result(ml, ell=np.arange(lmax_mask + 1), spin=(0, 0), axis=(0,))


def test_apod_window():
    theta = np.linspace(0.0, 180.0, 181)

    # no type or no thetamax: flat window
    assert apod_window(theta, 30.0, type=None) == 1.0
    np.testing.assert_array_equal(apod_window(theta, None), 1.0)
    np.testing.assert_array_equal(apod_window(theta, None, type="gaussian"), 1.0)

    # logistic: one half at thetamax, decreasing
    w = apod_window(theta, 30.0, type="logistic")
    assert w[30] == pytest.approx(0.5)
    assert np.all(np.diff(w) <= 0)

    # gaussian: one at theta = 0, exactly zero beyond thetamax
    w = apod_window(theta, 30.0, type="gaussian")
    assert w[0] == 1.0
    assert np.all(w[theta >= 30.0] == 0.0)

    with pytest.raises(ValueError, match="Unknown apodization type"):
        apod_window(theta, 30.0, type="cosine")


def test_purify_xip_pure_e():
    """
    For a pure-E signal, the purified Xi_+ equals Xi_- below thetamax, so
    the purified correlation functions xi_EE = (Xi_+^pure + Xi_-)/2 and
    xi_BB = (Xi_+^pure - Xi_-)/2 are xi_EE = Xi_- and xi_BB = 0.
    """
    lmax = 64
    cl_ee = _pure_e_cl(lmax)[0, 0]

    theta_max = 30.0
    thetamax_rad = np.radians(theta_max + 3.0)
    xvals, _ = _cached_shifted_gauss_legendre(4 * (lmax + 1), np.cos(thetamax_rad), 1.0)
    xi_p = _cl2corr(cl_ee, (2, 2), lmax=lmax, xvals=xvals)
    xi_m = _cl2corr(cl_ee, (2, -2), lmax=lmax, xvals=xvals)

    xi_p_pure = purify_xip(xvals, xi_p, thetamax_rad)
    xi_ee = 0.5 * (xi_p_pure + xi_m)
    xi_bb = 0.5 * (xi_p_pure - xi_m)

    theta = np.degrees(np.arccos(xvals))
    apod = apod_window(theta, theta_max, type="gaussian")
    xi_ee, xi_bb = apod * xi_ee, apod * xi_bb

    scale = np.abs(xi_ee).max()
    assert scale > 0
    np.testing.assert_allclose(xi_ee, apod * xi_m, rtol=0, atol=1e-2 * scale)
    np.testing.assert_allclose(xi_bb, 0.0, rtol=0, atol=1e-2 * scale)


@pytest.mark.parametrize("apodization", ["gaussian", "logistic"])
@pytest.mark.parametrize("theta_max", [30.0, 60.0])
def test_unmix_purify_pure_e(fields, apodization, theta_max):
    """
    Pure-E Cls on the full sky, apodized to theta_max: the apodization
    alone leaks E into B in the natural estimator, while the purified
    estimator keeps a non-zero EE and a vanishing BB.
    """
    lmax = 64
    cl = _pure_e_cl(lmax)
    key = ("SHE", "SHE", 1, 1)
    d = {key: heracles.Result(cl, ell=np.arange(lmax + 1), spin=(2, 2), axis=(2,))}
    m = {("WHT", "WHT", 1, 1): _full_sky_mask_cls(2 * lmax)}

    kwargs = dict(theta_max=theta_max, apodization=apodization)
    natural = unmix(d, m, fields, purify=False, **kwargs)[key].array
    pure = unmix(d, m, fields, purify=True, **kwargs)[key].array

    cl_ee = cl[0, 0, 2:]
    ee, bb = pure[0, 0, 2:], pure[1, 1, 2:]

    # EE is non-zero away from the low-l and lmax edges
    ratio = ee[28:48] / cl_ee[28:48]
    assert 0.8 < np.mean(ratio) < 1.2
    if apodization == "gaussian":
        # the sharp logistic edge rings in l, so only check the smooth window
        # multipole by multipole
        np.testing.assert_allclose(ratio, 1.0, rtol=0.1)

    # BB vanishes after purification...
    assert np.all(np.abs(bb) < 1e-2 * cl_ee)
    # ...but not without it
    assert np.max(np.abs(natural[1, 1, 2:]) / cl_ee) > 0.1
