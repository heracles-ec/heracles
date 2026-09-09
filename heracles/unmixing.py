# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of Heracles.
#
# Heracles is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Heracles is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Heracles. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from .progress import NoProgress, Progress
from .result import binned, get_result_array
from .transforms import cl2corr, corr2cl, _cl2corr, _corr2cl
from .utils import get_cl

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def logistic(x, x0=-2, k=20):
    return 1.0 + np.exp(-k * (x - x0))


def gaussian(theta, fwhm, thetamax=None):
    """
    Gaussian apodization window in theta (degrees), matching PolSpice's
    `apodizefunction` type 0 (apodize_mod.f90): `fwhm` (PolSpice's
    `-apodizesigma`, despite the name) sets the taper's FWHM, and
    `thetamax` (PolSpice's separate `-thetamax`) sets its hard cutoff --
    these are two independent PolSpice options, not the same value.
    If `thetamax` is None, it defaults to `fwhm` (matching the previous,
    single-parameter behaviour). If `fwhm` is None, no apodization is
    applied (flat weight of 1 everywhere).
    """
    if fwhm is None:
        return np.ones_like(theta)
    if thetamax is None:
        thetamax = fwhm
    sigma = fwhm / np.sqrt(8 * np.log(2))
    return np.where(theta < thetamax, np.exp(-0.5 * (theta / sigma) ** 2), 0.0)


def purify_xip(cl_ee, cl_bb, cl_mask, lmax, xvals, thetamax):
    """
    Port of PolSpice's `cumul` (cumul2.f90): the cumulative-integral
    correction that turns the natural (mask-ratio) Xi_+/Xi_- correlation
    into the "pure" E/B correlation function used by the decouple
    estimator (Chon et al. 2004, eq. 60-65). This is the piece missing
    from a plain Fl-normalized Legendre transform: it accounts for E/B
    leakage from the finite integration range (`thetamax`) via a
    cumulative integral of C+(beta) = Xi_+^raw(beta)/Xi_mask(beta) against
    two trigonometric kernels.

    Args:
        cl_ee, cl_bb: raw (masked, not yet unmixed) Cl_EE, Cl_BB of the data
        cl_mask: raw Cl of the (scalar) mask
        lmax: maximum l
        xvals: cos(theta) values (Gauss-Legendre nodes) at which to
            evaluate the correction -- C+(beta) at these nodes is computed
            internally, the same way as at the coarse grid below (see
            module/notebook discussion: C+ is a ratio, not itself
            band-limited, so there is no way to get it here other than by
            evaluating cl_ee/cl_bb/cl_mask's own transform, same as the
            caller would have to do to hand it in as a separate argument)
        thetamax: integration domain in radians
    Returns:
        c_beta: the cumulative-integral correction, evaluated at `xvals`
    """
    cl_sum = cl_ee[: lmax + 1] + cl_bb[: lmax + 1]
    cl_mask = cl_mask[: lmax + 1]

    theta_nodes = np.arccos(xvals)

    # cumulative integral from 0 to each node, via a fixed grid + cumulative
    # trapezoid rule -- matching PolSpice's own `cumul_simpson` (cumul2.f90),
    # which likewise samples C+(beta) on an independent fine mesh of
    # [0, thetamax] rather than reusing the Gauss-Legendre node grid, since
    # the cumulative integral needs a genuinely finer sampling than the
    # handful of quadrature nodes gives; nodes beyond thetamax are capped
    # there, same as PolSpice's own cumul().
    #
    # c_beta = cp + sum1/sin^2(theta/2) - 2*sum2*(2+cos(theta))/sin^4(theta/2)
    # is a near-total cancellation between O(1) terms as theta -> 0 (sum1,
    # sum2 -> 0 just fast enough to keep c_beta finite), so a uniform grid's
    # *relative* error in sum1/sum2 (dominated by the well-resolved bulk of
    # [0, thetamax]) gets massively amplified by the 1/sin^2, 1/sin^4
    # factors for the smallest theta nodes -- this showed up as xi_B_final
    # (and hence the purified Cl_BB) spuriously blowing up at high l instead
    # of decaying like PolSpice's. Concentrating grid points near beta=0
    # (cubic spacing) fixes this far more cheaply than simply raising
    # ngrid uniformly (verified: matches a 100x larger uniform grid's
    # result at ~1/50th the points).
    ngrid = 20000
    eps = 1e-6
    beta_max = max(thetamax - eps, eps)
    u = np.linspace(0.0, 1.0, ngrid)
    beta_grid = eps + (beta_max - eps) * u**3

    # C+(beta) = Xi_p(beta)/Xi_mask(beta) is itself a finite sum of
    # Wigner-d/Legendre functions up to degree lmax, i.e. band-limited by
    # lmax -- evaluating it at all `ngrid` points (chosen only for the
    # *cumulative integral* below, which genuinely needs that many samples
    # to resolve the sin/cos kernels near beta=0) massively oversamples
    # what the transform itself carries information for. Evaluate it on a
    # much coarser sub-grid instead and cubic-spline interpolate onto the
    # full fine grid: verified against direct evaluation at all `ngrid`
    # points to ~1e-7 median relative error (occasional larger relative
    # outliers are all at C+'s own benign zero-crossings, where relative
    # error is meaningless -- absolute error there is still ~1e-6 on
    # values of order unity) -- >10x faster for this function's expensive
    # part, with no measurable effect on the final Cl.
    ncoarse = min(ngrid, max(200, 8 * (lmax + 1)))
    uc = np.linspace(0.0, 1.0, ncoarse)
    beta_coarse = eps + (beta_max - eps) * uc**3
    xvals_coarse = np.cos(beta_coarse)

    # C+(beta) at the native nodes (xvals) themselves is computed the same
    # way, in the same batch, rather than requiring the caller to have
    # already computed it -- both are cheap point sets (native: lmax+1;
    # coarse: ~8*(lmax+1)) evaluated with a single _cl2corr call each.
    n_native = len(xvals)
    xvals_all = np.concatenate([xvals, xvals_coarse])
    xi_p_all = _cl2corr(cl_sum, (2, 2), lmax=lmax, xvals=xvals_all)
    xi_mask_all = _cl2corr(cl_mask, (0, 0), lmax=lmax, xvals=xvals_all)
    with np.errstate(divide="ignore", invalid="ignore"):
        cp_all = np.where(xi_mask_all > 0, xi_p_all / xi_mask_all, 0.0)
    Xi_p, cp_coarse = cp_all[:n_native], cp_all[n_native:]
    cp_grid = CubicSpline(beta_coarse, cp_coarse)(beta_grid)

    # sin(beta)/cos(beta/2)**4 -> 0 as beta -> 0, no special-casing needed
    # there; singular as beta -> pi (see module docs/notebook)
    fsub1_grid = np.sin(beta_grid) / np.cos(beta_grid / 2) ** 4 * cp_grid
    fsub2_grid = np.tan(beta_grid / 2) ** 3 * cp_grid
    cumsum1_grid = cumulative_trapezoid(fsub1_grid, beta_grid, initial=0.0)
    cumsum2_grid = cumulative_trapezoid(fsub2_grid, beta_grid, initial=0.0)

    theta_capped = np.minimum(theta_nodes, thetamax)
    sum1_nodes = np.interp(theta_capped, beta_grid, cumsum1_grid)
    sum2_nodes = np.interp(theta_capped, beta_grid, cumsum2_grid)
    sum1_nodes[theta_capped <= 0] = 0.0
    sum2_nodes[theta_capped <= 0] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        c_beta = (
            Xi_p
            + sum1_nodes / np.sin(theta_nodes / 2) ** 2
            - 2 * sum2_nodes * (2 + np.cos(theta_nodes)) / np.sin(theta_nodes / 2) ** 4
        )
    return c_beta


def naturalspice(d, m, fields, theta_max=None, purify=False, apodization="logistic", progress: Progress | None = None):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: mask Cl
        fields: list of fields
        theta_max: maximum angle to use for the unmixing, in degrees. If None, use all angles.
        purify: whether to purify the EE/BB estimator (only affects s1=s2=2 fields)
        progress: optional progress reporter
    Returns:
        corr_d: Corrected Cl
    """
    if progress is None:
        progress = NoProgress()

    first_wd = list(d.values())[0]
    first_wm = list(m.values())[0]
    lmax = first_wd.shape[first_wd.axis[0]]
    lmax_mask = first_wm.shape[first_wm.axis[0]]

    # pad correlation functions to lmax_mask
    d = binned(d, np.arange(0, lmax_mask + 1))

    with progress.task("data correlations") as task:
        wd = cl2corr(d, progress=task)
    with progress.task("mask correlations") as task:
        wm = cl2corr(m, progress=task)
    with progress.task("unmixing") as task:
        corr_wd = _naturalspice(wd, wm, fields, theta_max=theta_max, apodization=apodization, progress=task)

    # trnasform back to Cl
    if purify:
        with progress.task("purified transform back to Cl") as task:
            # start from the regular (natural/mask-ratio) transform, which
            # already gives us correct TT/TE/EB -- purification (PolSpice's
            # "decouple") only changes how EE/BB are estimated.
            corr_d = corr2cl(corr_wd)

            masks = {}
            for key, field in fields.items():
                if field.mask is not None:
                    masks[key] = field.mask

            thetamax_rad = np.pi if theta_max is None else np.radians(theta_max)

            spin2_keys = [
                key for key, cwd in corr_wd.items() if cwd.spin[0] != 0 and cwd.spin[1] != 0
            ]
            current, total = 0, len(spin2_keys)
            for key in spin2_keys:
                current += 1
                task.update(current, total)

                # PolSpice's decoupled EE/BB (Chon et al. 2004, eq. 60-65;
                # spice_subs.f90 -> deal_with_xi_and_cl.f90/cumul2.f90) is
                # built in three stages: (1) the *natural*, unweighted
                # mask-ratio correlation Xi_p = Xi_data/Xi_mask (note: NOT
                # corr_wd, which additionally carries _naturalspice's
                # logistic/gaussian apodization -- correct_xi_from_mask in
                # PolSpice is a plain ratio, apodization is a separate,
                # later step); (2) a cumulative-integral correction
                # (`cumul`) that turns that into the "pure" E/B correlation
                # function, accounting for E/B leakage from the finite
                # integration range; (3) a Legendre transform weighted by
                # the csc^2(theta/2) kernel, normalized by Fl -- the same
                # kernel applied to the apodization window alone.
                #
                # Deriving PolSpice's xi(:,2)/xi(:,3) (its natural-ratio
                # QQ/UU, pre-cumul) in terms of heracles' Xi_p/Xi_m gives
                # exactly xi2 = (Xi_p+Xi_m)/2, xi3 = (Xi_p-Xi_m)/2, so
                # xi2-xi3 = Xi_m and xi2+xi3 = Xi_p; cumul()'s own
                # xi_E/B_final = (c_beta +/- (xi2-xi3))/2 therefore only
                # needs Xi_m (not Xi_p) here -- verified to < 0.1% against
                # PolSpice's own cumul() dump (SPICE_CUMUL_DEBUG) at every
                # node except where xi_B_final crosses zero.
                xvals = get_result_array(wd[key], "ell")[0]
                key_lmax = len(xvals) - 1
                theta = np.degrees(np.arccos(xvals))
                # NOTE: this apodization window is PolSpice's independent
                # -apodizesigma option, *not* the same thing as theta_max
                # (-thetamax, the integration cutoff handled via
                # thetamax_rad below) -- naturalspice doesn't currently
                # expose apodizesigma separately. Chon et al.
                # (2004) recommend apodizesigma = theta_max/2, so that is
                # used as the default width whenever theta_max is given
                # (verified against PolSpice's own Fl(l) dump,
                # SPICE_FL_DEBUG, with matching -apodizesigma: exact to
                # machine precision -- using theta_max itself as the width,
                # the previous behaviour, was wrong by a large,
                # l-dependent factor).
                apod = (
                    gaussian(theta, theta_max / 2, thetamax=theta_max)
                    if theta_max is not None
                    else np.ones_like(theta)
                )
                with np.errstate(divide="ignore"):
                    csc2 = 1.0 / np.sin(np.radians(theta) / 2) ** 2

                # Purify
                a, b, i, j = key
                m_key = (masks[a], masks[b], i, j)
                wm_arr = get_cl(m_key, wm).array
                Xi_m = wd[key][1, 1] / wm_arr
                cl_ee_raw = d[key].array[0, 0]
                cl_bb_raw = d[key].array[1, 1]
                cl_mask_raw = get_cl(m_key, m).array
                Xi_p_dec = purify_xip(
                    cl_ee_raw, cl_bb_raw, cl_mask_raw, key_lmax, xvals, thetamax_rad
                )
                xi_EE = 0.5 * (Xi_p_dec + Xi_m)
                xi_BB = 0.5 * (Xi_p_dec - Xi_m)

                # Apodize
                xi_EE = xi_EE * apod
                xi_BB = xi_BB * apod

                # Transform and normalize by Fl (the same kernel applied to the apodization window alone)
                fl = _corr2cl(apod * csc2, (2, -2), key_lmax)
                cl_EE = 2 * np.pi * _corr2cl(xi_EE, (2, -2), key_lmax) / fl
                cl_BB = 2 * np.pi * _corr2cl(xi_BB, (2, -2), key_lmax) / fl

                # Replace the EE/BB entries in the output dictionary with the purified values
                cl = np.array(corr_d[key].array, copy=True)
                cl[0, 0] = cl_EE
                cl[1, 1] = cl_BB
                corr_d[key] = replace(corr_d[key], array=cl)
    else:
        with progress.task("transform back to Cl") as task:
            corr_d = corr2cl(corr_wd, progress=task)

    # truncate to lmax
    corr_d = binned(corr_d, np.arange(0, lmax + 1))
    return corr_d


def _naturalspice(wd, wm, fields, theta_max=None, apodization="logistic", progress: Progress | None = None):
    """
    Natural unmixing of the data correlation function.
    Args:
        wd: data correlation function
        wm: mask correlation function
        fields: list of fields
        theta_max: maximum angle in degrees for the logistic cutoff. If None, uses default x0=-2.
        progress: optional progress reporter
    Returns:
        corr_d: Corrected Cl
    """
    if progress is None:
        progress = NoProgress()

    masks = {}
    for key, field in fields.items():
        if field.mask is not None:
            masks[key] = field.mask

    if theta_max is not None:
        first_wm = list(wm.values())[0]
        # reuse wm's own grid rather than recomputing a fresh one
        xvals = get_result_array(first_wm, "ell")[0]
        theta = np.arccos(xvals) * 180 / np.pi
        i_theta_max = np.abs(theta - theta_max).argmin()
        x0 = np.log10(abs(first_wm[i_theta_max]))
    else:
        x0 = -5

    corr_wds = {}
    current, total = 0, len(wd)
    for key in wd.keys():
        current += 1
        progress.update(current, total)
        a, b, i, j = key
        m_key = (masks[a], masks[b], i, j)
        # get_cl returns the array stored in wm/wd by reference (not a
        # copy), so *=/ /= below would otherwise mutate wm's own arrays in
        # place -- corrupting later, unrelated uses of wm (e.g. purify's
        # own C+(beta) computation, which needs the pristine, unapodized
        # mask correlation)
        _wm = get_cl(m_key, wm).array.copy()
        _wd = wd[key].array
        if apodization == "logistic":
            _wm *= logistic(np.log10(abs(_wm)), x0=x0)
        elif apodization == "gaussian":
            xvals = wm[m_key].ell
            theta = np.degrees(np.arccos(xvals))
            _wm /= gaussian(theta, theta_max)
        corr_wds[key] = replace(wd[key], array=_wd/_wm)

    return corr_wds
