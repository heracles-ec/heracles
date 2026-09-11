import numpy as np
import heracles


def test_cl2corr_corr2cl_roundtrip(cls0):
    from heracles.utils import get_cl

    # TT round-trip (spin (0, 0))
    cl_tt = get_cl(("POS", "POS", 1, 1), cls0)
    corr_tt = heracles.transforms._cl2corr(cl_tt, (0, 0))
    _cl_tt = heracles.transforms._corr2cl(corr_tt, (0, 0))
    assert np.isclose(cl_tt[2:], _cl_tt[2:]).all()

    # EE/BB round-trip (spin (2, 2), no EB/BE cross-term): rotate manually
    # into the "+"/"-" combinations, then transform each rotated component
    # with its own kernel -- (2, 2) for "+", (2, -2) for "-"
    cl_ee = get_cl(("SHE", "SHE", 1, 1), cls0)[0, 0]
    cl_bb = get_cl(("SHE", "SHE", 1, 1), cls0)[1, 1]
    cp, cm = cl_ee + cl_bb, cl_ee - cl_bb
    xi_p = heracles.transforms._cl2corr(cp, (2, 2))
    xi_m = heracles.transforms._cl2corr(cm, (2, -2))
    _cp = heracles.transforms._corr2cl(xi_p, (2, 2))
    _cm = heracles.transforms._corr2cl(xi_m, (2, -2))
    _cl_ee, _cl_bb = (_cp + _cm) / 2, (_cp - _cm) / 2
    assert np.isclose(cl_ee[2:], _cl_ee[2:]).all()
    assert np.isclose(cl_bb[2:], _cl_bb[2:]).all()

    # TE round-trip (one spin zero, no TB counterpart): both rotated
    # combinations share the same (2, 0) kernel
    cl_te = get_cl(("POS", "SHE", 1, 1), cls0)[0]
    cp, cm = cl_te, cl_te
    corr_p = heracles.transforms._cl2corr(cp, (2, 0))
    corr_m = heracles.transforms._cl2corr(cm, (2, 0))
    _cp = heracles.transforms._corr2cl(corr_p, (2, 0))
    _cm = heracles.transforms._corr2cl(corr_m, (2, 0))
    _cl_te = (_cp + _cm) / 2
    assert np.isclose(cl_te[2:], _cl_te[2:]).all()
