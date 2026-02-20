import sys
import types
import numpy as np
from scipy import integrate

import heracles
from heracles.result import Result


def test_transforms(cls0):
    corr = heracles.cl2corr(cls0)
    for key in list(cls0.keys()):
        assert corr[key].shape == cls0[key].shape

    _cls = heracles.corr2cl(corr)
    for key in list(cls0.keys()):
        assert _cls[key].shape == cls0[key].shape
        assert np.isclose(cls0[key].array, _cls[key].array).all()


def _mock_shear_cls(ells):
    cls = {}
    for i, j, amp in ((1, 1, 1.0), (1, 2, 1.5), (2, 2, 2.0)):
        arr = np.zeros((2, 2, ells.size), dtype=np.float64)
        arr[0, 0] = amp * (ells + 1.0)
        arr[1, 1] = amp * (ells + 2.0)
        cls[("SHE", "SHE", i, j)] = Result(
            array=arr,
            spin=(2, 2),
            axis=-1,
            ell=ells,
            lower=ells,
            upper=ells + 1,
        )
    return cls


def test_cl2cosebis_with_cloe_adapter_and_internal_kernel_build(monkeypatch):
    class FakeCosebi:
        def __init__(self, array, mode):
            self.array = array
            self.mode = mode

    class FakeAngularTwoPoint:
        def get_Cl(self, ells, nl, ks):
            raise AssertionError("Adapter should provide get_Cl")

        def get_cosebis(self, ells, nl, ks, w_ell, ns):
            cells = self.get_Cl(ells, nl, ks)
            tomo_cosebis = {}
            for i in range(1, self.tracer1.n_z_bins + 1):
                for j in range(i, self.tracer1.n_z_bins + 1):
                    key = ("SHE", "SHE", i, j)
                    if key not in cells:
                        continue
                    cl = cells[key][0, 0]
                    array = np.array(
                        [
                            integrate.simpson(
                                ells * cl * w_ell[int(n)],
                                x=ells,
                            )
                            / (2 * np.pi)
                            for n in ns
                        ]
                    )
                    tomo_cosebis[key] = FakeCosebi(array=array, mode=ns)
            return tomo_cosebis

    captured = {}

    ells = np.arange(2, 10, dtype=np.float64)
    n_cosebis = 2
    ns = np.arange(1, n_cosebis + 1, dtype=int)
    cls = _mock_shear_cls(ells)

    def fake_get_w_ell(thetagrid, nmax, ells, n_thread):
        captured["thetagrid"] = thetagrid
        captured["nmax"] = nmax
        captured["ells"] = ells
        captured["n_thread"] = n_thread
        return {1: np.ones_like(ells), 2: 0.25 * np.ones_like(ells)}

    cloelib = types.ModuleType("cloelib")
    cloelib.__path__ = []
    auxiliary = types.ModuleType("cloelib.auxiliary")
    auxiliary.__path__ = []
    cosebi_helpers = types.ModuleType("cloelib.auxiliary.cosebi_helpers")
    summary_statistics = types.ModuleType("cloelib.summary_statistics")
    summary_statistics.__path__ = []
    angular_two_point = types.ModuleType("cloelib.summary_statistics.angular_two_point")

    cosebi_helpers.get_W_ell = fake_get_w_ell
    angular_two_point.AngularTwoPoint = FakeAngularTwoPoint
    cloelib.auxiliary = auxiliary
    cloelib.summary_statistics = summary_statistics
    auxiliary.cosebi_helpers = cosebi_helpers
    summary_statistics.angular_two_point = angular_two_point

    monkeypatch.setitem(sys.modules, "cloelib", cloelib)
    monkeypatch.setitem(sys.modules, "cloelib.auxiliary", auxiliary)
    monkeypatch.setitem(sys.modules, "cloelib.auxiliary.cosebi_helpers", cosebi_helpers)
    monkeypatch.setitem(sys.modules, "cloelib.summary_statistics", summary_statistics)
    monkeypatch.setitem(
        sys.modules,
        "cloelib.summary_statistics.angular_two_point",
        angular_two_point,
    )

    cosebis = heracles.cl2cosebis(
        cls,
        n_cosebis,
        n_thread=3,
    )

    assert captured["thetagrid"].ndim == 1
    assert captured["thetagrid"].size >= 64
    assert np.all(captured["thetagrid"] > 0)
    assert np.all(np.diff(captured["thetagrid"]) > 0)
    assert captured["nmax"] == 2
    assert np.array_equal(captured["ells"], ells)
    assert captured["n_thread"] == 3

    w_ell = fake_get_w_ell(captured["thetagrid"], int(np.max(ns)), ells, 3)
    assert set(cosebis) == set(cls)
    for key in cls:
        cl = cls[key][0, 0]
        expected = np.array(
            [
                integrate.simpson(ells * cl * w_ell[int(n)], x=ells) / (2 * np.pi)
                for n in ns
            ]
        )
        assert np.allclose(cosebis[key].array, expected)
        assert np.array_equal(cosebis[key].ell, ns)
