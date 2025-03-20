import numpy as np
from heracles.result import Result
from .polspice import (
    cl2corr,
    corr2cl,
)


def mask_correction(Mljk, Mls0):
    """
    Internal method to compute the mask correction.
    input:
        Mljk (np.array): mask of delete1 Cls
        Mls0 (np.array): mask Cls
    returns:
        alpha (Float64): Mask correction factor
    """
    _Mls0 = np.array(
        [
            Mls0,
            np.zeros_like(Mls0),
            np.zeros_like(Mls0),
            np.zeros_like(Mls0),
        ]
    )
    _Mljk = np.array(
        [
            Mljk,
            np.zeros_like(Mljk),
            np.zeros_like(Mljk),
            np.zeros_like(Mljk),
        ]
    )
    # Transform to real space
    wMls0 = cl2corr(_Mls0.T)
    wMls0 = wMls0.T[0]
    wMljk = cl2corr(_Mljk.T)
    wMljk = wMljk.T[0]
    # Compute alpha
    alpha = wMls0 / wMljk
    alpha *= logistic(np.log10(abs(wMljk)))
    alpha *= logistic(np.log10(abs(wMls0)))
    return alpha


def correct_mask(Cljk, Mljk, Mls0):
    """
    Private method to correct the fact that when a Jackknife region is removed,
    the mask of the region changes, which affects the Cls.
    returns:
        Cljk_corr (dict): Corrected Cls
    """
    corr_Cljk = {}
    Cl_keys = list(Cljk.keys())
    Clmm_keys = list(Mljk.keys())
    for Cl_key, Clmm_key in zip(Cl_keys, Clmm_keys):
        # get alpha
        _Mls0 = Mls0[Clmm_key]
        _Mljk = Mljk[Clmm_key]
        alpha = mask_correction(_Mljk, _Mls0)
        # Grab metadata
        dtype = Cljk[Cl_key].__array__().dtype
        ell = Cljk[Cl_key].ell
        # Correct Cl by mask
        _Cljk = Cljk[Cl_key]
        k1, k2, b1, b2 = Cl_key
        if k1 == k2 == "POS":
            __Cljk = np.array(
                [
                    _Cljk,
                    np.zeros_like(_Cljk),
                    np.zeros_like(_Cljk),
                    np.zeros_like(_Cljk),
                ]
            )
            wCljk = cl2corr(__Cljk.T).T
            corr_wCljk = wCljk * alpha
            # Transform back to Cl
            __corr_Cljk = corr2cl(corr_wCljk.T).T
            _corr_Cljk = list(__corr_Cljk[0])
        elif k1 == k2 == "SHE":
            if b1 == b2:
                __Cljk = np.array(
                    [
                        np.zeros_like(_Cljk[0]),
                        _Cljk[0],  # EE like spin-2
                        _Cljk[1],  # BB like spin-2
                        np.zeros_like(_Cljk[0]),
                    ]
                )
                __iCljk = np.array(
                    [
                        np.zeros_like(_Cljk[0]),
                        -_Cljk[2],  # EB like spin-0
                        _Cljk[2],  # EB like spin-0
                        np.zeros_like(_Cljk[0]),
                    ]
                )
                # Correct by alpha
                wCljk = cl2corr(__Cljk.T).T + 1j * cl2corr(__iCljk.T).T
                corr_wCljk = (wCljk * alpha).real
                icorr_wCljk = (wCljk * alpha).imag
                # Transform back to Cl
                __corr_Cljk = corr2cl(corr_wCljk.T).T
                __icorr_Cljk = corr2cl(icorr_wCljk.T).T
                _corr_Cljk = [
                    __corr_Cljk[1],  # EE like spin-2
                    __corr_Cljk[2],  # BB like spin-2
                    -__icorr_Cljk[1],  # EB like spin-0
                ]
            if b1 != b2:
                __Cljk = np.array(
                    [
                        np.zeros_like(_Cljk[0]),
                        _Cljk[0],  # EE like spin-2
                        _Cljk[1],  # BB like spin-2
                        np.zeros_like(_Cljk[0]),
                    ]
                )
                __iCljk = np.array(
                    [
                        np.zeros_like(_Cljk[0]),
                        -_Cljk[2],  # EB like spin-0
                        _Cljk[3],  # BE like spin-0
                        np.zeros_like(_Cljk[0]),
                    ]
                )
                # Correct by alpha
                wCljk = cl2corr(__Cljk.T).T + 1j * cl2corr(__iCljk.T).T
                corr_wCljk = (wCljk * alpha).real
                icorr_wCljk = (wCljk * alpha).imag
                # Transform back to Cl
                __corr_Cljk = corr2cl(corr_wCljk.T).T
                __icorr_Cljk = corr2cl(icorr_wCljk.T).T
                _corr_Cljk = [
                    __corr_Cljk[1],  # EE like spin-2
                    __corr_Cljk[2],  # BB like spin-2
                    -__icorr_Cljk[1],  # EB like spin-0
                    __icorr_Cljk[2],  # BE like spin-0
                ]
        else:
            # Treat everything as spin-0
            _corr_Cljk = []
            for cl in _Cljk:
                __Cljk = np.array(
                    [
                        cl,
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                    ]
                )
                wCljk = cl2corr(__Cljk.T).T
                corr_wCljk = wCljk * alpha
                # Transform back to Cl
                __corr_Cljk = corr2cl(corr_wCljk.T).T
                _corr_Cljk.append(__corr_Cljk[0])

        # Undo at least2D
        if len(_corr_Cljk) == 1:
            _corr_Cljk = _corr_Cljk[0]
        # Add metadata back
        _corr_Cljk = np.array(_corr_Cljk, dtype=dtype)
        corr_Cljk[Cl_key] = Result(_corr_Cljk, ell=ell)
    return corr_Cljk


def logistic(x, x0=-5, k=50):
    return 1 / (1.0 + np.exp(-k * (x - x0)))
