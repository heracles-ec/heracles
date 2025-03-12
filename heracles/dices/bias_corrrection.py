import numpy as np
from heracles.result import Result
from .utils import (
    add_to_Cls,
    sub_to_Cls,
)


def get_bias(cls):
    """
    Internal method to compute the bias.
    inputs:
        cls (dict): Dictionary of Cls
    returns:
        bias (dict): Dictionary
    """
    bias = {}
    for key in list(cls.keys()):
        cl = cls[key]  # .__array__()
        meta = cl.dtype.metadata
        if "bias" in meta.keys():
            b = meta["bias"]
        else:
            b = 0
        bias[key] = b
    return bias


def get_delete_fsky(jkmaps, jk=0, jk2=0):
    """
    Returns the fraction of the sky after deleting two regions.
    inputs:
        jkmaps (dict): Dictionary of Jackknife maps
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        fskyjk2 (np.array): Fraction of the sky after deleting two regions.
    """
    rel_fskys = {}
    for key in jkmaps.keys():
        jkmap = jkmaps[key]
        mask = np.copy(jkmap)
        mask[mask != 0] = mask[mask != 0] / mask[mask != 0]
        fsky = sum(mask) / len(mask)
        cond = np.where((mask == 1.0) & (jkmap != jk) & (jkmap != jk2))[0]
        rel_fskys[key] = (len(cond) / len(mask)) / fsky
    return rel_fskys


def get_biasjk(bias, fsky):
    """
    Returns the bias for deleting a Jackknife region.
    inputs:
        bias (dict): Dictionary of biases
        fsky (dict): Dictionary of relative fskys
    returns:
        bias_jk (dict): Dictionary of biases
    """
    bias_jk = {}
    for key in list(bias.keys()):
        f1, f2, b1, b2 = key
        b = bias[key]
        if (f1, b1) == (f2, b2):
            if f1 == "POS":
                f1 = "VIS"
            elif f1 == "SHE":
                f1 = "WHT"
            fskyjk = fsky[(f1, b1)]
        else:
            fskyjk = 0.0
        b_jk = b * fskyjk
        bias_jk[key] = b_jk
    return bias


def correct_bias(cls, jkmaps, jk=0, jk2=0):
    """
    Corrects the bias of the Cls due to taking out a region.
    inputs:
        cls (dict): Dictionary of Cls
        jkmaps (dict): Dictionary of Jackknife maps
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        cls_cbias (dict): Corrected Cls
        cls_wbias (dict): Cls with bias
    """
    # Bias correction
    bias = get_bias(cls)
    fskyjk = get_delete_fsky(jkmaps, jk=jk, jk2=jk2)
    bias_jk = get_biasjk(bias, fskyjk)
    cls_wbias = add_to_Cls(cls, bias)
    cls_cbias = sub_to_Cls(cls_wbias, bias_jk)
    for key in cls_cbias.keys():
        cl = cls_cbias[key]
        meta = dict(cl.__array__().dtype.metadata)
        meta["bias"] = bias_jk[key]
        dtype = np.dtype(
            float,
            metadata=meta,
        )
        cls_cbias[key] = Result(np.array(cl, dtype=dtype), ell=cl.ell)
    return cls_cbias
