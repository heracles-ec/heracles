import numpy as np
from copy import deepcopy
from ..fields import Positions, Shears, Visibility, Weights
from ..mapping import transform
from ..result import Result
from ..healpy import HealpixMapper
from ..twopoint import angular_power_spectra


def get_cls(data_maps, vis_maps):
    """
    Internal method to compute the Cls.
    """
    # grab metadata
    _d = data_maps[list(data_maps.keys())[0]]
    meta = _d.dtype.metadata
    nside = meta["nside"]
    lmax = meta["lmax"]
    ls = np.arange(lmax + 1)
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = {
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }

    # compute cls
    alms = transform(fields, data_maps)
    cls = angular_power_spectra(alms)
    alms_mm = transform(fields, vis_maps)
    cls_mm = angular_power_spectra(alms_mm)
    for key in cls.keys():
        cls[key] = Result(cls[key], ell=ls)
    for key in cls_mm.keys():
        cls_mm[key] = Result(cls_mm[key], ell=ls)
    return cls, cls_mm


def get_delete_cls(data_maps, vis_maps, jkmaps, jk, jk2):
    """
    Internal method to compute the Cls of removing 2 Jackknife.
    returns:
        cls (dict): Dictionary of data Cls
        cls_mm (dict): Dictionary of mask Cls
    """
    # grab metadata
    print(f" - Computing Cls for regions ({jk},{jk2})", end="\r", flush=True)
    _d = data_maps[list(data_maps.keys())[0]]
    meta = _d.dtype.metadata
    nside = meta["nside"]
    lmax = meta["lmax"]
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = {
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT")}
    fields_vis = {
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }

    # deep copy to avoid modifying the original maps
    vmaps = deepcopy(vis_maps)
    datas = deepcopy(data_maps)
    for key_data, key_mask in zip(datas.keys(), vmaps.keys()):
        _data = datas[key_data]
        _vmap = vmaps[key_mask]
        _jkmap = jkmaps[key_mask]
        _mask = np.copy(_jkmap)
        _mask[_mask != 0] = _mask[_mask != 0] / _mask[_mask != 0]

        # Remove jk 2 regions
        cond = np.where((_jkmap == float(jk)) | (_jkmap == float(jk2)))[0]
        _mask[cond] = 0.0
        if key_data[0] == "POS":
            _data *= _mask
        elif key_data[0] == "SHE":
            _data[0] *= _mask
            _data[1] *= _mask
        else:
            raise ValueError(f"{key_data[0]} Unknown field type")
        _vmap *= _mask
    # compute alms
    alms = transform(fields, datas)
    alms_mm = transform(fields_vis, vmaps)
    # compute cls
    cls = angular_power_spectra(alms)
    cls_mm = angular_power_spectra(alms_mm)
    return cls, cls_mm
