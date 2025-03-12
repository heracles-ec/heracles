import numpy as np
from copy import deepcopy
from ..fields import Positions, Shears, Visibility, Weights
from ..mapping import transform
from ..result import Result
from ..healpy import HealpixMapper
from ..twopoint import angular_power_spectra


def get_cls(maps, jkmaps, jk=0, jk2=0):
    """
    Internal method to compute the Cls of removing 2 Jackknife.
    inputs:
        maps (dict): Dictionary of data maps
        jkmaps (dict): Dictionary of mask maps
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        cls (dict): Dictionary of data Cls
    """
    # grab metadata
    print(f" - Computing Cls for regions ({jk},{jk2})", end="\r", flush=True)
    _m = maps[list(maps.keys())[0]]
    meta = _m.dtype.metadata
    nside = meta["nside"]
    lmax = meta["lmax"]
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = {
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }

    # deep copy to avoid modifying the original maps
    _maps = deepcopy(maps)
    for key_data, key_mask in zip(maps.keys(), jkmaps.keys()):
        _map = _maps[key_data]
        _jkmap = jkmaps[key_mask]
        _mask = np.copy(_jkmap)
        _mask[_mask != 0] = _mask[_mask != 0] / _mask[_mask != 0]
        # Remove jk 2 regions
        cond = np.where((_jkmap == float(jk)) | (_jkmap == float(jk2)))[0]
        _mask[cond] = 0.0
        # Apply mask
        _map *= _mask
    # compute alms
    alms = transform(fields, _maps)
    # compute cls
    cls = angular_power_spectra(alms)
    return cls
