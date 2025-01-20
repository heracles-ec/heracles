import os
import math
import heracles
import numpy as np
import healpy as hp
from copy import deepcopy
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform
from heracles.healpy import HealpixMapper
from heracles.result import binned
from heracles.io import write, read

from .utils_cl import (
    get_lgrid,
    get_Cls_bias,
    add_to_Cls,
    compsep_Cls,
    sub_to_Cls,
    get_covkeys,
    cov2corr,
    make_posdef,
    get_Cl_cov,
)

from .utils_sh import (
    get_W,
    get_T_rbar,
    get_T_new,
    get_lambda_star_single_rbar,
)

from .utils_polspice import (
    cl2corr,
    corr2cl,
)


class DICES:
    def __init__(self, data_maps, jkmap, vis_maps, config):
        self.data_maps = data_maps
        self.jkmap = jkmap
        self.mask = np.copy(jkmap)
        self.mask[self.mask != 0] = (
            self.mask[self.mask != 0] / self.mask[self.mask != 0]
        )
        self.vis_maps = vis_maps
        self._load_defaults(config)

        # Step 1 Make Cls
        self.data_cls, self.mask_cls = self.get_cls()
        self.Cl_keys = list(self.data_cls.keys())
        self.Clmm_keys = list(self.mask_cls.keys())
        self.covkeys = get_covkeys(self.Cl_keys)

        self.delete1_data_cls = None
        self.delete2_data_cls = None
        self.delete1_mask_cls = None
        self.delete2_mask_cls = None

        # Step 2 Correct Cls' bias
        self.ngal = None
        self.wmean = None
        self.var = None
        self.bias = None
        self.bias_jk = None
        self.bias_jk2 = None

        # Step 4 Make delte1 covariance
        self.delete1_cov = None
        self.target_cov = None

        # Step 5 Apply delete2 correction
        self.delete2_cov = None

        # Step 6 Get Dices Covariance
        self.dices_cov = None

    def _load_defaults(self, config):
        # Basics
        self.config = config
        self.JackNjk = int(config["Njk"])
        self.nside = config["Nside"]
        self.npix = hp.nside2npix(self.nside)
        self.save = bool(config["Output"]["save"])
        self.output_path = str(config["Output"]["path"])

        if self.save:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.output_path + "cls"):
                os.mkdir(self.output_path + "cls")
            if not os.path.exists(self.output_path + "cov"):
                os.mkdir(self.output_path + "cov")

        # Mapper
        self.ls = np.arange(self.nside + 1)
        self.mapper = HealpixMapper(nside=self.nside, lmax=self.nside)
        # make unseen pixels 0
        self.mask[self.mask == hp.UNSEEN] = 0.0
        # Upgrade mask to wanted nside
        if hp.npix2nside(len(self.mask)) != self.nside:
            self.mask = hp.ud_grade(self.mask, self.nside)
        self.fsky = np.sum(self.mask) / len(self.mask)

        # Fields
        self.Nbins = int(self.config["Fields"]["Nbins"])
        self.Keys_Pos_Ra = str(self.config["Fields"]["Keys"]["Pos"]["Ra"])
        self.Keys_Pos_Dec = str(self.config["Fields"]["Keys"]["Pos"]["Dec"])
        self.Pos_lonlat = (self.Keys_Pos_Ra, self.Keys_Pos_Dec)
        self.Keys_She_Ra = str(self.config["Fields"]["Keys"]["She"]["Ra"])
        self.Keys_She_Dec = str(self.config["Fields"]["Keys"]["She"]["Dec"])
        self.She_lonlat = (self.Keys_She_Ra, self.Keys_She_Dec)
        self.Keys_She_E1 = str(self.config["Fields"]["Keys"]["She"]["E1"])
        self.Keys_She_E2 = str(self.config["Fields"]["Keys"]["She"]["E2"])
        self.Keys_She_Weights = str(self.config["Fields"]["Keys"]["She"]["Weights"])

        # Maps
        self.fields = {
            "POS": Positions(self.mapper, *self.Pos_lonlat, mask="VIS"),
            "SHE": Shears(
                self.mapper,
                *self.She_lonlat,
                self.Keys_She_E1,
                self.Keys_She_E2,
                self.Keys_She_Weights,
                mask="WHT",
            ),
        }

        self.fields_mm = {
            "VIS": Visibility(self.mapper),
            "WHT": Weights(self.mapper, *self.She_lonlat, self.Keys_She_Weights),
        }

        # Covariance Parameters
        self.Cov_Lmin = int(config["Cov"]["Lmin"])
        self.Cov_Lmax = int(config["Cov"]["Lmax"])
        self.Cov_Lbin = int(config["Cov"]["Lbin"])
        self.Cov_Logbins = bool(config["Cov"]["Logbins"])

        # Cl bins
        self.lgrid, self.ledges, self.dl = get_lgrid(
            self.Cov_Lmin, self.Cov_Lmax, self.Cov_Lbin, uselog=self.Cov_Logbins
        )
        # Polspice
        self.kL = 50
        self.epsilon = -5

        # DICES
        self.mask_correction = bool(config["DICES"]["mask_correction"])
        self.shrinkage = bool(config["DICES"]["shrinkage"])
        self.delete2_correction = bool(config["DICES"]["delete2_correction"])

    def get_dices_cov(self):
        """
        Checks if the Dices covariance exists.
        If not, load it from file. If it does not exist, compute it.
        returns:
            dices_cov (dict): Dictionary of Dices covariance
        """
        if self.dices_cov is None:
            fname = self.output_path + "cov/jackcov_dices_njk_%i.fits" % (self.JackNjk)
            if os.path.exists(fname):
                print(f"Dices covariance exists: {fname}")
                self.dices_cov = read(fname)
            else:
                dices_cov = self.get_delete1_cov(
                    mask_correction=self.mask_correction,
                    shrink=self.shrinkage,
                )
                if self.delete2_correction:
                    delete2_cov = self.get_delete2_cov(
                        mask_correction=self.mask_correction,
                    )
                    dices_cov = self._get_delete2_correction(dices_cov, delete2_cov)

                self.dices_cov = dices_cov
                if self.save:
                    write(fname, self.dices_cov)
        return self.dices_cov

    ###
    # Step 1: Make Cls
    ###

    def get_cls(self):
        """
        Checks if Cls exist, if not, load them.
        If files do not exist, compute them.
        returns:
            data_cls (dict): Dictionary of data Cls
            mask_cls (dict): Dictionary of mask Cls
        """
        data_fname = self.output_path + "cls/cls_nojk.fits"
        mask_fname = self.output_path + "cls/mls_nojk.fits"
        if os.path.exists(data_fname) & os.path.exists(mask_fname):
            print(f"Data Cls exists: {data_fname}")
            print(f"Mask Cls exists: {mask_fname}")
            self.data_cls = read(data_fname)
            self.mask_cls = read(mask_fname)
        else:
            self.data_cls, self.mask_cls = self._get_cls()
            if self.save:
                write(data_fname, self.data_cls)
                write(mask_fname, self.mask_cls)
        return self.data_cls, self.mask_cls

    def _get_cls(self):
        """
        Internal method to compute the Cls.
        """
        # Deep copy to avoid imodifying the original maps
        _mask = np.copy(self.mask)
        data = {}
        for key in self.data_maps.keys():
            if key[0] == "POS" or key[0] == "SHE":
                data[key] = self.data_maps[key] * _mask
            else:
                print(key, " Unknown field type")
        # compute to alms
        alms = transform(self.fields, data)
        # compute cls
        cls = heracles.angular_power_spectra(alms)

        data_mm = deepcopy(self.vis_maps)
        for key in data_mm.keys():
            data_mm[key] *= _mask
        alms_mm = transform(self.fields_mm, data_mm)
        cls_mm = heracles.angular_power_spectra(alms_mm)
        return cls, cls_mm

    def get_data_cls(self):
        """
        Returns data Cls
        returns:
            data_cls (dict): Dictionary of data Cls
        """
        self.get_cls()
        return self.data_cls

    def get_mask_cls(self):
        """
        Returns mask Cls
        returns:
            mask_cls (dict): Dictionary of mask Cls
        """
        self.get_cls()
        return self.mask_cls

    def get_delete1_cls(self, mask_correction=False):
        """
        Check if the Cls of removing 1 Jackknife region exist, if not, load them.
        If files do not exist, compute them.
        input:
            mask_correction (bool): If True, correct the mask Cls
        returns:
            delete1_data_cls (dict): Dictionary of delete1 data Cls
            delete1_mask_cls (dict): Dictionary of delete1 mask Cls
        """
        if (self.delete1_data_cls is None) or (self.delete1_mask_cls is None):
            self.delete1_data_cls = {}
            self.delete1_mask_cls = {}
            for jk in range(1, self.JackNjk + 1):
                data_fname = self.output_path + "cls/cls_njk_%i_jkid_%i.fits" % (
                    self.JackNjk,
                    jk,
                )
                mask_fname = self.output_path + "cls/mls_njk_%i_jkid_%i.fits" % (
                    self.JackNjk,
                    jk,
                )

                if os.path.exists(data_fname) & os.path.exists(mask_fname):
                    print(f"Delete1 Data Cls exists: {data_fname}")
                    print(f"Delete1 Mask Cls exists: {mask_fname}")
                    self.delete1_data_cls[jk] = read(data_fname)
                    self.delete1_mask_cls[jk] = read(mask_fname)
                else:
                    self.get_bias()
                    self.get_delete1_bias()
                    cls, cls_mm = self._get_delete1_cls(jk)
                    cls_wbias = add_to_Cls(cls, self.bias)
                    cls_cbias = sub_to_Cls(cls_wbias, self.bias_jk[jk])
                    if mask_correction:
                        _, mls0 = self.get_cls()
                        cls_cbias = self._correct_data_cls_mask(cls_cbias, cls_mm, mls0)
                    self.delete1_data_cls[jk] = cls_cbias
                    self.delete1_mask_cls[jk] = cls_mm
                    # Save Cls
                    if self.save:
                        write(data_fname, cls_cbias)
                        write(mask_fname, cls_mm)
        return (
            self.delete1_data_cls,
            self.delete1_mask_cls,
        )

    def _get_delete1_cls(self, jk):
        """
        Internal method to compute the Cls of removing 1 Jackknife.
        returns:
            cls (dict): Dictionary of data Cls
            cls_mm (dict): Dictionary of mask Cls
        """
        # Load previous steps

        # deep copy to avoid modifying the original maps
        _mask = np.copy(self.mask)
        # Remove jk region
        cond = np.where(self.jkmap == float(jk))[0]
        _mask[cond] = 0.0
        data = deepcopy(self.data_maps)
        for key in data.keys():
            if key[0] == "POS":
                data[key] *= _mask
            else:
                data[key][0] *= _mask
                data[key][1] *= _mask
        # Compute to alms
        alms = transform(self.fields, data)
        # Compute cls from alms
        cls = heracles.angular_power_spectra(alms)

        # Compute mask Cls
        data_mm = deepcopy(self.vis_maps)
        for key in data_mm.keys():
            data_mm[key] *= _mask
        # compute alms
        alms_mm = transform(self.fields_mm, data_mm)
        # compute cls
        cls_mm = heracles.angular_power_spectra(alms_mm)
        print("Making delete-1 Cls - JK %i/%i" % (jk, self.JackNjk))
        return cls, cls_mm

    def get_delete2_cls(self, mask_correction=False):
        """
        Check if the Cls of removing 2 Jackknife regions exist, if not, load them.
        If files do not exist, compute them.
        input:
            mask_correction (bool): If True, correct the mask Cls
        returns:
            delete2_data_cls (dict): Dictionary of delete2 data Cls
            delete2_mask_cls (dict): Dictionary of delete2 mask Cls
        """
        if (self.delete2_data_cls is None) or (self.delete2_mask_cls is None):
            self.delete2_data_cls = {}
            self.delete2_mask_cls = {}
            for jk in range(1, self.JackNjk + 1):
                for jk2 in range(jk + 1, self.JackNjk + 1):
                    data_fname = (
                        self.output_path
                        + "cls/cls_njk_%i_jkid2_%i_%i.fits" % (self.JackNjk, jk, jk2)
                    )
                    mask_fname = (
                        self.output_path
                        + "cls/mls_njk_%i_jkid2_%i_%i.fits" % (self.JackNjk, jk, jk2)
                    )
                    if os.path.exists(data_fname) & os.path.exists(mask_fname):
                        print(f"Delete2 Data Cls exists: {data_fname}")
                        print(f"Delete2 Mask Cls exists: {mask_fname}")
                        self.delete2_data_cls[(jk, jk2)] = read(data_fname)
                        self.delete2_mask_cls[(jk, jk2)] = read(mask_fname)
                    else:
                        self.get_bias()
                        self.get_cls()
                        self.get_delete2_cls()
                        self.get_delete2_bias()

                        cls, cls_mm = self._get_delete2_cls(jk, jk2)
                        cls_wbias = add_to_Cls(cls, self.bias)
                        cls_cbias = sub_to_Cls(cls_wbias, self.bias_jk2[(jk, jk2)])
                        if mask_correction:
                            _, mls0 = self.get_cls()
                            cls_cbias = self._correct_data_cls_mask(
                                cls_cbias, cls_mm, mls0
                            )

                        self.delete2_data_cls[(jk, jk2)] = cls_cbias
                        self.delete2_mask_cls[(jk, jk2)] = cls_mm
                        # Save Cls
                        if self.save:
                            write(data_fname, cls_cbias)
                            write(mask_fname, cls_mm)
        return (
            self.delete2_data_cls,
            self.delete2_mask_cls,
        )

    def _get_delete2_cls(self, jk, jk2):
        """
        Internal method to compute the Cls of removing 2 Jackknife.
        returns:
            cls (dict): Dictionary of data Cls
            cls_mm (dict): Dictionary of mask Cls
        """
        # Load previous steps

        # deep copy to avoid modifying the original maps
        _mask = np.copy(self.mask)
        if jk != 0:
            # Remove jk regions
            cond = np.where((self.jkmap == float(jk)) | (self.jkmap == float(jk2)))[0]
            _mask[cond] = 0.0
        data = deepcopy(self.data_maps)
        for key in data.keys():
            if key[0] == "POS":
                data[key] *= _mask
            else:
                data[key][0] *= _mask
                data[key][1] *= _mask
        # compute alms
        alms = transform(self.fields, data)
        # compute cls
        cls = heracles.angular_power_spectra(alms)

        # Compute mask Cls
        data_mm = deepcopy(self.vis_maps)
        for key in data_mm.keys():
            data_mm[key] *= _mask
        # compute alms
        alms_mm = transform(self.fields_mm, data_mm)
        # compute cls
        cls_mm = heracles.angular_power_spectra(alms_mm)
        print("Making delete-2 Cls - JK [%i, %i]/%i" % (jk, jk2, self.JackNjk))
        return cls, cls_mm

    def get_delete1_mask_cls(self):
        """
        Returns delete1 mask Cls
        returns:
            delete1_mask_cls (dict): Dictionary of delete1 mask Cls
        """
        self.get_delete1_cls()
        return self.delete1_mask_cls

    def get_delete1_data_cls(self):
        """
        Returns delete1 data Cls with the entire map bias added.
        returns and after substractig the Jackknife region bias:
            delete1_data_cls (dict): Dictionary of delete1 data Cls
        """
        if self.delete1_data_cls is None:
            self.get_delete1_cls()
        return self.delete1_data_cls

    def get_delete2_mask_cls(self):
        """
        Returns delete2 mask Cls
        returns:
            delete2_mask_cls (dict): Dictionary of delete2 mask Cls
        """
        self.get_delete2_cls()
        return self.delete2_mask_cls

    def get_delete2_data_cls(self):
        """
        Returns delete2 data Cls with the entire map bias and after
        substractig the Jackknife regions biases.
        returns:
            delete2_data_cls (dict): Dictionary of delete2 data Cls
        """
        if self.delete2_data_cls is None:
            self.get_delete2_cls()
        return self.delete2_data_cls

    ###
    # Step 3: correct Cls' bias
    ###

    def get_bias(self):
        """
        Checks if biases for each Cl exists.
        If not, load it from the maps metadata.
        returns:
            bias (dict): Dictionary of biases
        """
        if self.bias is None:
            self.bias = self._get_bias()
        return self.bias

    def _get_bias(self):
        """
        Internal method to compute the bias.
        returns:
            bias (dict): Dictionary
        """
        # Compute Entire Region Biases
        bias_pos_all = np.zeros(self.Nbins)
        bias_she_all = np.zeros(self.Nbins)
        for key in list(self.data_maps.keys()):
            data_map = self.data_maps[key]
            meta = data_map.dtype.metadata
            if key[0] == "POS":
                bias_pos_all[key[1] - 1] = meta["bias"]
            elif key[0] == "SHE":
                bias_she_all[key[1] - 1] = meta["bias"]
        print(" - Biases for entire region:", bias_pos_all, bias_she_all)
        bias = get_Cls_bias(self.Cl_keys, bias_pos_all, bias_she_all)
        return bias

    def get_delete1_bias(self):
        """
        Bias of each delete 1 Cl needs to be modified
        since the number of galaxies in the region changes
        when a region is removed.
        Checks if delete1 biases for each Cl exists.
        If not, compute the bias after removing
        one Jackknife region:
            bias_jk (dict): Dictionary of delete1 biases
        """
        if self.bias_jk is None:
            self.bias_jk = self._get_delete1_bias()
        return self.bias_jk

    def _get_delete1_bias(self):
        """
        Internal method to compute the bias after removing one Jackknife region.
        We do so by adding the biases of each Jackknife region.
        returns:
            bias_jk (dict): Dictionary of delete1 biases
        """
        print("Computing Ngal and Wmean/var for Jackknife regions")
        bias_pos_jk = []
        bias_she_jk = []
        fskyjk = self.get_delete1_fsky()
        for j in range(1, self.JackNjk + 1):
            _fskyjk = fskyjk[j - 1]
            _bias_pos_jk = np.zeros(self.Nbins)
            _bias_she_jk = np.zeros(self.Nbins)
            for key in list(self.data_maps.keys()):
                data_map = self.data_maps[key]
                meta = data_map.dtype.metadata
                if key[0] == "POS":
                    _bias_pos_jk[key[1] - 1] = (_fskyjk / self.fsky) * meta["bias"]
                elif key[0] == "SHE":
                    _bias_she_jk[key[1] - 1] = (_fskyjk / self.fsky) * meta["bias"]
            bias_pos_jk.append(_bias_pos_jk)
            bias_she_jk.append(_bias_she_jk)
        print(" - Biases for jackknife regions:", bias_pos_jk, bias_she_jk)

        _bias_jk = [
            get_Cls_bias(self.Cl_keys, bias_pos_jk[i], bias_she_jk[i])
            for i in range(0, len(bias_pos_jk))
        ]
        bias_jk = {}
        for j in range(1, self.JackNjk + 1):
            bias_jk[j] = _bias_jk[j - 1]
        return bias_jk

    def get_delete2_bias(self):
        """
        Checks if biases for each delete2 Cl exists.
        if not, compute it.
        reutrns:
            bias_jk2 (dict): Dictionary of delete2 biases
        """
        if self.bias_jk2 is None:
            self.bias_jk2 = self._get_delete2_bias()
        return self.bias_jk2

    def _get_delete2_bias(self):
        """
        Internal method to compute the bias after
        removing two Jackknife regions.
        We do so by adding the biases of each Jackknife
        region except for the two being removed.
        returns:
            bias_jk2 (dict): Dictionary of delete2 biases
        """
        print("Computing Ngal and Wmean/var for delete-2 Jackknife regions")
        bias_pos_jk2 = []
        bias_she_jk2 = []
        fskyjk2 = self.get_delete2_fsky()
        for j1 in range(0, self.JackNjk):
            for j2 in range(j1 + 1, self.JackNjk):
                _fskyjk2 = fskyjk2[j1, j2]
                _bias_pos_jk2 = np.zeros(self.Nbins)
                _bias_she_jk2 = np.zeros(self.Nbins)
                for key in list(self.data_maps.keys()):
                    data_map = self.data_maps[key]
                    meta = data_map.dtype.metadata
                    if key[0] == "POS":
                        _bias_pos_jk2[key[1] - 1] = (_fskyjk2 / self.fsky) * meta["bias"]
                    elif key[0] == "SHE":
                        _bias_she_jk2[key[1] - 1] = (_fskyjk2 / self.fsky) * meta["bias"]
                bias_pos_jk2.append(_bias_pos_jk2)
                bias_she_jk2.append(_bias_she_jk2)
        _bias_jk2 = [
            get_Cls_bias(self.Cl_keys, bias_pos_jk2[i], bias_she_jk2[i])
            for i in range(0, len(bias_pos_jk2))
        ]
        bias_jk2 = {}
        i = 0
        for j1 in range(0, self.JackNjk):
            for j2 in range(j1 + 1, self.JackNjk):
                bias_jk2[(j1 + 1, j2 + 1)] = _bias_jk2[i]
                i += 1
        return bias_jk2

    def get_delete2_fsky(self):
        """
        Returns the fraction of the sky after deleting two regions.
        returns:
            fskyjk2 (np.array): Fraction of the sky after deleting two regions.
        """
        self.fskyjk2 = np.zeros((self.JackNjk, self.JackNjk))
        for j1 in range(0, self.JackNjk):
            for j2 in range(j1 + 1, self.JackNjk):
                cond = np.where(
                    (self.mask == 1.0) & (self.jkmap != j1 + 1) & (self.jkmap != j2 + 1)
                )[0]
                self.fskyjk2[j1, j2] = len(cond) / len(self.mask)
        return self.fskyjk2

    def get_delete1_fsky(self):
        """
        Returns the fraction of the sky after deleting one region.
        returns:
            fskyjk (np.array): Ffraction of the sky after deleting one region.
        """
        self.fskyjk = np.zeros((self.JackNjk))
        for j in range(0, self.JackNjk):
            cond = np.where((self.mask == 1.0) & (self.jkmap != j + 1))[0]
            self.fskyjk[j] = len(cond) / len(self.mask)
        return self.fskyjk

    ###
    # Step 4: correct Cls for mask
    ###

    def _compute_alpha(self, Mljk, Mls0):
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
        #wMljk *= self._logistic(np.log10(abs(wMljk)))
        alpha = wMls0 / wMljk
        return alpha

    def _correct_data_cls_mask(self, Cljk, Mljk, Mls0):
        """
        Private method to correct the fact that when a Jackknife region is removed,
        the mask of the region changes, which affects the Cls.
        returns:
            Cljk_corr (dict): Corrected Cls
        """
        corr_Cljk = {}
        for i in range(0, len(self.Cl_keys)):
            # get alpha
            _Mls0 = Mls0[self.Clmm_keys[i]]
            _Mljk = Mljk[self.Clmm_keys[i]]
            alpha = self._compute_alpha(_Mljk, _Mls0)
            cl_key = self.Cl_keys[i]
            _Cljk = np.atleast_2d(Cljk[cl_key])
            # Correct Cl by mask
            ncls, nells = _Cljk.shape
            k1, k2, b1, b2 = cl_key
            if k1 == k2 == "SHE":
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
                    _corr_Cljk = np.array(
                        [
                            __corr_Cljk[1],  # EE like spin-2
                            __corr_Cljk[2],  # BB like spin-2
                            __icorr_Cljk[1],  # EB like spin-0
                        ]
                    )
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
                    _corr_Cljk = np.array(
                        [
                            __corr_Cljk[1],  # EE like spin-2
                            __corr_Cljk[2],  # BB like spin-2
                            __icorr_Cljk[1],  # EB like spin-0
                            -__icorr_Cljk[2],  # BE like spin-0
                        ]
                    )
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
                _corr_Cljk = np.array(_corr_Cljk)

            if len(_corr_Cljk) == 1:
                _corr_Cljk = _corr_Cljk[0]
            corr_Cljk[self.Cl_keys[i]] = _corr_Cljk
        return corr_Cljk

    def _logistic(self, x, x0=-2, k=50):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    ###
    # Step 5: delete1 Covariance
    ###

    def get_delete1_cov(self, mask_correction=True, shrink=True):
        """
        Checks if the delete1 covariance exists.
        If not, load it from file. If it does not exist, compute it.
        inputs:
            bias_correction (bool): If True, apply bias correction
            mask_correction (bool): If True, apply mask correction
        returns:
            delete1_cov (dict): Dictionary of delete1 covariance
        """
        if self.delete1_cov is None:
            fname_target = self.output_path + "cov/target_cov_%i.fits" % (self.JackNjk)
            fname_cov = self.output_path + "cov/jackcov_delete1_njk_%i" % (self.JackNjk)
            if mask_correction:
                fname_cov += "_cmask"
            fname_cov += ".fits"
            if os.path.exists(fname_cov) and os.path.exists(fname_target):
                print(f"Delete1 covariance exists: {fname_cov}")
                self.delete1_cov = read(fname_cov)
                print(f"Target covariance exists: {fname_target}")
                self.target_cov = read(fname_target)
            else:
                Clsjks, _ = self.get_delete1_cls(mask_correction=mask_correction)
                self.delete1_cov, self.target_cov = self._get_delete1_cov(
                    Clsjks, shrink=shrink
                )
                if self.save:
                    write(fname_cov, self.delete1_cov)
                    write(fname_target, self.target_cov)
        return self.delete1_cov

    def _get_delete1_cov(self, Clsjks, shrink=True):
        """
        Internal method to compute the shrunk covariance.
        inputs:
            Clsjks (dict): Dictionary of delete1 data Cls
            shrink (bool): If True, apply shrinkage
        returns:
            shrunk_cov (dict): Dictionary of shrunk covariance
            target_cov (dict): Dictionary of target covariance
        """
        # Bin Cls
        Cqsjks = {}
        for key in Clsjks.keys():
            Cqsjks[key] = binned(Clsjks[key], self.ledges)

        # From list of Dicts to Dict of lists
        _Cqsjks = {}
        for key in self.Cl_keys:
            cqs = []
            for i in range(1, self.JackNjk + 1):
                Cqsjk = Cqsjks[i][key]
                cqs.append(Cqsjk)
            _Cqsjks[key] = np.array(cqs)

        # Separate component Cls
        _Cqsjks = compsep_Cls(_Cqsjks)
        compsep_covkeys = get_covkeys(list(_Cqsjks.keys()))

        # Mean of binned delete1 Cls
        Cqsjks_bar = {}
        for key in list(_Cqsjks.keys()):
            Cqsjks_bar[key] = np.mean(_Cqsjks[key], axis=0)

        # Gaussian Covariance Expectation
        ClGauss_cov = get_T_new(Cqsjks_bar, compsep_covkeys)
        ClGauss_corr = {}
        for key in ClGauss_cov.keys():
            cov = ClGauss_cov[key]
            posdef_cov = make_posdef(cov)
            corr = cov2corr(posdef_cov)
            ClGauss_corr[key] = corr

        # Gaussian Correlation Expectation
        Ws = {}
        Ws_bar = {}
        for i, key1 in enumerate(list(_Cqsjks.keys())):
            for j, key2 in enumerate(list(_Cqsjks.keys())):
                if i <= j:
                    clsjk1 = _Cqsjks[key1]
                    clsjk_bar1 = Cqsjks_bar[key1]
                    clsjk2 = _Cqsjks[key2]
                    clsjk_bar2 = Cqsjks_bar[key2]
                    W = get_W(clsjk1, clsjk_bar1, clsjk2, clsjk_bar2, jk=True)
                    cov_key = (
                        key1[0],
                        key1[1],
                        key2[0],
                        key2[1],
                        key1[2],
                        key1[3],
                        key2[2],
                        key2[3],
                    )
                    Ws[cov_key] = np.array(W)
                    Ws_bar[cov_key] = np.mean(W, axis=0)

        # Compute Jackknife covariance
        Ss = {}
        for key in Ws_bar.keys():
            S = (self.JackNjk / (self.JackNjk - 1)) * Ws_bar[key]
            Ss[key] = make_posdef(S)

        # Compute target matrix
        Ts_rbar = {}
        for key in Ss.keys():
            S = Ss[key]
            rbar = ClGauss_corr[key]
            Ts_rbar[key] = get_T_rbar(S, rbar)

        # Compute scalar shrinkage intensity
        self.lambda_star = 0
        if shrink:
            lambda_star_numenator = 0
            lambda_star_denominator = 0
            for key in Ss.keys():
                if (key[0], key[1]) == (key[2], key[3]):
                    S = Ss[key]
                    W = Ws[key]
                    Wbar = Ws_bar[key]
                    rbar = ClGauss_corr[key]
                    print(key)
                    l_n, l_d = get_lambda_star_single_rbar(S, W, Wbar, rbar)
                    ls = l_n / l_d
                    if math.isnan(ls):
                        print("Failure! - Shrinkage intensity Lambda = %0.4f" % ls)
                    else:
                        lambda_star_numenator += l_n
                        lambda_star_denominator += l_d
                        print("Success! - Shrinkage intensity Lambda = %0.4f" % ls)
                    self.lambda_star = lambda_star_numenator / lambda_star_denominator
            print("Final shrinkage intensity Lambda = %0.4f" % self.lambda_star)

        # Shrink covariance
        shrunk_covs = {}
        for key in Ts_rbar.keys():
            T_rbar = Ts_rbar[key]
            S = Ss[key]
            shrunk_cov = self.lambda_star * T_rbar + (1 - self.lambda_star) * S
            shrunk_covs[key] = shrunk_cov

        return shrunk_covs, Ts_rbar

    def get_shrinkage_factors(self):
        """
        Returns the shrinkage factors for each Cl.
        returns:
            lambda_stars (dict): Dictionary of shrinkage factors
        """
        if self.lambda_star is None:
            self.get_delete1_cov()
        return self.lambda_star

    def get_target_cov(self):
        """
        Returns the target covariance matrix.
        returns:
            target_cov (dict): Dictionary of target covariance
        """
        if self.target_cov is None:
            self.get_dices_cov()
        return self.target_cov

    ###
    # Step 6: Delete2 Correction
    ###

    def get_delete2_cov(self, mask_correction=True):
        """
        Checks if the delete 2 covariance exists.
        If not, load it from file. If it does not exist, compute it.
        inputs:
            mask_correction (bool): If True, apply mask correction
        returns:
            delete2_cov (dict): Dictionary of delete2 covariance
        """
        if self.delete2_cov is None:
            fname = self.output_path + "cov/jackcov_delete2_njk_%i" % (self.JackNjk)
            if mask_correction:
                fname += "_cmask"
            fname += ".fits"
            if os.path.exists(fname):
                print(f"Delete2 covariance exists: {fname}")
                self.delete2_cov = read(fname)
            else:
                Cls0 = self.get_data_cls()
                # Load Jackknife delete-2 Samples
                Clsjks, _ = self.get_delete1_cls(mask_correction=mask_correction)
                Clsjk2s, _ = self.get_delete2_cls(mask_correction=mask_correction)
                self.delete2_cov = self._get_delete2_cov(
                    Cls0,
                    Clsjks,
                    Clsjk2s,
                )
                if self.save:
                    write(fname, self.delete2_cov)
        return self.delete2_cov

    def _get_delete2_cov(self, Cls0, Clsjks, Clsjk2s):
        """
        Internal method to compute the delete2 covariance.
        inputs:
            Cls0 (dict): Dictionary of data Cls
            Clsjks (dict): Dictionary of delete1 data Cls
            Clsjk2s (dict): Dictionary of delete2 data Cls
        returns:
            Cljk_cov (dict): Dictionary of delete2 covariance
        """
        # Bin Cls
        Cqs0 = compsep_Cls(binned(Cls0, self.ledges))
        Cqsjks = []
        for key in Clsjks.keys():
            cqs = binned(Clsjks[key], self.ledges)
            Cqsjks.append(compsep_Cls(cqs))

        jk1 = []
        jk2 = []
        Cqsjks2 = []
        for jk in range(1, self.JackNjk):
            _jk2 = np.arange(jk + 1, self.JackNjk + 1)
            _jk1 = jk * np.ones(len(_jk2))
            _jk1 = _jk1.astype("int")
            _jk2 = _jk2.astype("int")
            _Clsjks = []
            for __jk2 in _jk2:
                cqs = binned(Clsjk2s[(jk, __jk2)], self.ledges)
                _Clsjks.append(compsep_Cls(cqs))
            jk1.append(_jk1)
            jk2.append(_jk2)
            [Cqsjks2.append(_Cls) for _Cls in _Clsjks]
        jk1 = np.concatenate(jk1)
        jk2 = np.concatenate(jk2)

        # Compute bias correction
        Qii = []
        for i in range(0, len(Cqsjks2)):
            i1 = jk1[i]
            i2 = jk2[i]
            _Qii = {}
            for key in list(Cqs0.keys()):
                __Qii = self.JackNjk * Cqs0[key]
                __Qii -= (self.JackNjk - 1) * (
                    Cqsjks[i1 - 1][key] + Cqsjks[i2 - 1][key]
                )
                __Qii += (self.JackNjk - 2) * Cqsjks2[i][key]
                _Qii[key] = __Qii
            Qii.append(_Qii)

        n = self.JackNjk * (self.JackNjk - 1) / 2
        Q_cov = get_Cl_cov(Qii)
        for key in Q_cov.keys():
            Q_cov[key] *= n - 1
            Q_cov[key] *= 1 / (self.JackNjk * (self.JackNjk + 1))
        return Q_cov

    def _get_delete2_correction(self, cov1, cov2):
        """
        Internal method to apply the delete-2 correction
        to compute the Dices covariance.
        inputs:
            cov1 (dict): Dictionary of delete1 covariance
            cov2 (dict): Dictionary of delete2 covariance
        returns:
            dices_cov (dict): Dictionary of Dices covariance
        """
        self.dices_cov = {}
        for key in list(cov1.keys()):
            _cov1 = cov1[key]
            _cov2 = cov1[key] - cov2[key]
            _corr1 = cov2corr(_cov1)
            _var1 = np.diag(_cov1).copy()
            _var2 = np.diag(_cov2).copy()
            cond = np.where(_var2 < 0)[0]
            _var2[cond] = _var1[cond]
            _sig2 = np.sqrt(_var2)
            _corr2 = np.outer(_sig2, _sig2)
            dices_cov = _corr2 * _corr1
            self.dices_cov[key] = dices_cov
        return self.dices_cov
