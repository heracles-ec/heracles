import os
import healpy as hp
from ..result import binned
from ..io import write, read


from .cls import (
    get_cls,
    get_delete_cls,
)

from .utils_cl import (
    get_lgrid,
    cov2spinblocks,
)

from .bias_corrrection import (
    correct_bias,
)

from .mask_correction import (
    correct_mask,
)

from .delete1 import get_delete1_cov

from .delete2 import (
    get_delete2_correction,
    get_delete2_cov,
)

from .dices import (
    get_dices_cov,
)


class DICES:
    def __init__(self, data_maps, jkmaps, vis_maps, config):
        self._load_defaults(config)
        self.data_maps = data_maps
        self.vis_maps = vis_maps
        self.jkmaps = jkmaps

        # Step 1 Make Cls
        self.delete1_data_cls = None
        self.delete2_data_cls = None
        self.delete1_mask_cls = None
        self.delete2_mask_cls = None

        # Step 2 Make delte1 covariance
        self.lambda_star = 0
        self.gaussian_cov = None
        self.delete1_cov = None
        self.target_cov = None

        # Step 3 Apply delete2 correction
        self.delete2_cov = None

        # Step 4 Get Dices Covariance
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

        # Cl bins
        self.lmin = int(config["bins"]["Lmin"])
        self.lmax = int(config["bins"]["Lmax"])
        self.lbins = int(config["bins"]["Lbin"])
        self.logbins = bool(config["bins"]["Logbins"])
        self.lgrid, self.ledges, self.dl = get_lgrid(
            self.lmin, self.lmax, self.lbins, uselog=self.logbins
        )

        # DICES
        self.bin = bool(config["DICES"]["bin"])
        self.mask_correction = bool(config["DICES"]["mask_correction"])
        self.shrinkage = bool(config["DICES"]["shrinkage"])
        self.delete2_correction = bool(config["DICES"]["delete2_correction"])

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
            self.data_cls = read(data_fname)
            self.mask_cls = read(mask_fname)
        else:
            self.data_cls, self.mask_cls = get_cls(
                self.data_maps,
                self.vis_maps,
            )
            if self.save:
                write(data_fname, self.data_cls)
                write(mask_fname, self.mask_cls)
        return self.data_cls, self.mask_cls

    def get_delete1_cls(self):
        """
        Check if the Cls of removing 1 Jackknife region exist, if not, load them.
        If files do not exist, compute them.
        returns:
            delete1_data_cls (dict): Dictionary of delete1 data Cls
            delete1_data_cls_wbias (dict): Dictionary of delete1 data Cls with bias
            delete1_mask_cls (dict): Dictionary of delete1 mask Cls
        """
        if (self.delete1_data_cls is None) or (self.delete1_mask_cls is None):
            self.delete1_data_cls = {}
            self.delete1_mask_cls = {}
            for jk in range(1, self.JackNjk + 1):
                data_fname = self.output_path + "cls/cls_njk_%i_jkid_%i_%i.fits" % (
                    self.JackNjk,
                    jk,
                    jk,
                )
                mask_fname = self.output_path + "cls/mls_njk_%i_jkid_%i_%i.fits" % (
                    self.JackNjk,
                    jk,
                    jk,
                )

                if os.path.exists(data_fname) & os.path.exists(mask_fname):
                    self.delete1_data_cls[jk] = read(data_fname)
                    self.delete1_mask_cls[jk] = read(mask_fname)
                else:
                    # Compute Cls
                    cls, cls_mm = get_delete_cls(
                        self.data_maps,
                        self.vis_maps,
                        self.jkmaps,
                        jk,
                        jk,
                    )

                    # Mask correction
                    if self.mask_correction:
                        _, mls0 = get_cls(self.data_maps, self.vis_maps)
                        cls_cbias = correct_mask(
                            cls, cls_mm, mls0
                        )

                    # Bias correction
                    cls_cbias = correct_bias(
                        cls,
                        self.jkmaps,
                        jk,
                        jk,
                        )

                    self.delete1_data_cls[(jk, jk)] = cls_cbias
                    self.delete1_mask_cls[(jk, jk)] = cls_mm
                    # Save Cls
                    if self.save:
                        write(data_fname, cls_cbias)
                        write(mask_fname, cls_mm)
        return (
            self.delete1_data_cls,
            self.delete1_mask_cls,
        )

    def get_delete2_cls(self):
        """
        Check if the Cls of removing 2 Jackknife regions exist, if not, load them.
        If files do not exist, compute them.
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
                        self.delete2_data_cls[(jk, jk2)] = read(data_fname)
                        self.delete2_mask_cls[(jk, jk2)] = read(mask_fname)
                    else:
                        # Compute Cls
                        cls, cls_mm = get_delete_cls(
                            self.data_maps,
                            self.vis_maps,
                            self.jkmaps,
                            jk,
                            jk2,
                        )

                        # Mask correction
                        if self.mask_correction:
                            _, mls0 = get_cls(self.data_maps, self.vis_maps)
                            cls_cbias = correct_mask(
                                cls, cls_mm, mls0
                            )

                        # Bias correction
                        cls_cbias = correct_bias(
                            cls,
                            self.jkmaps,
                            jk,
                            jk2,
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


    ###
    # Step 2: delete1 Covariance
    ###
    def get_delete1_cov(self):
        """
        Checks if the delete1 covariance exists.
        If not, load it from file. If it does not exist, compute it.
        inputs:
            bin (bool): If True, use the binned Cls
        returns:
            delete1_cov (dict): Dictionary of delete1 covariance
        """
        if self.delete1_cov is None:
            fname_target = self.output_path + "cov/target_cov_%i.fits" % (self.JackNjk)
            fname_cov = self.output_path + "cov/jackcov_delete1_njk_%i" % (self.JackNjk)
            fname_shrunkcov = self.output_path + "cov/jackcov_shrunk_delete1_njk_%i" % (
                self.JackNjk
            )
            fname_cov += ".fits"
            if os.path.exists(fname_cov) and os.path.exists(fname_target):
                print(f"Delete1 covariance exists: {fname_cov}")
                self.delete1_cov = read(fname_cov)
                print(f"Target covariance exists: {fname_target}")
                self.target_cov = read(fname_target)
            else:
                Cls0, _ = get_cls(self.data_maps, self.vis_maps)
                Clsjks, _ = self.get_delete1_cls()
                if self.bin:
                    Cls0 = binned(Cls0, self.ledges)
                    Clsjks = binned(Clsjks, self.ledges)
                (
                    self.shrunk_delete1_cov,
                    self.delete1_cov,
                    self.target_cov,
                ) = get_delete1_cov(
                    Cls0,
                    Clsjks,
                    shrink=self.shrinkage,
                )
                if self.save:
                    write(fname_shrunkcov, self.shrunk_delete1_cov)
                    write(fname_cov, self.delete1_cov)
                    write(fname_target, self.target_cov)
        return self.shrunk_delete1_cov, self.delete1_cov, self.target_cov

    ###
    # Step 3: Delete2 Correction
    ###
    def get_delete2_cov(self):
        """
        Checks if the delete 2 covariance exists.
        If not, load it from file. If it does not exist, compute it.
        inputs:
            bin (bool): If True, use the bin cls
        returns:
            delete2_cov (dict): Dictionary of delete2 covariance
        """
        if self.delete2_cov is None:
            fname = self.output_path + "cov/jackcov_delete2_njk_%i" % (self.JackNjk)
            fname += ".fits"
            if os.path.exists(fname):
                print(f"Delete2 covariance exists: {fname}")
                self.delete2_cov = read(fname)
            else:
                Cls0, _ = self.get_cls()
                Clsjks, _ = self.get_delete1_cls()
                Clsjk2s, _ = self.get_delete2_cls()
                _, delete1_cov, _ = self.get_delete1_cov()
                if self.bin:
                    Cls0 = binned(Cls0, self.ledges)
                    Clsjks = binned(Clsjks, self.ledges)
                    Clsjk2s = binned(Clsjk2s, self.ledges)
                Q = get_delete2_correction(
                    Cls0,
                    Clsjks,
                    Clsjk2s,
                )
                self.delete2_cov = get_delete2_cov(delete1_cov, Q)
                for key in list(delete1_cov.keys()):
                    self.delete2_cov[key] = delete1_cov[key] - Q[key]
                if self.save:
                    write(fname, self.delete2_cov)
        return self.delete2_cov

    ###
    # Step 4: Make DICES cov
    ###
    def get_dices_cov(self, to_spinblocks=False):
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
                cls0, _ = self.get_cls()
                if self.bin:
                    cls0 = binned(cls0, self.ledges)
                cov1, _, _ = self.get_delete1_cov()
                cov2 = self.get_delete2_cov()
                dices_cov = get_dices_cov(cls0, cov1, cov2)
                if to_spinblocks:
                    dices_cov = cov2spinblocks(cls0, dices_cov)
                self.dices_cov = dices_cov
                if self.save:
                    write(fname, self.dices_cov)
        return self.dices_cov
