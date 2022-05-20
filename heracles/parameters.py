"""
A Python module for creating Healpix number and ellipticity maps
from an input catalogue. Includes map randomization code to be used when
estimating pure noise power spectra.

This module interfaces with PClEst.

Copyright 2018 Lee Whittaker
"""

import os
import configparser
from typing import NamedTuple, Sequence, Mapping

import numpy as np
import fitsio


class NoSpecRequested(Exception):
    """
    Exception class for checking that the number of redshift bins in the mask match the number of bins for
    which the Cls are estimated
    """
    pass


class MaskWrongBins(Exception):
    """
    Exception class for checking that the number of redshift bins in the mask match the number of bins for
    which the Cls are estimated
    """
    pass


class WrongBinningScheme(Exception):
    """
    Exception class for checking that the Nside of the input mask matches that of the Healpix maps generated.
    """
    pass


class Params(NamedTuple):
    '''data structure holding all input parameters'''

    # general parameters
    nside: int = 16
    out_value: int = 1
    calc_pp: bool = True
    calc_pe: bool = True
    calc_ee: bool = True
    nlsamp: int = 4
    keep_alms: bool = False
    nsplitmaps: int = 0
    nbar_cut: float = 0.8
    shear_coord: int = 1
    flask_noise_only: bool = False
    out_masks: bool = False
    out_effect: bool = False
    seed: int = None
    nthreads: int = None

    # redshift binning parameters
    zselect: Mapping[int, str] = None

    # angular binning parameters
    lmin: float = 2
    lmax: float = 16
    linlogspace: int = None
    lbins: Sequence[float] = None
    nell_bins: int = None

    # catalogue parameters
    cat_name: str = None
    cat_name_pos: str = None

    # visibility mask parameters
    mask_name: str = None
    extra_mask_name: str = None

    # catalogue column name parameters
    x_name: str = 'RIGHT_ASCENSION'
    y_name: str = 'DECLINATION'
    g1_name: str = 'G1'
    g2_name: str = 'G2'
    weight_name: str = 'WEIGHT'
    z_name: str = 'Z'
    bin_id_name: str = None

    # parameter to determine how many splits of the catalogue to make when reading in chains
    nsplit_cat: int = 1000

    # spherical harmonic transform parameters
    use_pixel_weights: bool = True

    # mixing matrix parameters
    lmax_mm: int = None
    l3max: int = None
    mask_bins: bool = False
    mask_pw: bool = False

    # jackknife parameters
    nsideKmeans: int = 128
    nClustersMax: int = 50
    nClustersMin: int = 22
    areaTol: float = 2.5

    # paths
    workdir: str = None


def params_from_ini(filename):
    '''read input parameters from a config file in INI format'''

    # parse the file
    config = configparser.ConfigParser()
    config.read(filename)

    # read options from config, with defaults from params
    options = {}
    options['nside'] = config.getint('MAIN', 'Nside', fallback=None)
    options['lmin'] = config.getfloat('MAIN', 'Lmin', fallback=None)
    options['lmax'] = config.getfloat('MAIN', 'Lmax', fallback=None)
    options['nell_bins'] = config.getint('MAIN', 'NellBin', fallback=None)
    options['linlogspace'] = config.getint('MAIN', 'LinLogSpace', fallback=None)
    options['out_value'] = config.getint('MAIN', 'OutValue', fallback=None)
    options['calc_pp'] = config.getboolean('MAIN', 'PosPos', fallback=None)
    options['calc_pe'] = config.getboolean('MAIN', 'PosShear', fallback=None)
    options['calc_ee'] = config.getboolean('MAIN', 'ShearShear', fallback=None)
    options['zselect'] = config.get('MAIN', 'ZSelect', fallback=None)
    options['nlsamp'] = config.getint('MAIN', 'NlSamp', fallback=None)
    options['keep_alms'] = config.getboolean('MAIN', 'KeepAlms', fallback=None)
    options['nsplitmaps'] = config.getint('MAIN', 'NsplitMaps', fallback=None)
    options['nbar_cut'] = config.getfloat('MAIN', 'NbarCut', fallback=None)
    options['shear_coord'] = config.getint('MAIN', 'ShearCoord', fallback=None)
    options['flask_noise_only'] = config.getboolean('MAIN', 'FlaskNoiseOnly', fallback=None)
    options['out_masks'] = config.getboolean('MAIN', 'OutMasks', fallback=None)
    options['out_effect'] = config.getboolean('MAIN', 'OutEffect', fallback=None)
    options['seed'] = config.getint('MAIN', 'Seed', fallback=None)
    options['nthreads'] = config.getint('MAIN', 'NThreads', fallback=None)

    # catalogue column name parameters
    options['x_name'] = config.get('MAIN', 'XName', fallback=None)
    options['y_name'] = config.get('MAIN', 'YName', fallback=None)
    options['g1_name'] = config.get('MAIN', 'G1Name', fallback=None)
    options['g2_name'] = config.get('MAIN', 'G2Name', fallback=None)
    options['weight_name'] = config.get('MAIN', 'WeightName', fallback=None)
    options['z_name'] = config.get('MAIN', 'ZName', fallback=None)
    options['bin_id_name'] = config.get('MAIN', 'BinIDName', fallback=None)

    # mixing matrix parameters
    options['lmax_mm'] = config.getint('MIXMAT', 'LmaxMm', fallback=None)
    options['l3max'] = config.getint('MIXMAT', 'L3max', fallback=None)
    options['mask_bins'] = config.getboolean('MIXMAT', 'MaskBins', fallback=None)
    options['mask_pw'] = config.getboolean('MIXMAT', 'MaskPixWin', fallback=None)

    # jackknife parameters
    options['nsideKmeans'] = config.getint('JK', 'nsideKmeans', fallback=None)
    options['nClustersMax'] = config.getint('JK', 'nClustersMax', fallback=None)
    options['nClustersMin'] = config.getint('JK', 'nClustersMin', fallback=None)
    options['areaTol'] = config.getfloat('JK', 'areaTol', fallback=None)

    # paths
    options['cat_name'] = config.get('MAIN', 'CatName', fallback=None)
    options['cat_name_pos'] = config.get('MAIN', 'PosCatName', fallback=None)
    options['workdir'] = config.get('MAIN', 'OutDir', fallback=None)
    options['mask_name'] = config.get('MAIN', 'MaskName', fallback=None)

    # filter out None (i.e. missing) values
    options = {key: value for key, value in options.items() if value is not None}

    # turn options into params
    return make_params(options)


def make_params(options, test=False, verbose=False):
    '''create a params structure from a dictionary of options'''

    # copy the options dict for modifications
    options = options.copy()

    # create the multipole bins
    create_lbins(options)

    # create the (immutable!) params tuple from the options dict
    params = Params(**options)

    # test requested spectra
    test_spec_request(params)

    if not test:
        # test input catalogue exists
        if params.cat_name:
            test_incat(params.cat_name)

        # test input position catalogue exists
        if params.cat_name_pos:
            test_incat(params.cat_name_pos)

        # check mask
        if params.mask_name:
            check_mask(params.mask_name, params.mask_bins, max(list(params.zselect.keys())), params.nside)

        # test output directory exists
        if params.workdir:
            test_workdir(params)

    # print user options
    if verbose:
        print_options(params)

    # return the params
    return params


def create_lbins(options):
    """Creates the multipole bins"""
    options['lmin'] = np.math.ceil(options.get('lmin', Params._field_defaults['lmin']))
    options['lmax'] = int(options.get('lmax', Params._field_defaults['lmax']))
    if not options.get('nell_bins'):
        options['lbins'] = None
    else:
        if options['linlogspace'] == 0:
            options['lbins'] = np.linspace(options['lmin'], options['lmax'], num=options['nell_bins']+1)
        elif options['linlogspace'] == 1:
            options['lbins'] = np.geomspace(options['lmin'], options['lmax'], num=options['nell_bins']+1)
        else:
            raise WrongBinningScheme('\nYour binning scheme is not recognised')


def test_incat(cat_name):
    """
    Check whether the input catalogue exists, raise exception if not
    """

    if os.path.isfile(cat_name):
        print('\nUsing catalogue ' + cat_name)
    else:
        raise FileNotFoundError('\nThe input catalogue does not exist')


def test_workdir(params):
    """
    Check whether the output directory exists, raise exception if not
    """

    if os.path.isdir(params.workdir):
        print('\nPutting all output in ' + params.workdir)
    else:
        print('\nOutput directory not found, trying to create it...')
        try:
            os.mkdir(params.workdir)
        except FileExistsError:
            raise FileNotFoundError('\nThe output directory does not exist')


def test_spec_request(params):
    """Check that spectra have been requested"""

    test = False
    print('\nYou have requested:')
    if params.calc_pp:
        test = True
        print('Angular clustering')
    if params.calc_pe:
        test = True
        print('Galaxy-galaxy lensing')
    if params.calc_ee:
        test = True
        print('Cosmic shear')
    if not test:
        raise NoSpecRequested('\nNo power spectra have been requested')


def check_mask(mask_name, mask_bins, nbins, nside):
    """Function to read in the mask"""

    # If mask file doesn't exist assume full-sky.
    # If mask file does exist use either constant or z-dependent mask as required.
    # Exception is raised if if the number of z-bins for the z-dependent case
    # doesn't match the number of z-bins for the Cls.
    if os.path.isfile(mask_name):
        print('\nReading in mask from ' + mask_name)
        h = fitsio.read_header(mask_name, ext=1)
        if not mask_bins:
            print('\nAssuming mask is constant for all z-bins')
        else:
            print('\nUsing z-dependent mask')
            if h['TFIELDS'] < nbins:
                raise MaskWrongBins('\nNot enough z-bins found in mask file!')
    else:
        print('\nNo mask provided')


def print_options(params):
    """
    Prints the user options
    """

    print('\nCreating HEALPix maps with NSIDE = ' + str(params.nside))
    print('\nCalculating Cls to lmin = ' + str(params.lmin))
    print('\nCalculating Cls to lmax = ' + str(params.lmax))
    print('\nNumber of Nl samples requested = ' + str(params.nlsamp))
    print('\nz-bins to be analysed:')
    for i, (ibin, z) in enumerate(params.zselect.items()):
        print('\t%s: %s, %s' % (ibin, z[0], z[1]))
    if params.nlsamp:
        print('Random seed used for this run = ' + str(params.seed))
