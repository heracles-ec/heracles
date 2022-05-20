import unittest
import configparser
import tempfile
import os
import numpy.testing as npt
import numpy as np

from le3_pk_wl.parameters import (
    Params,
    make_params,
    params_from_ini,
    NoSpecRequested,
    WrongBinningScheme,
)


class TestParameters(unittest.TestCase):

    def test_make_params(self):
        # set up an options dictionary
        options = {
            'nside': 128,
            'lmax': 100,
            'calc_pp': True,
            'calc_pe': False,
            'calc_ee': True,
        }

        # make params from options dict
        params = make_params(options)

        # check that given values match
        for par, value in options.items():
            with self.subTest(par=par, value=value):
                npt.assert_equal(getattr(params, par), value)

    def test_print_options(self):

        options = {
            'nside': 128,
            'lmin': 10,
            'lmax': 100,
            'nlsamp': 5,
            'zselect': {0: (0., 0.8), 1: (1.0, 1.2)},
            'seed': 999
        }

        params = make_params(options, verbose=True)
        assert params.seed == 999

    def test_create_lbins(self):

        options = {
            'lmin': 2,
            'lmax': 100
        }
        params = make_params(options)

        assert params.lmin == np.math.ceil(options['lmin'])
        assert params.lmax == int(options['lmax'])
        assert params.nell_bins is None
        assert params.lbins is None

    def test_create_lbins_binlinear(self):

        options = {
            'lmin': 2,
            'lmax': 100,
            'nell_bins': 10,
            'linlogspace': 0
        }
        params = make_params(options)
        lbins = np.linspace(options['lmin'], options['lmax'], num=options['nell_bins']+1)
        nell_bins = options['nell_bins']

        assert params.lmin == np.math.ceil(options['lmin'])
        assert params.lmax == int(options['lmax'])
        assert (params.lbins == lbins).all()
        assert params.nell_bins == nell_bins

    def test_create_lbins_binlog(self):

        options = {
            'lmin': 2,
            'lmax': 100,
            'nell_bins': 10,
            'linlogspace': 1
        }
        params = make_params(options)
        lbins = np.geomspace(options['lmin'], options['lmax'], num=options['nell_bins']+1)
        nell_bins = len(lbins) - 1
        assert params.lmin == np.math.ceil(options['lmin'])
        assert params.lmax == int(options['lmax'])
        assert (params.lbins == lbins).all()
        assert params.nell_bins == nell_bins

    def test_params_from_ini(self):
        # create a ConfigParser instance for testing
        config = configparser.ConfigParser()

        # set test options
        config['MAIN'] = {
            'Nside': 128,
            'Lmax': 100,
            'PosPos': True,
            'PosShear': False,
            'ShearShear': True,
        }

        # create a temporary ini file for testing
        with tempfile.NamedTemporaryFile(mode='w') as f:

            # write the config to ini file
            config.write(f)

            # rewind the ini file
            f.seek(0)

            # read the ini using our function
            params = params_from_ini(f.name)

        # check values that were set
        self.assertEqual(params.nside, config.getint('MAIN', 'Nside'))
        self.assertEqual(params.lmax, config.getfloat('MAIN', 'Lmax'))
        self.assertEqual(params.calc_pp, config.getboolean('MAIN', 'PosPos'))
        self.assertEqual(params.calc_pe, config.getboolean('MAIN', 'PosShear'))
        self.assertEqual(params.calc_ee, config.getboolean('MAIN', 'ShearShear'))

        # check default values were not overwritten
        for par, default_value in Params._field_defaults.items():
            with self.subTest(par=par, default_value=default_value):
                if par not in ['nside', 'lmax', 'calc_pp', 'calc_pe', 'calc_ee'] \
                        and default_value is not None:
                    npt.assert_equal(getattr(params, par), default_value)

    def test_existing_incat(self):
        # set up options with an existing input catalogue (tempfile)
        with tempfile.NamedTemporaryFile(mode='w') as f:
            options = {
                'cat_name': f.name,
            }

            params = make_params(options)

            self.assertEqual(params.cat_name, f.name)

    def test_missing_incat(self):
        # set up options with a missing input catalogue
        options = {
            'cat_name': 'missing.fits',
        }

        with self.assertRaises(FileNotFoundError):
            make_params(options)

    def test_existing_workdir(self):
        # set up options with an existing workdir (tempfile)
        with tempfile.TemporaryDirectory() as workdir:
            options = {
                'workdir': workdir,
            }

            params = make_params(options)

            self.assertEqual(params.workdir, workdir)

    def test_create_workdir(self):
        # make a temporary directory, then delete it, but keep the name
        workdir = tempfile.TemporaryDirectory()
        workdir = str(workdir.name)

        # workdir should have been deleted
        self.assertFalse(os.path.exists(workdir))

        # set up options with a creatable workdir
        options = {
            'workdir': workdir,
        }

        params = make_params(options)

        self.assertEqual(params.workdir, workdir)
        self.assertTrue(os.path.isdir(workdir))

        # delete the temporary workdir
        os.rmdir(workdir)

    def test_bad_workdir(self):
        # set up options with a bad workdir (tempfile)
        with tempfile.NamedTemporaryFile() as f:
            options = {
                'workdir': f.name,
            }

            with self.assertRaises(FileNotFoundError):
                make_params(options)

    def test_bad_spec_request(self):
        # make options that request no spectra
        options = {
            'calc_pp': False,
            'calc_pe': False,
            'calc_ee': False,
        }

        with self.assertRaises(NoSpecRequested):
            make_params(options)

    def test_bad_binning_scheme(self):
        # make options that request unknown binning scheme
        options = {
            'nell_bins': 10,
            'linlogspace': -1,
        }

        with self.assertRaises(WrongBinningScheme):
            make_params(options)


if __name__ == '__main__':
    unittest.main()
