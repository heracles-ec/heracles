import numpy as np
import heracles


def test_transforms(cls0):
    corr = heracles.cl2corr(cls0)
    for key in list(cls0.keys()):
        assert corr[key].shape == cls0[key].shape

    _cls = heracles.corr2cl(corr)
    for key in list(cls0.keys()):
        assert _cls[key].shape == cls0[key].shape
        assert np.isclose(cls0[key].array, _cls[key].array).all()


def test_cl2cosebis(cls0):
    key = next(key for key, value in cls0.items() if value.spin == (2, 2))
    ell = np.arange(cls0[key].shape[-1], dtype=np.float64)

    wn = np.vstack(
        [
            np.exp(-0.5 * ((ell - 2.0) / 1.2) ** 2),
            np.exp(-0.5 * ((ell - 4.0) / 1.2) ** 2),
            np.exp(-0.5 * ((ell - 6.0) / 1.2) ** 2),
        ]
    )

    cosebis = heracles.cl2cosebis(cls0, wn, wn_ell=ell)

    for k in cls0:
        assert cosebis[k].shape[:-1] == cls0[k].shape[:-1]
        assert cosebis[k].shape[-1] == 3

    expected = np.trapz(
        cls0[key].array[0, 0][np.newaxis, :] * (ell[np.newaxis, :] * wn) / (2 * np.pi),
        x=ell,
        axis=-1,
    )
    assert np.allclose(cosebis[key].array[0, 0], expected)


def test_cl2cosebis_cosmosis_adapter(cls0):
    class FakeDataBlock:
        def __init__(self):
            self._data = {}

        def __setitem__(self, key, value):
            self._data[key] = value

        def __getitem__(self, key):
            return self._data[key]

    class FakeModule:
        setup_options = None

        def __init__(self, name, path):
            self.name = name
            self.path = path

        def setup(self, options):
            FakeModule.setup_options = options
            return {"ok": True}

        def execute(self, block, config=None):
            input_section = FakeModule.setup_options[("options", "input_section_name")]
            output_section = FakeModule.setup_options[("options", "output_section_name")]
            n_modes = FakeModule.setup_options[("options", "n_max")]
            output_section_b = FakeModule.setup_options._data.get(
                ("options", "output_section_name_b")
            )

            for (section, name), value in list(block._data.items()):
                if section != input_section or not name.startswith("bin_"):
                    continue
                base = np.sum(value)
                block[output_section, name] = np.arange(1, n_modes + 1) * base
                if output_section_b is not None:
                    block[output_section_b, name] = -np.arange(1, n_modes + 1) * base
            return 0

        def cleanup(self, config):
            return None

    key = next(key for key, value in cls0.items() if value.spin == (2, 2))
    subset = {key: cls0[key]}
    e_modes, b_modes = heracles.transforms.cl2cosebis_cosmosis(
        subset,
        module_path="/tmp/fake_cl_to_cosebis.so",
        theta_min=1.0,
        theta_max=300.0,
        n_modes=4,
        output_section_b="cosebis_b",
        _bindings=(FakeDataBlock, "options", FakeModule),
    )

    assert list(e_modes.keys()) == [key]
    assert list(b_modes.keys()) == [key]
    assert e_modes[key].shape == (4,)
    assert b_modes[key].shape == (4,)
    assert np.array_equal(e_modes[key].ell, np.arange(1, 5))
    assert np.allclose(b_modes[key].array, -e_modes[key].array)
