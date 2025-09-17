# _Heracles_ — Harmonic-space statistics on the sphere

[![PyPI](https://img.shields.io/pypi/v/heracles)](https://pypi.org/project/heracles)
[![Python](https://img.shields.io/pypi/pyversions/heracles)](https://www.python.org)
[![Documentation](https://readthedocs.org/projects/heracles/badge/?version=latest)](https://heracles.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2408.16903-red)](https://arxiv.org/abs/2408.16903)
[![NASA/ADS](https://img.shields.io/badge/ads-2024arXiv240816903E-blueviolet)](https://ui.adsabs.harvard.edu/abs/2024arXiv240816903E)
[![DOI](https://img.shields.io/badge/doi-10.48550/arXiv.2408.16903-blue)](https://doi.org/10.48550/arXiv.2408.16903)
[![Tests](https://github.com/heracles-ec/heracles/actions/workflows/tests.yml/badge.svg)](https://github.com/heracles-ec/heracles/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/heracles-ec/heracles/graph/badge.svg?token=SYAZYYFLLL)](https://codecov.io/gh/heracles-ec/heracles)

This is _Heracles_, a code for harmonic-space statistics on the sphere.
_Heracles_ can take catalogues of positions a function values on the sphere and
turn them into wonderful things like angular power spectra and mixing matrices.

To get started, read the [documentation](https://heracles.readthedocs.io).

🛰️ **Made in the Euclid Science Ground Segment**

## Installation (latest)

To install the latest released version of the package:

    pip install heracles

You should do this in a dedicated environment (conda, venv, etc.)

## Installation (git main)

To install the latest unreleased version from the git main branch:

    pip install git+https://github.com/heracles-ec/heracles.git

## Installation (develop)

Clone the repository, cd into the local copy, then install in editable mode:

    pip install -e .

# Cite Heracles

- Main paper:

```
@ARTICLE{2025A&A...694A.141E,
       author = {{Euclid Collaboration} and {Tessore}, N. and {Joachimi}, B. and {Loureiro}, A. and {Hall}, A. and {Ca{\~n}as-Herrera}, G. and {Tutusaus}, I. and {Jeffrey}, N. and {Naidoo}, K. and {McEwen}, J.~D. and {Amara}, A. and {Andreon}, S. and {Auricchio}, N. and {Baccigalupi}, C. and {Baldi}, M. and {Bardelli}, S. and {Bernardeau}, F. and {Bonino}, D. and {Branchini}, E. and {Brescia}, M. and {Brinchmann}, J. and {Caillat}, A. and {Camera}, S. and {Capobianco}, V. and {Carbone}, C. and {Cardone}, V.~F. and {Carretero}, J. and {Casas}, S. and {Castellano}, M. and {Castignani}, G. and {Cavuoti}, S. and {Cimatti}, A. and {Colodro-Conde}, C. and {Congedo}, G. and {Conselice}, C.~J. and {Conversi}, L. and {Copin}, Y. and {Courbin}, F. and {Courtois}, H.~M. and {Cropper}, M. and {Da Silva}, A. and {Degaudenzi}, H. and {De Lucia}, G. and {Dinis}, J. and {Dubath}, F. and {Duncan}, C.~A.~J. and {Dupac}, X. and {Dusini}, S. and {Farina}, M. and {Farrens}, S. and {Faustini}, F. and {Ferriol}, S. and {Frailis}, M. and {Franceschi}, E. and {Fumana}, M. and {Galeotta}, S. and {Gillard}, W. and {Gillis}, B. and {Giocoli}, C. and {G{\'o}mez-Alvarez}, P. and {Grazian}, A. and {Grupp}, F. and {Guzzo}, L. and {Haugan}, S.~V.~H. and {Hoekstra}, H. and {Holmes}, W. and {Hormuth}, F. and {Hornstrup}, A. and {Hudelot}, P. and {Jahnke}, K. and {Jhabvala}, M. and {Keih{\"a}nen}, E. and {Kermiche}, S. and {Kiessling}, A. and {Kubik}, B. and {K{\"u}mmel}, M. and {Kunz}, M. and {Kurki-Suonio}, H. and {Ligori}, S. and {Lilje}, P.~B. and {Lindholm}, V. and {Lloro}, I. and {Mainetti}, G. and {Maiorano}, E. and {Mansutti}, O. and {Marggraf}, O. and {Martinelli}, M. and {Martinet}, N. and {Marulli}, F. and {Massey}, R. and {Medinaceli}, E. and {Mei}, S. and {Melchior}, M. and {Mellier}, Y. and {Meneghetti}, M. and {Merlin}, E. and {Meylan}, G. and {Mohr}, J.~J. and {Moresco}, M. and {Morin}, B. and {Moscardini}, L. and {Munari}, E. and {Nakajima}, R. and {Niemi}, S. -M. and {Padilla}, C. and {Paltani}, S. and {Pasian}, F. and {Pedersen}, K. and {Percival}, W.~J. and {Pettorino}, V. and {Pires}, S. and {Polenta}, G. and {Poncet}, M. and {Popa}, L.~A. and {Raison}, F. and {Renzi}, A. and {Rhodes}, J. and {Riccio}, G. and {Romelli}, E. and {Roncarelli}, M. and {Rossetti}, E. and {Saglia}, R. and {Sakr}, Z. and {S{\'a}nchez}, A.~G. and {Sapone}, D. and {Sartoris}, B. and {Schirmer}, M. and {Schneider}, P. and {Schrabback}, T. and {Secroun}, A. and {Seidel}, G. and {Seiffert}, M. and {Serrano}, S. and {Sirignano}, C. and {Sirri}, G. and {Stanco}, L. and {Steinwagner}, J. and {Tallada-Cresp{\'\i}}, P. and {Taylor}, A.~N. and {Tereno}, I. and {Toledo-Moreo}, R. and {Torradeflot}, F. and {Valenziano}, L. and {Vassallo}, T. and {Wang}, Y. and {Weller}, J. and {Zamorani}, G. and {Zucca}, E. and {Biviano}, A. and {Bolzonella}, M. and {Boucaud}, A. and {Bozzo}, E. and {Burigana}, C. and {Calabrese}, M. and {Di Ferdinando}, D. and {Escartin Vigo}, J.~A. and {Finelli}, F. and {Gracia-Carpio}, J. and {Matthew}, S. and {Mauri}, N. and {Pezzotta}, A. and {P{\"o}ntinen}, M. and {Scottez}, V. and {Spurio Mancini}, A. and {Tenti}, M. and {Viel}, M. and {Wiesmann}, M. and {Akrami}, Y. and {Anselmi}, S. and {Archidiacono}, M. and {Atrio-Barandela}, F. and {Balaguera-Antolinez}, A. and {Ballardini}, M. and {Benielli}, D. and {Blanchard}, A. and {Blot}, L. and {B{\"o}hringer}, H. and {Borgani}, S. and {Bruton}, S. and {Cabanac}, R. and {Calabro}, A. and {Camacho Quevedo}, B. and {Cappi}, A. and {Caro}, F. and {Carvalho}, C.~S. and {Castro}, T. and {Chambers}, K.~C. and {Cooray}, A.~R. and {de la Torre}, S. and {Desprez}, G. and {D{\'\i}az-S{\'a}nchez}, A. and {Di Domizio}, S. and {Dole}, H. and {Escoffier}, S. and {Ferrari}, A.~G. and {Ferreira}, P.~G. and {Ferrero}, I. and {Finoguenov}, A. and {Fontana}, A. and {Fornari}, F.},
        title = "{Euclid preparation: LIX. Angular power spectra from discrete observations}",
      journal = {\aap},
     keywords = {gravitational lensing: weak, methods: statistical, surveys, cosmology: observations, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = feb,
       volume = {694},
          eid = {A141},
        pages = {A141},
          doi = {10.1051/0004-6361/202452018},
archivePrefix = {arXiv},
       eprint = {2408.16903},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...694A.141E},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

- Covariance implementation:

```
@ARTICLE{2025arXiv250609118E,
       author = {{Euclid Collaboration} and {Naidoo}, K. and {Ruiz-Zapatero}, J. and {Tessore}, N. and {Joachimi}, B. and {Loureiro}, A. and {Aghanim}, N. and {Altieri}, B. and {Amara}, A. and {Amendola}, L. and {Andreon}, S. and {Auricchio}, N. and {Baccigalupi}, C. and {Bagot}, D. and {Baldi}, M. and {Bardelli}, S. and {Battaglia}, P. and {Biviano}, A. and {Branchini}, E. and {Brescia}, M. and {Camera}, S. and {Capobianco}, V. and {Carbone}, C. and {Cardone}, V.~F. and {Carretero}, J. and {Castellano}, M. and {Castignani}, G. and {Cavuoti}, S. and {Chambers}, K.~C. and {Cimatti}, A. and {Colodro-Conde}, C. and {Congedo}, G. and {Conversi}, L. and {Copin}, Y. and {Courbin}, F. and {Courtois}, H.~M. and {Da Silva}, A. and {Degaudenzi}, H. and {De Lucia}, G. and {Dubath}, F. and {Dupac}, X. and {Dusini}, S. and {Escoffier}, S. and {Farina}, M. and {Farinelli}, R. and {Farrens}, S. and {Faustini}, F. and {Ferriol}, S. and {Finelli}, F. and {Fosalba}, P. and {Frailis}, M. and {Franceschi}, E. and {Fumana}, M. and {Galeotta}, S. and {George}, K. and {Gillis}, B. and {Giocoli}, C. and {Gracia-Carpio}, J. and {Grazian}, A. and {Grupp}, F. and {Holmes}, W. and {Hormuth}, F. and {Hornstrup}, A. and {Jahnke}, K. and {Jhabvala}, M. and {Keih{\"a}nen}, E. and {Kermiche}, S. and {Kiessling}, A. and {Kilbinger}, M. and {Kubik}, B. and {K{\"u}mmel}, M. and {Kunz}, M. and {Kurki-Suonio}, H. and {Le Brun}, A.~M.~C. and {Ligori}, S. and {Lilje}, P.~B. and {Lindholm}, V. and {Lloro}, I. and {Mainetti}, G. and {Maino}, D. and {Maiorano}, E. and {Mansutti}, O. and {Marcin}, S. and {Marggraf}, O. and {Martinelli}, M. and {Martinet}, N. and {Marulli}, F. and {Massey}, R. and {Medinaceli}, E. and {Mei}, S. and {Mellier}, Y. and {Meneghetti}, M. and {Merlin}, E. and {Meylan}, G. and {Mora}, A. and {Moscardini}, L. and {Neissner}, C. and {Niemi}, S. -M. and {Padilla}, C. and {Paltani}, S. and {Pasian}, F. and {Pedersen}, K. and {Percival}, W.~J. and {Pettorino}, V. and {Pires}, S. and {Polenta}, G. and {Poncet}, M. and {Popa}, L.~A. and {Raison}, F. and {Rebolo}, R. and {Renzi}, A. and {Rhodes}, J. and {Riccio}, G. and {Romelli}, E. and {Roncarelli}, M. and {Rosset}, C. and {Saglia}, R. and {Sakr}, Z. and {S{\'a}nchez}, A.~G. and {Sapone}, D. and {Sartoris}, B. and {Schneider}, P. and {Schrabback}, T. and {Secroun}, A. and {Sefusatti}, E. and {Seidel}, G. and {Seiffert}, M. and {Serrano}, S. and {Simon}, P. and {Sirignano}, C. and {Sirri}, G. and {Spurio Mancini}, A. and {Stanco}, L. and {Steinwagner}, J. and {Tallada-Cresp{\'\i}}, P. and {Tavagnacco}, D. and {Taylor}, A.~N. and {Tereno}, I. and {Toft}, S. and {Toledo-Moreo}, R. and {Torradeflot}, F. and {Tutusaus}, I. and {Valenziano}, L. and {Valiviita}, J. and {Vassallo}, T. and {Verdoes Kleijn}, G. and {Veropalumbo}, A. and {Wang}, Y. and {Weller}, J. and {Zamorani}, G. and {Zerbi}, F.~M. and {Zucca}, E. and {Allevato}, V. and {Ballardini}, M. and {Bolzonella}, M. and {Bozzo}, E. and {Burigana}, C. and {Cabanac}, R. and {Calabrese}, M. and {Cappi}, A. and {Di Ferdinando}, D. and {Escartin Vigo}, J.~A. and {Gabarra}, L. and {Mart{\'\i}n-Fleitas}, J. and {Matthew}, S. and {Mauri}, N. and {Metcalf}, R.~B. and {Pezzotta}, A. and {P{\"o}ntinen}, M. and {Risso}, I. and {Scottez}, V. and {Sereno}, M. and {Tenti}, M. and {Viel}, M. and {Wiesmann}, M. and {Akrami}, Y. and {Andika}, I.~T. and {Anselmi}, S. and {Archidiacono}, M. and {Atrio-Barandela}, F. and {Balaguera-Antolinez}, A. and {Bertacca}, D. and {Bethermin}, M. and {Blanchard}, A. and {Blot}, L. and {Borgani}, S. and {Brown}, M.~L. and {Bruton}, S. and {Calabro}, A. and {Camacho Quevedo}, B. and {Caro}, F. and {Carvalho}, C.~S. and {Castro}, T. and {Cogato}, F. and {Conseil}, S. and {Cooray}, A.~R. and {Davini}, S. and {Desprez}, G. and {D{\'\i}az-S{\'a}nchez}, A. and {Diaz}, J.~J.},
        title = "{Euclid preparation. Accurate and precise data-driven angular power spectrum covariances}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = jun,
          eid = {arXiv:2506.09118},
        pages = {arXiv:2506.09118},
          doi = {10.48550/arXiv.2506.09118},
archivePrefix = {arXiv},
       eprint = {2506.09118},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250609118E},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

- Decoupling algorithms: WIP
