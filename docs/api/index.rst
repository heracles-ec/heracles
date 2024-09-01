
Using *Heracles* from Python
============================

*Heracles* is a fully-featured library for harmonic-space statistics on the
sphere.  The Python interface can be used, e.g., in exploratory work using
Jupyter notebooks, or to create automated data processing pipelines for a large
collaboration like *Euclid*.


Importing *Heracles*
--------------------

To use *Heracles* from Python, it is usually enough to import its main module::

    import heracles

The ``heracles`` module contains most user-facing functionality.  However, some
of *Heracles*' features require additional external dependencies; these
features are therefore encapsulated in their own separate modules::

    import heracles.healpy  # requires healpy
    import heracles.ducc  # requires ducc0
    import heracles.notebooks  # requires IPython, ipywidgets
    import heracles.rich  # requires rich


Python documentation
--------------------

The best way to get started with the Python interface for *Heracles* is to look
at the provided :doc:`/api/examples`.

The full list of user functionality is documented in the :doc:`/api/reference`.
