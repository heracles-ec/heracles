Getting started
===============

These are the first steps to get up and running with *Heracles*.  First,
install the *Heracles* package for Python using one of the methods described
below.  Then, see the documentation on :doc:`/api/index` or :doc:`/cli/index`.


Installing a release
--------------------

This is the usual way of installing *Heracles*, and should be your first port
of call.

The *Heracles* package requires Python 3.9+ and a number of dependencies.  It
can be installed using `pip` in the usual manner:

.. code-block:: console

   $ pip install heracles

This will also install any missing dependencies.

.. note::

   As usual, it is recommended to install *Heracles* into a dedicated
   environment (conda, venv, etc.).

For alternative ways to install the package, see `Installing from the
repository`_ and `Developer installation`_.


Installing from the repository
------------------------------

To install the latest work-in-progress version, install directly from the main
branch of the git repository:

.. code-block:: console

   $ pip install git+https://github.com/heracles-ec/heracles.git

This will install *Heracles* with a local version string that encodes the last
release and any subsequent commits.


Developer installation
----------------------

If you want to install *Heracles* for local development, clone the repository
and install the package in editable mode:

.. code-block:: console

   $ # clone the repository
   $ cd heracles
   $ pip install -e .

The result is similar to `Installing from the repository`_, but uses the local
copy of the repository as the package source.  This means changes to your local
files are applied without needing to reinstall the package.
