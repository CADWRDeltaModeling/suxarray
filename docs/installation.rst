.. highlight:: shell

============
Installation
============


Stable release
--------------

The stable version may be slow to use the latest features and bug fixes. If you want to use the latest features and bug fixes, you can install the development version from the source, following the instruction in the next section.

If you use conda, run the following to install ``suxarray``:

.. code-block:: console

    $ conda install -c conda-forge suxarray

To install suxarray using ``pip``, run the command below in your terminal:

.. code-block:: console

    $ pip install suxarray

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From the source
---------------

``suxarray`` is actively developed on a forked version of ``uxarray``, so it may be best to install ``suxarray`` and ``uxarray`` from the sources to use the latest features and bug fixes.

The source codes of ``uxarray`` can be downloaded from a `forked uxarray Github repo`_.  Use ``suxarray`` branch from it. Install the source codes in the development mode as shown below:

.. code-block:: console

    $ git clone -b suxarray git://github.com/kjnam/uxarray
    $ cd uxarray
    $ python setup.py install -e .

The source codes for ``suxarray`` can be downloaded from the `Github repo`_. Check out a branch you want to use. Install the source codes in the development mode as shown below:

.. code-block:: console

    $ git clone git://github.com/cadwrdeltamodeling/suxarray
    $ cd suxarray
    $ python setup.py install -e .

.. _forked uxarray Github repo: https://github.com/kjnam/uxarray
.. _Github repo: https://github.com/cadwrdeltamodeling/suxarray
.. _tarball: https://github.com/cadwrdeltamodeling/suxarray/tarball/main
