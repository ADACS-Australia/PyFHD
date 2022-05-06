.. PyFHD documentation master file, created by
   sphinx-quickstart on Thu Feb 10 09:34:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyFHD's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents

**Py**\thon 
**F**\ast 
**H**\olographic 
**D**\econvolution

Started as a project created in IDL, FHD is an open-source imaging algorithm for radio interferometers, 
specifically tested on MWA Phase I, MWA Phase II, PAPER, and HERA. There are three main use-cases for FHD: 
efficient image deconvolution for general radio astronomy, fast-mode Epoch of Reionization analysis, and simulation.

Contents
========

* :ref:`genindex`
* :ref:`modindex`

Usage
=====

.. argparse::
   :ref: PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_parser
   :prog: pyfhd


PyFHD Setup
===========

.. automodule:: PyFHD.pyfhd_tools.pyfhd_setup
   :members:

Data Setup
==========

.. automodule:: PyFHD.data_setup.params
   :members:

.. automodule:: PyFHD.data_setup.obs
   :members:

Beam Setup
==========

Calibration
===========


calibration_utils
-----------------

.. automodule:: PyFHD.calibration.calibration_utils
   :members:

vis_calibrate_subroutine
------------------------

.. automodule:: PyFHD.calibration.vis_calibrate_subroutine
   :members:

Gridding
========

Gridding Utilities
------------------

.. automodule:: PyFHD.gridding.gridding_utils
   :members:

Filters
-------

.. automodule:: PyFHD.gridding.filters
   :members:

Visibility Grid
---------------

.. automodule:: PyFHD.gridding.visibility_grid
   :members:

Visibility Degrid
-----------------

.. automodule:: PyFHD.gridding.visibility_degrid
   :members:

Input & Output
==============

.. automodule:: PyFHD.io.pyfhd_io
   :members:

Tools & Utilities
=================

.. automodule:: PyFHD.pyfhd_tools.pyfhd_utils
   :members:

Testing Utilities
=================

.. automodule:: PyFHD.pyfhd_tools.test_utils
   :members:

Version History
===============

1.0
---

This repository is currently in development.