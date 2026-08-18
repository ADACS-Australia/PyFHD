.. pyfhd documentation master file, created by
   sphinx-quickstart on Thu Feb 10 09:34:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyfhd's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents

Fast Holographic Deconvolution in Python

Started as a project created in IDL, FHD is an open-source imaging algorithm for
radio interferometers, specifically tested on MWA Phase I, MWA Phase II, PAPER,
HERA and OVRO-LWA. There are three main use-cases for FHD:

- efficient image deconvolution for general radio astronomy
- fast-mode Epoch of Reionization analysis
- simulation

**pyfhd** is a project to translate FHD into python to make it more widely available
to the community. pyfhd is under activate development as we work to translate
all the major components of FHD. The only major missing block currently is
deconvolution, but some less used options in FHD have not yet been implemented.
We invite the community to engage with us either by directly contributing to the
code via PRs or by making issues in our Issue Log.

Contents
-----------------

.. toctree::
   :maxdepth: 2

   installation/installation
   tutorial/tutorial
   develop/contribution_guide
   develop/idl_translation
   documentation/documentation
   changelog/changelog

Browse
-----------------

* :ref:`genindex`
* :ref:`modindex`
