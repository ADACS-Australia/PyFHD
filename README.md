# PyFHD
Python Fast Holographic Deconvolution

[![Documentation Status](https://readthedocs.org/projects/pyfhd/badge/?version=latest)](https://pyfhd.readthedocs.io/en/latest/?badge=latest)

TODO: Add Testing Results and Testing coverage

## FHD
FHD is an open-source imaging algorithm for radio interferometers, specifically tested on MWA Phase I, MWA Phase II, PAPER, and HERA. There are three main use-cases for FHD: efficient image deconvolution for general radio astronomy, fast-mode Epoch of Reionization analysis, and simulation.

PyFHD is the translated library of FHD from IDL to Python, it aims to get close to the same results as the original FHD project. Do expect some minor differences compared to the original FHD project due to the many differences between IDL and Python. These differences are often due to the difference in precision between IDL and Python with IDL being single-precision (accurate upto 1e-8) and Python being double-precision (1e-16). Some of the IDL functions are double-precision but most default to single-precision.

## Quick Start
TODO: Add instructions here for getting started

## Useful Documentation Resources
TODO: Incorporate resources from the original FHD repository into here.

## Community Guidelines
We are an open-source community that interacts and discusses issues via GitHub. We encourage collaborative development. New users are encouraged to submit issues and pull requests and to create branches for new development and exploration. Comments and suggestions are welcome.

Please cite [Sullivan et al 2012](https://arxiv.org/abs/1209.1653) and [Barry et al 2019a](https://arxiv.org/abs/1901.02980) when publishing data reduction from FHD.

## Maintainers
FHD was built by Ian Sullivan and the University of Washington radio astronomy team. Maintainance is a group effort split across University of Washington and Brown University, with contributions from University of Melbourne and Arizona State University. 

PyFHD is currently being created by Nichole Barry and Astronomy Data and Computing Services (ADACS) members Joel Dunstan and Paul Hancock. ADACS is a collaboration between the University of Swinburne and Curtin Institute for Computation (CIC) located in Curtin University.
