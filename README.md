# PyFHD
Python Fast Holographic Deconvolution

[![Documentation Status](https://readthedocs.org/projects/pyfhd/badge/?version=latest)](https://pyfhd.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/)

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

# TODO

- [ ] Reproduce the full_size_visibility_grid test with the group_arr fix (**Joel**)
- [ ] Produce a test for visibility_grid that specifically depends on how group_arr is created (**Nichole**)
- [ ] Produce a test for the model_ptr on visibility_grid (**Nichole**)
- [ ] Reproduce the third test for visibility_degrid and add a warning for having beam_per_baseline and interp_flag on at the same time. (**Joel**)
- [ ] Provide multiple examples for the input of fhd_main and fhd_setup (**Nichole**)
- [ ] Create Tests for in_situ_sim_setup and beam_setup, I hope I got them all, but I might've missed one somewhere. Feel free to check the IDL functions yourself, whereever there is a function from the FHD package, it should have a test ideally. (**Nichole**)
  - [ ] vis_noise_simulation
  - [ ] in_situ_sim_setup 
  - [ ] fhd_struct_init_antenna
  - [ ] mwa_dipole_mutual_coupling
  - [ ] mwa_beam_setup_init
  - [ ] mwa_beam_setup_gain
  - [ ] general_antenna_response
  - [ ] beam_power
  - [ ] beam_gaussian_decomp (This one may require some more thinking, I'll talk to Paul)
  - [ ] beam_image_hyperresolved
  - [ ] beam_dim_fit
  - [ ] fhd_struct_init_psf
  - [ ] apply_astrometry (we'll see how we go with this, AstroPy may want it different to this)
  - [ ] fhd_struct_update_obs
  - [ ] fhd_struct_init_jones
- [ ] Logging and Checkpointing 
  - [ ] Create a HDF5 Save (pyfhd_save) and Load (pyfhd_load) infrastructure for the project
  - [ ] Add logging using the logging package in Python
  - [ ] As a part of the logging, add the timing for each part of the PyFHD package.
- [ ] Decide how to split the main function into pieces to make it modular (via argparse, or pipeline packages like Nextflow), allowing people to choose to run certain parts of the PyFHD project, rather than running the full package (**Joel & Nichole**)
