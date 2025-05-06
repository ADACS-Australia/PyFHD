# PyFHD
**Py**thon **F**ast **H**olographic **D**econvolution

[![Python](https://img.shields.io/badge/Python-3.10--3.13-%231475b3?logo=python&logoColor=%23fff)](https://www.python.org/)
![GitHub last commit](https://img.shields.io/github/last-commit/ADACS-Australia/PyFHD?logo=github&color=blue&link=https%3A%2F%2Fgithub.com%2FADACS-Australia%2FPyFHD%2Fcommits%2Fmain%2F)
[![Documentation Status](https://readthedocs.org/projects/pyfhd/badge/?version=latest)](https://pyfhd.readthedocs.io/en/latest/?badge=latest)
![GitHub branch status](https://img.shields.io/github/checks-status/ADACS-Australia/PyFHD/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TODO: Add Testing Results and Testing coverage

## FHD
FHD is an open-source imaging algorithm for radio interferometers, specifically tested on MWA Phase I, MWA Phase II, PAPER, and HERA. There are three main use-cases for FHD: efficient image deconvolution for general radio astronomy, fast-mode Epoch of Reionization analysis, and simulation.

PyFHD is the translated library of FHD from IDL to Python, it aims to get close to the same results as the original FHD project. Do expect some minor differences compared to the original FHD project due to the many differences between IDL and Python. These differences are often due to the difference in precision between IDL and Python with IDL being single-precision (accurate upto 1e-8) and Python being double-precision (1e-16). Some of the IDL functions are double-precision but most default to single-precision.

## Quick Start
Clone (and move into) the PyFHD repo:

```
git clone https://github.com/ADACS-Australia/PyFHD && cd PyFHD
```

Then just pip install the repo:

```
pip install -r requirements.txt .
```

For full installation notes, including dependencies on FHD, check out the [ReadTheDocs installation page](https://pyfhd.readthedocs.io/en/latest/installation/installation.html).

> Note: Once PyFHD is feature-complete, we aim to make this both a `conda` install and a `pip` install.

To check if PyFHD is available on your path, run the following command:

```
pyfhd -v
```

You should see output that resembles something like this:

```
   ________________________________________________________________________
    |    ooooooooo.               oooooooooooo ooooo   ooooo oooooooooo.    |
    |    8888   `Y88.             8888       8 8888    888   888     Y8b    |
    |    888   .d88' oooo    ooo  888          888     888   888      888   |
    |    888ooo88P'   `88.  .8'   888oooo8     888ooooo888   888      888   |
    |    888           `88..8'    888          888     888   888      888   |
    |    888            `888'     888          888     888   888     d88'   |
    |    o888o            .8'     o888o        o888o   o888o o888bood8P'    |
    |                 .o..P'                                                |
    |                `Y8P'                                                  |
    |_______________________________________________________________________|
    
    Python Fast Holographic Deconvolution 

    Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

    Repository: https://github.com/ADACS-Australia/PyFHD

    Documentation: https://pyfhd.readthedocs.io/en/latest/

    Version: 1.0

    Git Commit Hash: a36c37a966eb00ddcd936b9acee6bb7c71f9bb5a
```

To run the examples built into the repository and beyond, please find them here: [PyFHD Examples](https://pyfhd.readthedocs.io/en/latest/examples/examples.html)

## Useful Documentation Resources
 - [PyFHD documentation](https://pyfhd.readthedocs.io/en/latest/)
 - [MWA ASVO](https://asvo.mwatelescope.org/) - service to obtain MWA data
 - [FHD repository](https://github.com/EoRImaging/FHD) - the original IDL code
 - [FHD examples](https://github.com/EoRImaging/FHD/blob/master/examples.md) - examples on how to use the original IDL code
 - [FHD pipeline scripts](https://github.com/EoRImaging/pipeline_scripts) - pipeline scripts using the original IDL code

## Community Guidelines
We are an open-source community that interacts and discusses issues via GitHub. We encourage collaborative development. New users are encouraged to submit issues and pull requests and to create branches for new development and exploration. Comments and suggestions are welcome.

If you wish to contribute to PyFHD, first of all thank you, second please read the contribution guide which can be found here, [Contribution Guide](https://pyfhd.readthedocs.io/en/latest/develop/contribution_guide.html). The contribution will cover all you need to know for developing in PyFHD from adding features, formatting adding tests and some advice in translating IDL to Python.

Please cite [Sullivan et al 2012](https://arxiv.org/abs/1209.1653) and [Barry et al 2019a](https://arxiv.org/abs/1901.02980) when publishing data reduction from FHD.

## Maintainers
FHD was built by Ian Sullivan and the University of Washington radio astronomy team. Maintainance is a group effort split across University of Washington and Brown University, with contributions from University of Melbourne and Arizona State University. 

PyFHD is currently being created by Nichole Barry and Astronomy Data and Computing Services (ADACS) members Joel Dunstan, Paul Hancock, and Jack Line. ADACS is a collaboration between the University of Swinburne and Curtin Institute for Data Science (CIDS) located in Curtin University.

