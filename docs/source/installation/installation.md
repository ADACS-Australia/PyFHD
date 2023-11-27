# Installation

The dependencies on PyFHD to run with FHD have been removed, this makes installing and `PyFHD` much easier than FHD. `PyFHD` is currently supported for Python 3.10 and 3.11, support for newer Python versions of these should be simple once the dependencies that `PyFHD` relies on have support for newer versions.

## pip

PyFHD is not on [PyPi](https://pypi.org/) yet, so you'll need to clone the repository and install from the repository. The pip installation assumes you have already created a `venv` as per best practice.

1. Clone and change directory into the `PyFHD` repo:
    ```bash
    git clone https://github.com/ADACS-Australia/PyFHD && cd PyFHD
    ```
2. Install everything from the `requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```
3. Install PyFHD
    ```bash
    pip install .
    ```
    ```{note}
    `PyFHD` will eventually be available via PyPi meaning this step will be no longer required
    ```
4. Verify the installation using the version command
    ```bash
    pyfhd -v
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

       Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS)
       and the Epoch of Reionisation (EoR) Team.

       Repository: https://github.com/ADACS-Australia/PyFHD

       Documentation: https://pyfhd.readthedocs.io/en/latest/

       Version: 1.0

       Git Commit Hash: b77d18d0ef640297264ce700696d75aa4ff5ea82
    ```

## conda/mamba

PyFHD is not on [conda-forge](https://conda-forge.org/) yet, so you'll need to clone the code repository and install from the repository. The below commands will be shown using `conda`, however I heavily recommend using `mamba` instead of conda as it's generally faster and more reliable than `conda` in most cases and requires no learning to use it as`mamba` wraps around `conda`.

1. Clone and change directory into the `PyFHD` repo:
    ```bash
    git clone https://github.com/ADACS-Australia/PyFHD && cd PyFHD
    ```
2. Create a `pyfhd` virtual environment and automatically install the required dependencies
   ```bash
   conda env create --file environment.yml
   ```
3. Install PyFHD
    ```bash
    pip install .
    ```
    ```{note}
    `PyFHD` will eventually be available via conda-forge meaning this step will be no longer required
    ```
    
4. Verify the installation using the version command
    ```bash
    pyfhd -v
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

       Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS)
       and the Epoch of Reionisation (EoR) Team.

       Repository: https://github.com/ADACS-Australia/PyFHD

       Documentation: https://pyfhd.readthedocs.io/en/latest/

       Version: 1.0

       Git Commit Hash: b77d18d0ef640297264ce700696d75aa4ff5ea82
    ```

## Updating Depedencies and PyFHD

If you wish to update the existing packages in your repositories, it's mostly a case of re-running steps 2 and 3 with slight alterations to their commands.

### pip

1. Change Directory to your PyFHD install
   ```bash
   cd /path/to/PyFHD
   ```
2. Install the requirements from the `requirements.txt` and install PyFHD
    ```bash
    pip install -r requirements.txt --upgrade && pip install .
    ```

### conda/mamba

1. Change Directory to your PyFHD install
   ```bash
   cd /path/to/PyFHD
   ```
2. Update the environment from the `environment.yml` and install PyFHD
    ```bash
    conda env update --file environment.yml && pip install .
    ```

## Installing for additional development?

Please follow the contribution guide.