# Contribution Guide

Welcome to your one stop shop for all the tips, tricks, best practices and HOW-TOs on contributing to `PyFHD`. This guide was made in an effort to keep `PyFHD` documented and maintainable encouraging all new and experienced people to follow best practices.

## Developer Installation of PyFHD

If you're a developer of PyFHD, welcome to the club! 

For development purposes I personally recommend using `mamba` or `conda` as changing your Python version when you need to is much easier than using `venv` for such purposes. Alternatively, if you wish to add features to `PyFHD` or eventually decide you have the need for speed and want modules made from other languages in there such as C/C++/Fortran/Julia etc. `mamba` /`conda` make it easier to install compilers or tools (even CUDA).

Here's how you install `PyFHD` for development purposes:

1. Clone and change directory into the `PyFHD` repo:
    ```bash
    git clone https://github.com/ADACS-Australia/PyFHD && cd PyFHD
    ```
2. Create a `pyfhd` virtual environment and automatically install the required dependencies
   ```bash
   conda env create --file environment.yml
   ```
3. Install PyFHD
    ```{important}
    The most important difference between a standard install and a developer install is in this step the `-e` in the command makes the package editable, meaning new runs of `PyFHD` will interact with the changes you make without having to reinstall `PyFHD` via pip for every change.
    ```
    ```bash
    pip install -e .
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

## The How-To on contributing to PyFHD

The below part of the guide will cover how to contribute to PyFHD, we'll go over creating a new function from scratch for `PyFHD`. We're going to use the examples of the `PyFHD.pyfhd_tools.pyfhd_utils.histogram` function because it's one of the best examples I've got for an excellent contribution to `PyFHD`. So let's go over making the histogram function for `PyFHD`, with each step focusing on best practices and the practices used in PyFHD.Many of the best practices applied in `PyFHD` are useful for any projet you're doing, so if you're new to programming and development in general, I hope you find these docs useful to you.

### Has someone invented the wheel?

The very first question you should ask yourself before making a new function is, do you need to or has someone already created it?

Take the `histogram` function for example, are you wondering why we did a histogram function? 

After all, `IDL's` histogram function can be done using a combination of `np.histogram` and the reverse indices can technically be done using many indexing functions available in NumPy as well. Well it turns out the NumPy functions like `np.histogram` suffer from a fatal flaw, they're awfully slow with large arrays, and this isn't necessarily their fault either, NumPy (rightly for their particular case) prioritised compatibility over raw speed. Other than speed, a function needed to be made to create the reverse indices part of `IDL's` histogram anyway as there is no `SciPy` or `NumPy` equivalent. As such it was decided to rewrite the histogram from scratch with speed as the priority given that `histogram` in `PyFHD` get's called hundreds, maybe hundreds of times.

To summarise, only make a new function if it hasn't been made before, or the ones that do exist do not meet your requirements. The requirements initally should not include speed unless you know before hand that the function is going to be called a lot or in a loop. 

If you're wondering what to do in the case that the function is in a library/package that is no longer being supported anymore? It's likely you'll need to remake it at some stage, and depending on the license of the original library/package, you might be able to copy and paste the function into PyFHD (ideally with acknowledgements to the original library/package).

### Making a new function

You've decided PyFHD needs a new function, hooray!

The next question you ought to ask yourself is where to put it. `PyFHD` has a directory setup where any functions that are used by certain parts of the `PyFHD` pipeline are stored in named directories. For example any function used exclusively for `gridding` ends up in the gridding directory, either as a new file or in the `gridding_utils.py` file. If the function will be used in all areas of `PyFHD` then and **only** then should you put the function inside the `pyfhd_tools/pyfhd_utils.py` file. 

If you think it's the start of a new part of the pipeline, then create a new directory, and inside that new directory you must also create an `__init__.py` file. The `__init__.py` file tells Python this is a part of the package and to look there for new modules, the `__init__.py` can be completely empty if you have no use for it. 

From there I'll leave it upto you, time to put that function from your head to a screen.

### Documenting and Typing the new function

You have made your new function, congratulations, now it's time to add some documentation and comments if you haven't already. In PyFHD, we use docstrings in the [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html) which is well documented in terms of what you should put into your docstrings and what each section in the docstring is for. There is specific formatting to follow and deviations from the format will result in weird outputs for read the docs once the docstring is read in for auto generating the API reference.

### Testing the new function

### The Need for Speed

### Adding your new function to the Changelog

### Pull Request for your new function

## IDL to Python Translation Guide

