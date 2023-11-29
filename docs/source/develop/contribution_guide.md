# Contribution Guide

Welcome to your one stop shop for all the tips, tricks, best practices and HOW-TOs on contributing to `PyFHD`. This guide was made in an effort to keep `PyFHD` documented and maintainable encouraging all new and experienced people to follow best practices.

## Developer Installation of PyFHD

If you're a developer of PyFHD, welcome to the club! 

For development purposes using `mamba` or `conda` as your package and environment manager makes changing your Python version when you need to is much easier than using `venv` for such purposes. Alternatively, if you wish to add features to `PyFHD` or eventually decide you have the need for speed and want modules made from other languages in there such as C/C++/Fortran/Julia etc. `mamba`/`conda` make it easier to install compilers or tools (even makes installing CUDA relatively easy).

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
    ```
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

This part of the guide will cover how to contribute to PyFHD, we'll go over some of the thoughts and ideas that one ought to use when creating a new function from scratch for `PyFHD`. We're going to use the examples along the way from `PyFHD` most of it showcasing the best of what `PyFHD` has done in each section. So let's go over making the histogram function for `PyFHD`, with each step focusing on best practices and the practices used in `PyFHD`. Many of the best practices applied in `PyFHD` are useful for any project you're doing, so if you're new to programming and development in general, I hope you find these docs useful to you.

### Has someone invented the wheel?

The very first question you should ask yourself before making a new function is, do you need to or has someone already created the function?

Take the `histogram` function for example, are you wondering why we did a histogram function? 

After all, `IDL's` histogram function can be done using a combination of `np.histogram` and the reverse indices can technically be done using many indexing functions available in NumPy as well. Well it turns out the NumPy functions like `np.histogram` suffer from a fatal flaw, they're awfully slow with large arrays, and this isn't necessarily their fault either, NumPy (rightly for their particular case) prioritised compatibility over raw speed. Other than speed, a function needed to be made to create the reverse indices part of `IDL's` histogram anyway as there is no `SciPy` or `NumPy` equivalent. As such it was decided to rewrite the histogram from scratch with speed as the priority given that `histogram` in `PyFHD` get's called hundreds, maybe hundreds of times.

To summarise, only make a new function if it hasn't been made before, or the ones that do exist do not meet your requirements. The requirements initally should not include anything in rgeards to the speed of the function unless you know before hand that the function is going to be called a lot or in a loop. 

If you're wondering what to do in the case that the function is in a library/package that is no longer being supported anymore? It's likely you'll need to remake it at some stage, and depending on the license of the original library/package, you might be able to copy and paste the function into PyFHD (with acknowledgements to the original library/package).

### Making a new function

You've decided PyFHD needs a new function, excellent looking forward to your contribution!

The next question you ought to ask yourself is where to put the new function. `PyFHD` has a directory system where any functions that are used by certain parts of the `PyFHD` pipeline are stored in named directories. For example any function used exclusively for `gridding` ends up in the gridding directory, either as a new file or in the `gridding_utils.py` file. If the function will be used in all areas of `PyFHD` then and **only** then should you put the function inside the `pyfhd_tools/pyfhd_utils.py` file.

If you think it's the start of a new part of the pipeline, then create a new directory, and inside that new directory you must also create an `__init__.py` file. The `__init__.py` file tells Python this is a part of the package and to look there for new modules, the `__init__.py` can be completely empty if you have no use for it. 

From there I'll leave it upto you, time to put that function from your head to a screen.

### Documenting and Typing the new function

You have made your new function, congratulations, now it's time to add some documentation and comments if you haven't already. In PyFHD, we use docstrings in the [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html) which is well documented in terms of what you should put into your docstrings and what each section in the docstring is for. There is specific formatting to follow and deviations from the format will result in weird outputs for read the docs once the docstring is read in for auto generating the API reference so please follow the numpydoc format precisely.

When it comes to comments in PyFHD, there are some functions which are heavily commented and others are not, in general comments should be about the intention and the reason why the code exists, rather than what it does. It's often said good, clean code shouldn't need comments as it's descriptive enough to be followed, which for the most part is somewhat True, however, I personally don't think that's always True, and so having comments about what something does isn't necessarily a bad thing. The rule I like to follow is somewhere in between, some pieces of code do occasionally need you to say what they are doing as good docstrings can also tell people why you have done what you have done. Remember, comments are gifts to yourself when you have to re-read the same function a year later, so if a function has more comments than you need, who cares?

In PyFHD we're also using the Python typing systems and packages. In PyFHD they have been primarily used in the definition of functions, to ensure the not only is the type of each variable visible, but in the case of NumPy typing, you can also show the expected precision. The reason for using the typing system is making the development experience better in IDE's like VSCode or PyCharm, as the IDE will show you the function as you're typing it out. It only takes an extra 2 seconds per parameter on the defintion of a new function but can save you hours of pain. Features inside IDE's such as code completion, code hinting usually come under the umbrella of [Intellisense](https://code.visualstudio.com/docs/editor/intellisense). In the future, it's possible to use tools such as [MyPy](https://mypy.readthedocs.io/en/stable/) to enforce these types too.

For examples of docstrings in PyFHD check out the following doc strings from the `get_ri` and `histogram` functions. 

```python
# Notice how we can specificy we're wanting a numpy array of type int64 or float64, makes it clear to users of this
# function the limits and bounds of what the function can take as input.
def get_ri(data: NDArray[np.float_ | np.int_ | np.complex_], bins: NDArray[np.float64 | np.int64], hist: NDArray[np.int64], min: int | float, max: int | float) -> NDArray[np.int64]:
    """
    Calculates the reverse indices of a data and histogram. 
    The function replicates IDL's REVERSE_INDICES keyword within
    their HISTOGRAM function. The reverse indices array which is returned
    by this function can be hard to understand at first, I will explain it here
    and also link JD Smith's famous article on IDL's HISTOGRAM.

    The reverse indices array is two vectors concatenated together. The first vector contains
    indexes for the second vector, this vector should be the size of bins + 1. 
    The second vector contains indexes from the data itself, and should be the size of data.
    The justification for having such an array is to quickly make adjustments to certain bins
    without having to search the array multiple times, thus avoiding multiple O(data.size) loops.

    The first vector indexes contain the starting positions of each bin in the second vector. For example, 
    between the indexes given by first_vector[0] and first_vector[1] of the second vector should be all the 
    indexes of bins[0] from inside the data. So if I wanted to make adjustments to the entire first bin, 
    and only the first bin I can use the reverse indices array, ri to do this. Let's say I wanted to flag
    all values of bins[0] with -1 for some reason to make them invalid in other calculations with the data, 
    then I could do this:

    `data[ri[ri[0] : ri[1]]] = -1`

    Or more generally

    `data[ri[ri[i] : ri[i + 1]]] = -1`

    Where i is 0 <= i <= bins.size.

    If you wish to gain a better understanding of how this get_ri function works, and the associated
    histogram function I have created here, please use the link given in the Notes section. This
    link will take you JD Smith's article on IDL's HISTOGRAM, it is an article which explains the
    IDL HISTOGRAM function better than IDL's own documentation. If you must gain a deeper understanding,
    read it once, gasp and get your shocks and many cries of why out of your system, then read it again.
    And keep reading it till you understand, as per the editor's note on the article:

    "...If you read it enough, the secrets of the command will be revealed to you. Stranger things have happened"

    Parameters
    ----------
    data : NDArray[np.float\_ | np.int\_ | np.complex\_]
        A NumPy array of the data
    bins : NDArray[np.float64 | np.int64]
        A NumPy array containing the bins for the histogram
    hist : NDArray[np.int64]
        A NumPy array containing the histogram
    min : int | float
        The minimum for the dataset
    max : int | float
        The maximum for the dataset

    Returns
    -------
    ri : NDArray[np.int64]
        An array containing the reverse indices, which is two vectors, the first vector containing indexes for the second vector.
        The second vector contains indexes for the data.

    See Also
    ---------
    histogram: Calculates the bins, histogram and reverse indices.
    get_bins: Calculates the bins only
    get_hist: Calculates histogram only

    Notes
    ------
    'HISTOGRAM: The Breathless Horror and Disgust' : http://www.idlcoyote.com/tips/histogram_tutorial.html
    """

    # Hopefully you noticed a long descriptive docstring, you should also notice the types
    # and returns are in there as well
    pass

# Again notice the typing, and the typing of the return, again makes it clear to people to always expect three return
# variables, and not only that, what type to expect them to be.
def histogram(data : NDArray[np.float_ | np.int_ | np.complex_], bin_size: int = 1, num_bins: int | None = None, min: int | float | None = None, max: int | float | None = None) -> tuple[NDArray[np.int64], NDArray[np.float64 | np.int64], NDArray[np.int64]]:
    """
    The histogram function combines the use of the get_bins, get_hist and get_ri
    functions into one function. For the descriptions and docs of those functions
    look in See Also. This function will return the histogram, bin/bin_edges and
    the reverse indices.

    Parameters
    ----------
    data : NDArray[np.float\_ | np.int\_ | np.complex\_]
        A NumPy array containing the data we want a histogram of
    bin_size : int, optional
        Sets the bin size for this histogram, by default 1
    num_bins : int | None, optional
        Set the number of bins this does override bin_size completely, by default None
    min :  int | float | None, optional
        Set a minimum for the dataset, by default None
    max :  int | float | None, optional
        Set a maximum for the dataset, by default None

    Returns
    -------
    hist : NDArray[np.int64]
        The histogram of the data
    bins : NDArray[np.float64 | np.int64] 
        The bins of the histogram
    ri : NDArray[np.int64]
        The reverse indices array for the histogram and data
    
    See Also
    --------
    get_bins: Calculates the bins only
    get_hist: Calculates histogram only
    get_ri: Calculates the reverse indices only
    """

    # Take note of how multiple returns are defined here, that format is needed exactly for numpydocs and readthedocs
    # to represent multiple returns properly.
    # Also take note of the See Also sections pointing potentially to similar functions, or functions used
    # inside of the histogram function
    pass
```

You should notice the typing system in use with the colon's after each parameter telling you what types the inputs should be. Using extensions for VSCode or PyCharm can allow you to automatically create the skeleton for the docstring such as [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) for VSCode. You'll notice as well when using extensions for the auto doc strings that the typing you have hopefully just implemented will be picked up automatically with little to no extra effort needed for formatting. Using the typing system may feel like a waste of time at first, but the more you use typing and the more you use typed functions the more you realise you're no longer thinking about the small details of functions anymore, but focusing on what they do and how/when they're used instead.

Combining docstrings, comments and typing together when used well and best practices followed allows for easier, faster debugging in the future. Furthermore, it's easier to keep track of your changes and how they could affect the rest of the codebase. If that sounds good to you as it does to us, let's keep the faster debugging train going. 

### Testing the new function

In an ideal world, every new function you make has at least 3 tests. In this ideal world the tests cover corner cases as well as expected cases. It has been my experience so far in `PyFHD` that small simulated data seems to work well for testing these functions and usually gives a great indication that `PyFHD` will work with real data. The simulated data also allows you to more easily find those corner cases and ensures testing the function doesn't take hours per test. With testing functions like `histogram` we had the benefit of creating as many tests as we wanted given we were replicating the IDL function, this is the best case scenario. With functions that were translated from `FHD`, in general we'd run the functions in `FHD` and save the input variables and the output variables. The `sav` files would then be read in by a testing function and generally converted to HDF5 for faster and easier reading in the future for the same test. 

For creating a test in `PyFHD` we have utilised the pytest framework which is a powerful testing framework that searches for any function and/or directory that has the word `test` at the start or end of the function name or directory name. A useful feature of pytest are `fixtures` which are heavily used throughout the testing of PyFHD. Fixtures allow you to make functions which perform routines needed for a function, in the case of `test_histogram.py` I used a fixture to spread the data directory across all functions without having to contunally copy and paste the path. When making a test for `PyFHD` the `numpy.testing` package was particularly useful as it contains functions like `np.testing.assert_allclose` which allow you to test the differences between arrays upto an absolute and/or relative precision. See below for a test made for the histogram function which utilises a pytest fixture:

```python
import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.pyfhd_utils import histogram
from PyFHD.pyfhd_tools.test_utils import get_data, get_data_items

@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return Path(env.get('PYFHD_TEST_PATH'), 'histogram')

@pytest.fixture
def full_data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'full_size_histogram')

def test_idl_example(data_dir: Path) :
    """
    This test is based on the example from the IDL documentation.
    This ensures we get the same behaviour as an example everybody can see.
    """
    # Setup the test from the histogram data file
    data, expected_hist, expected_indices = get_data(data_dir, 'idl_hist_example.npy', 'idl_example_hist.npy', 'idl_example_inds.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)
```

Pytest fixtures have the ability to group other fixtures too, take `test_cal_auto_ratio_divide` as an example, `test_cal_auto_ratio_divide` was set up with multiple pytest fixtures with multiple parameters. These pytest fixtures in `test_cal_auto_ratio_divide` automatically generate groups of tests without duplicating code for each test. You should also notice the use of the `numpy.testing` library as well. The below code generates 6 tests using the fixtures `tag` and `run` primarily. This test also shows how you can skip certain groups if you need to.

```python
from PyFHD.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from PyFHD.calibration.calibration_utils import cal_auto_ratio_divide
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from PyFHD.io.pyfhd_io import save, load
import numpy.testing as npt

@pytest.fixture()
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "cal_auto_ratio_divide")

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"]]

# For each combination of tag and run, check if the hdf5 file exists, if not, create it and either way return the path
# Tests will fail if the fixture fails, not too worried about exceptions here.

@pytest.fixture()
def before_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    obs = recarray_to_dict(sav_dict['obs'])
    cal = recarray_to_dict(sav_dict['cal'])
    vis_auto = sav_file_vis_arr_swap_axes(sav_dict['vis_auto'])
        
    #super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = obs
    h5_save_dict['cal'] = cal
    h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
    h5_save_dict['vis_auto'] = vis_auto
    h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']

    save(before_file, h5_save_dict, "before_file")

    return before_file

# Same as the before_file fixture, except we're taking the the after files
@pytest.fixture()
def after_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    
    sav_file = after_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    #super dictionary to save everything in
    h5_save_dict = {}

    h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])
    h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
    h5_save_dict['auto_ratio'] = sav_file_vis_arr_swap_axes(sav_dict['auto_ratio'])

    save(after_file, h5_save_dict, "after_file")

    return after_file

def test_cal_auto_ratio_divide(before_file, after_file):
    """
    Runs all the given tests on `cal_auto_ratio_divide` reads in the data in before_file and after_file,
    and then calls `cal_auto_ratio_divide`, checking the outputs match expectations
    """
    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}")

    h5_before = load(before_file)
    h5_after = load(after_file)

    obs = h5_before['obs']
    cal = h5_before['cal']
    vis_auto = h5_before['vis_auto']
    auto_tile_i = h5_before['auto_tile_i']

    expected_cal = h5_after['cal']
    expected_auto_ratio = h5_after['auto_ratio']

    result_cal, result_auto_ratio = cal_auto_ratio_divide(obs, cal, vis_auto, auto_tile_i)

    atol = 8e-6
    npt.assert_allclose(expected_auto_ratio, result_auto_ratio, atol=atol)

    #check the gains have been updated
    npt.assert_allclose(expected_cal['gain'], result_cal['gain'], atol=atol)
```

Once you have set up the tests, evry time you change the code and re-run the test you can be sure you haven't broken existing functionality accidentally giving you more confidence that the changes you have made will improve `PyFHD`. 

Another tool that will help you during testing is the debugging tools available in IDE's like those in VSCode and PyCharm. Debugging tools allow you to get away from using print statements to using breakpoints instead, allowing you to see the entire snapshot of your function at that point in time in the code. This does make it easier to track down issues, see portions of large arrays, watch certain variables throughout the code to see how they change etc. VSCode in particular can allow you to set log points which can act like print statements without you having to put anything into the code. Breakpoints can also have conditions attached to them, so if you know you're looping through frequencies and you know an issue happens with frequency index 13, you can make a breakpoint condition that will trigger only when the frequency index is 13. Check the debugging tools for VSCode [here](https://code.visualstudio.com/docs/python/debugging) and the tools for PyCharm [here](https://www.jetbrains.com/help/pycharm/debugging-code.html).

This section wasn't about teaching you how to test, but to show why we do it. It's testing that has enabled us to be sure that `PyFHD` actually does the same things as `FHD` and even in some cases detect bugs on the `FHD` side. If you want to learn more about testing check out the numpy testing functions [here](https://numpy.org/doc/stable/reference/routines.testing.html) and also check out pytest [here](https://docs.pytest.org/en/7.4.x/).

### The Need for Speed

If you're function is running a little slower than you'd like explore the [Numba](https://numba.readthedocs.io/en/stable/index.html) library. Numba is a **J**ust-**i**n-**T**ime (JIT) compiler that compiles your Python code at run time into machine code, it can make your code sveral magintudes faster than it is with little adjustments to the code. One of the nice things about Numba is that you can get away from using vectorization in NumPy and just do things in loops that require less thinking. However, Numba has a particular way of doing things and is harder to debug as the error messages are harder to decipher. Numba does also support CUDA, so if you want to write things for a GPU you can, however it's a slightly different paradigm when it comes to programming for GPUs vs CPUs so be aware your code will change significantly and will likely require you to have a CPU and GPU version of your function. If you are wanting to adopt GPUs consider using [CuPY](https://docs.cupy.dev/en/stable/user_guide/index.html) which can allow you to use GPUs by changing calls of NumPy via `np` to using CuPy using `cp`, it is also possible to create [CPU/GPU agnostic code](https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code) with CuPy. Another library that can support CPU/GPU agnostic code is [Dask](https://docs.dask.org/en/stable/10-minutes-to-dask.html) which is a library made for parallel computing with interpoability with NumPy and Pandas, however it does have less support for a lot of the mathematical functions and linear algebra done in `PyFHD`. 

Given that we're using Python, you can of course create a module in C/C++/Fortran/Julia/Rust and use that within `PyFHD`. I have purposely for now not done anything in other languages to avoid having to compile either during installation or publication, I'll leave that upto the EoR community to decide.

### Adding your new function to the Changelog

Once you have created your new function, add information about your new function in the new features, giving reasons and to why and when you'd want to use it. You should also notice in the changelog that there is sections for bug fixes, test changes, dependency changes, version changes and translation changes. The bug fixes section should be updated in case you have fixed issues found in the code. The test changes section should be updated in the case you have added, modified or deleted any of the tests, please indicate how they have changed and why. If tests now break because of your changes, ideally fix them before getting to this point by either changing them with comments in the tests as to why they changed, and updating the changelog accordingly. Dependency changes are there to indicate any changes made to the dependencies needed for `PyFHD`, changes mentioned here should be added to the `requirements.txt` and `environment.yml` files. Translation changes refer to any translation efforts from `FHD` to `PyFHD`, or more generally IDL to Python, mention what was translated and why. If there are parts of the `FHD` code you didn't translate mention why in the changelog so someone who does want to tackle it in the future can understand any potential challenges before they begin.

### Pull Request for your new function

Initiate the pull request back to whatever branch the function needs to go in, and during the pull request you should notice a Pull Request Template that pops up with checkboxes, please follow the pull request template and do what it asks of you, one of which you're doing now by reading this contribution guide (Thanks 🙌). The pull request template is there for you to make sure you have followed best practices and for the reviewers to check you have followed best practices as well. 

**If you're not doing any translation from IDL (`FHD`) to Python (`PyFHD`) you can safely stop here, and embark on your many scientific endeavours with PyFHD, good hunting for that mysterious signal.**

## IDL to Python Translation Guide

This part of the guide is going to be more personal and contains lessons from my own (Joel Dunstan, [SkyWa7ch3r](https://github.com/SkyWa7ch3r)) 2 year experience of translating the majority of `FHD` from IDL to Python to create `PyFHD`.

Hello there weary traveler, you wish to translate from IDL to Python or have been tasked with doing so, first thing my sincerest apologies for your future sanity and also to your colleagues as you likely vent to them about the annoyances of doing the translation. Here I'll try to put all the discoveries, roadbumps and potholes that will slow you down along the way. Be prepared to laugh, sigh, be filled with rage and maybe shed a tear if you find out a problem you've been facing for a week is because of some ~~dumb~~ *interesting* behaviour.

First thing, you'll need access to IDL, this is easier said than done as you require a license to use the language if you're lucky your instituion has an IDL license or the HPC you use has an IDL license. If you plan on installing IDL to your local machine you will need to contact [NV5 Geospatial](https://www.nv5geospatialsoftware.com/Products/IDL) directly to get a license and get the download link through their locked download links. If you're institution has a license server you should be able to get a license from the server and an install package from your institution. There are open source versions of IDL, [`GDL`](https://github.com/gnudatalanguage/gdl) and [`Fawlty`](https://www.flxpert.hu/fl/), however in my experience they didn't support running all the functions needed for `FHD` or produced different results to what's expected even in simple scenarios, hopefully that has changed and you can try to use them, feel free to let me know how that went.

Now that you have IDL, you'll need to follow the [FHD install steps](https://github.com/EoRImaging/FHD/tree/master#installation). You'll notice that all the functions get added to the IDL path, IDL only really has one namespace for functions, do keep that in mind unlike Python which has packages/libraries, modules/subpackages which divide up the name spaces and require explicit imports. 

Now in terms of matching up some of the `FHD` dependencies and `PyFHD` dependencies, anything from the `FHD`, `fhdps_utils`, and `pipeline_scripts` repositories will require straight translation line by line. For the `IDLAstro` library from NASA, a mix of `astropy`, `numpy` and `scipy` should allow you to either drectly replace functions when they appear in the `FHD` code or allow much easier translation/rewriting of the function. For the `IDLAstro` library all the `pro` files are publicly available so if there is a need to directly translate the function into Python that is possible. `coyote` does much of the graphical heavy lifting in `FHD`, for any plots inside of `PyFHD`, the plotting/imaging libraries `matplotlib`, `seaborn`, `plotly`, `Pillow` and `OpenCV` should get you what you need to make almost any visualization you wish to make (`plotly` is excellent when you need interactivity). For `HEALPix` you have the `healpy` library maintained by the same organization, so functions should have similar names and similar purposes without any translation needed.

In terms of tools you can use to translate IDL to Python, after all this time, I've decided the best tool is **you**. None of the tools out there like [`idl2py`](https://github.com/dnidever/idl2py) successfully translate to Python in a way that makes it faster than you translating the function line by line, they even occasionally miss things like `FLTARR` which translate perfectly into `np.zeros`. To be fair on these tools its an almost impossible task to automate because in IDL keywords in functions can *entirely change how the function works* thus any code you make to automatically translate will be full of exceptions and will still have problems translating pointer arrays due to the shapes of said arrays not always been easy to know ahead of time. Unfortunately not even ChatGPT or LLMs can fully help you, although to be honest they do a somewhat decent job, feel free to experiment with them and let me know how it goes, although I suspect it will miss some of the interesting behaviour that I will point out later. If you're using VSCode, NV5 recently brought out an official VSCode extension for IDL, check it out [here](https://marketplace.visualstudio.com/items?itemName=IDL.idl-for-vscode). IDL for VSCode will give you proper syntax highlighting for `pro` files, and hopefully they will continue to increase it's usefulness in the future.

Another pair of tools that are worth talking about is the `IDL to Python Bridge` and `Python to IDL Bridge` that is provided as a part of the IDL package. The `Python to IDL Bridge` can provide some nice to have features such as running IDL code in a Jupyter notebook which can be particularly useful for debugging some IDL to Python translations because you can rerun just parts of the code you're having issues with translating. The `IDL to Python Bridge` does also make it possible to use Python within IDL, while the `Python to IDL Bridge` makes it possible to use IDL in Python. At the time of writing this its now possible to pass objects to and from IDL in the `Python to IDL Bridge`, however I cannot confirm if you can still only run the built-in IDL functions in the `Python to IDL Bridge`. When I previously used the bridge I wasn't able to run functions made inside of `FHD` such as `weight_invert` (I suspected this was due to the IDL_PATH being setup everytime you re-ran IDL.run for every command). Hopefully if you try the bridges on the latest version of IDL, you may have better luck meaning you could have more power for debugging than I did because you can directly run the IDL code side by side and directly compare the IDL and Python outputs without the need for `sav` files. Feel free to let me know how this changes or to update this section yourself.

There are several gotchas hidden thoughout IDL, usually most of them won't appear to an unexperienced person who hasn't been burned by them before until they try to run the Python equivalent, for example, me. I will try to list as many as I can here, lot of these can be found in IDL documentation as extra notes (rather than a big notice saying "Hey this is important, if you use this function this way, this function is entirely different" which would be a little nicer):
* IDL is column based while Python is row based, meaning you will likely need to change the shape of certain arrays for the multiplication of arrays and matrices to work as you expect. It's the reason why the visibility arrays go by polarization then frequency then baselines in PyFHD compared to polarization then baselines then frequencies in `FHD` as it allows us to translate the indexing as it's done in `FHD` without having to constantly swap indexes around during translation which gets confusing.
* IDL is single precision even if it kills itself, in many cases even if you specify double precision using the `/DOUBLE` keyword, there's bound to be some function somewhere that is restricted to single precision. This is slowly changing.
* Function parameters in IDL can be input **or** output! If during your translation a variable exists seemingly without being initialized somewhere, check the function calls, parameters store the results, this is the case with many IDL functions, like `HISTOGRAM` where the `REVERSE_INDICES=ri` part of the call actually *returns* the reverse indices into an `ri` variable. This is also used extensively for many `FHD` functions and can get confusing, in Python doing returns this way is possible but for the sake of being consistent, any and all returns are done with an explicit `return` statement and put into the docstrings as a return.  
* When dealing with how IDL deals with subsetting, it's important to remember than IDL is inclusive of *both* the start and the end index, i.e. if we index an array between the 0th value and the 3rd value, we will get the 0th value and all the values upto and including the 3rd value. In Python the subsetting is only inclusive of the start but not the end index, i.e. if we index an array between the 0th value and the 3rd value, we will get the 0th value and all the values upto the 3rd value, but not including the third value.
    
    IDL
    ```idl
    IDL> test = [1,2,3,4,5]
    IDL> test[0:3]
           1       2       3       4
    ```

    Python
    ```python
    >>> test = np.array([1,2,3,4,5])
    >>> test[0:3]
    array([1, 2, 3])
    ```

    The inclusion of both ends in IDL also works the same for loops of any kind, so when translating you should be careful with how you index the array at the end primarily or the end conditions for your loop. You will find many users of IDL have already taken this into account and loop `n` times going from `0` to `n - 1`, so in python using the `range(n)` function should do the trick.
* The median function in IDL doesn't do the average between the two middle numbers by default when dealing with an array of even size unless you use the `/EVEN` keyword, it will take the maximum of the two numbers when dealing with an even sized array be default. This has often made a difference when dealing with close to single precision values, in my experience you're usually better off using `np.median` in most cases and accepting the differences, there has only been one notable case where we needed the IDL median over `np.median` and that's during the calibration process (as the comparison to `conv_thresh` would fail due to the precision differences).
* Matrix Multiply does the dot product for matrices, and because IDL is column based while Python is row based, you will need to switch around the order of multiplication in Python compared to IDL. Furthermore, matrix multiply does the outer product when you make on the provided matrices/arrays 1D array. So everytime you see matrix multiply you'll likely have to check at runtime during testing if the multiply is meant to be the dot product or outer product.
* In the case of an array containign only `NaNs` there is a slight difference between IDL's `total` and `mean`, where  `total(x,/nan)` will give you `0` while `mean(x, /nan)` will give you a `NaN`
* In IDL `ATAN` when used with complex numbers and the `/PHASE` keyword is actually an arctan with the imaginary part divided by the real part i.e. in IDL `ATAN(COMPLEX(2,1), /phase) EQ 0.46364760` while in Python the same code is done as `np.arctan(1/2)`. 
* When doing a divide by 0 in IDL, it does produce an error `% Program caused arithmetic error: Integer divide by 0` however it doesn't actually break the program, IDL continues to work, watch out for it! Don't ask me why they've done it...and if you don't believe me here is an example in IDL 8.8.0:
  ```idl
  IDL> test = [[1.,2.],[3.,4.]]
  IDL> test[0] = 1/0 + 10
  % Program caused arithmetic error: Integer divide by 0
  IDL> test
  11.000000       2.0000000
  3.0000000       4.0000000
  ```
* Indexing arrays using other arrays can also lead to ~~dumb~~ *interesting* behaviour, if we follow our test array again of 4 values and we try to access the 5th element of the array we expectedly get an error.
    ```idl
    IDL> test = [[1.,2.],[3.,4.]]
    IDL> test[5]
    % Attempt to subscript TEST with <INT      (       5)> is out of range.
    % Execution halted at: $MAIN$ 
    ```
    However if we put that index into another array, let's call it `idx` and then access the `test` array with the `idx` array we get the following:
    ```idl
    IDL> test = [[1.,2.],[3.,4.]]
    IDL> idx = [5]
    IDL> test[idx]
       4.0000000
    ```
    Yep that's the last value of the array, and to really get the ball into the endzone:
    ```idl
    IDL> idx = indgen(10) + 1000
    IDL> idx
      1000    1001    1002    1003    1004    1005    1006    1007    1008    1009
    IDL> test[idx]
       4.0000000       4.0000000       4.0000000       4.0000000       4.0000000       4.0000000       4.0000000
       4.0000000       4.0000000       4.0000000
    ```
    In summary, watch out for this behaviour, it does not fly in any other language, I have absolutely no idea why this is a thing or why you'd want it.
* The `POLY_FIT` function in IDL has the ability to do both `polyfit` and `polyval`, so watch out for the use of the `YFIT` keyword. In general, `poly_fit(a, b, 2)` in IDL is `np.polynomial.polynomial.Polynomial.fit(a, b, deg=2).convert().coef` in Python. If you see the `YFIT` keyword, then it's a case of calling `np.polynomial.polynomial.polyval` on the calculated coefficients i.e. `phase_fit = np.polynomial.polynomial.polyval(a, np.polynomial.polynomial.Polynomial.fit(a, b, deg=2).convert().coef)`.
* `LA_LEAST_SQUARES` in IDL and `np.linalg.lstsq` in Python (and the SciPy least square functions as well) will produce different results due to the solvers in use, and the assumption that the design matrix given to the solvers is full rank in `LA_LEAST_SQUARES` while NumPy and SciPy assuming that the design matrix is not full rank. Even using the same solvers doesn't give the same results, so it might come down to how each of the functions were compiled, either way, keep into consideration they just won't get the same results in most cirumstances.
* The `REBIN` has been replicated in `PyFHD`, however it's use should be used sparingly as its doing interpolation when increasng an array in size, and averaging when decreasing the size of an array i.e. it's  upscaling or downscaling and it's treating your array or matrix as an image rather than any ordinary array. In many cases it could be best to use `np.tile`, `np.pad` or `np.repeat` to do the same task more consistently.
* The `SMOOTH` function in IDL is a boxcar averaging function so it should be replaced with `scipy.ndimage.uniform_filter` and the `/edge` keywords will dictate the mode you need to use for the `uniform_filter` function.
* When dealing with complex numbers in IDL, it's possible to get different results when using `WHERE` in IDL and `np.where` as `WHERE` in IDL applies to the absolute values of the numbers, while `np.where` looks at just the real numbers for example:
  
  IDL
  ```idl
  IDL> test = COMPLEX([1,2,0],[1,2,3])
  IDL> WHERE(test gt 1)
           0           1           2
  ```

  Python
  ```python
  >>> test = np.array([1 + 1j, 2 + 2j, 0 + 3j])
  >>> np.where(test > 1)
  (array([0, 1]),)
  >>> np.where(np.abs(test) > 1)
  (array([0, 1, 2]),)
  ```

  It has only affected my translation once, but something to be aware of.
* Pointers are used in IDL a lot, especially in `FHD`, when translating pointer arrays my advice is to try and find out the full shape of the array and set the dtype appropriately, rather than creating an `object` array or `list` to represent the pointer array. `object` arrays in NumPy usually lead to more problems than they're worth, and make using the built in vectorized functions a real pain.
* There is a `readsav` function in `scipy.io` that gives you the ability to read in `sav` files, which is great that it exists. Unfortunately though, `readsav` is usually quite slow due to IDL's differences to other languages down to the byte level, as such it's having to loop through all the bytes to get the necessary bits out (poor thing, imagine having to read a 44GB file *several bytes at a time in a python loop*). `readsav` does read the `sav` file into a python dictionary which sounds good, except that it turns structures that were saved into the `sav` files into `np.recarrays` or NumPy record arrays. NumPy record arrays are often slower than dictionaries to access and in the case of `PyFHD` offer no benefits over dictionaries (in fact I'd argue a dictionary is better is almost all circumstances). `readsav` also usually makes a mess of pointer arrays too by turning pointer arrays into `object` arrays which each index contains a `numpy.ndarray`, meaning to access an array you might have to do something like `sav_file['array'][0]array[0][0]` to access the first numpy array in the array of arrays. As such I developed the `PyFHD.io.recarray_to_dict` function which can take in a `np.recarray` or `dict` and turn any record arrays into a `dict`. `recarray_to_dict` works recursively too, so any sub-record arrays will also turn into a `dict`, furthermore, additional formatting of arrays will take place to turn any `object` arrays into a proper NumPy array with a numpy data type (dtype). `recarray_to_dict` also deals with the values that are scalar values when loaded in from the `sav` file as well, `readsav` usually turns scalar values into arrays with a single vaue which are inconvenient to access. `recarray_to_dict` is always a work in progress, happy to try and edit the function as necessary.
* If you're dealing with large pointer arrays that can't convert to a proper dtype array due their size, e.g. the `beam_ptr` complex array was an example of this with a size of `2*384*8128*51*51*2916*16 bytes` or `~757.5 TB`, then ideally you need to convert these into HDF5 (with `h5py`) files which can allow you to chunk large datasets. It's also worth checking that there aren't pointers referencing other pointers, this was the case with `beam_ptr` as such theactual size ended up being `2*384*51*51*2916*16 bytes` or `~93.2 GB` instead. It's also possible to use other frameworks like `Dask` to achieve what we're doing with `h5py`, but we can use `h5py` without having to worry about compatiblity with NumPy functions.
* In IDL concatenating arrays can be done in many ways, one of the ways you may see is like so:
  ```idl
  IDL> test = [1,2,3]
  IDL> test = [test, test]
  IDL> test
         1       2       3       1       2       3
  ```
  So be on the look out for that sort of behaviour.
* The `>` and `<` operators in IDL do the same behaviour as `np.maximum` and `np.minimum` respectively, do not get them confused with the `GT` (greater than) and `LT` (less than) operators in IDL. Furthermore, the `>` and `<` operators in IDL can be chained together, I'll provide many examples below:
    ```idl
    IDL> test = [-1,2,3,-4,5]
    IDL> test > 0
       0       2       3       0       5
    IDL> test < 0
      -1       0       0      -4       0
    IDL> test < 0 > 3
       3       3       3       3       3
    IDL> test < 1 > 0
       0       1       1       0       1
    ```

    The bottom example is the most common example you'll see in `FHD` as it turns all numbers above 1 into 1, and the numbers below 0 into 0 allowing to quickly make a flagged array. To show the same examples in Python look below:

    ```python
    >>> test = np.array([-1,2,3,-4,5])
    >>> np.maximum(test, 0)
    array([0, 2, 3, 0, 5])
    >>> np.minimum(test, 0)
    array([-1,  0,  0, -4,  0])
    >>> np.maximum(np.minimum(test, 0), 3)
    array([3, 3, 3, 3, 3])
    >>> np.maximum(np.minimum(test, 1), 0)
    array([0, 1, 1, 0, 1])
    ```

I hope this contribution guide helps you in your translation efforts with much less pain than me, and all of this helps you get your function into Python as quick as possible. Translation and testing of IDL to Python code can be a frustrating task even at the best of times. So...

![Translators Ye Be Warned](ye_be_warned_smaller.gif)