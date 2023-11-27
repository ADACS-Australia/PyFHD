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

You have made your new function, congratulations, now it's time to add some documentation and comments if you haven't already. In PyFHD, we use docstrings in the [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html) which is well documented in terms of what you should put into your docstrings and what each section in the docstring is for. There is specific formatting to follow and deviations from the format will result in weird outputs for read the docs once the docstring is read in for auto generating the API reference so please follow the numpydoc format precisely.

When it comes to comments in PyFHD, there are some functions which are heavily commented and others are not, in general comments should be about the intention and the reason why the code exists, rather than what it does. It's often said good, clean code shouldn't need comments as it's descriptive enough to be followed, which for the most part is somewhat True, however, I personally don't think that's always True, and so having comments about what something does isn't necessarily a bad thing. The rule I like to follow is somewhere in between, comments are gifts to yourself when you have to re-read the same function a year later, if its got more comments than you need, who cares?

In PyFHD we've also using the relatively new, but already well established Python typing systems and packages. In PyFHD they have been primarily used in the definition of functions, to ensure the not only is the type of each variable visible, but in the case of NumPy typing, you can also show the expected precision. The reason for using the typing system is making the development experience better in IDE's like VSCode or PyCharm, as the IDE will show you the function as you're typing it out. It only takes an extra 2 seconds per parameter on the defintion of a new function but can save you hours of pain. Features inside IDE's such as code completion, code hinting usually come under the umbrella of [Intellisense](https://code.visualstudio.com/docs/editor/intellisense). In the future, it's possible to use tools such as [MyPy](https://mypy.readthedocs.io/en/stable/) to enforce these types too.

For examples of docstrings in PyFHD check out the following doc strings from the `get_ri` and `histogram` functions. 

```python
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
    pass

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
    pass
```

You should notice the typing system in use with the colon's after each parameter telling you what types the inputs should be. Using extensions for VSCode or PyCharm can allow you to automatically create the skeleton for the docstring such as [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) for VSCode. You'll notice as well when using extensions for the auto doc strings that the typing you have hopefully just implemented will be picked up automatically with little to no extra effort needed for formatting. 

Combining docstrings, comments and typing together when used well and best practices followed allows for easier, faster debugging in the future. Furthermore, it's easier to keep track of your changes and how they could affect the rest of the codebase. If that sounds good to you as it does to me, let's keep it up. 

### Testing the new function

In an ideal world, every new function you make has at least 3 tests, that's the general rule to follow. In this ideal world the tests cover corner cases as well as expected cases. It has been my experience so far in `PyFHD` that small simulated data seems to work well for testing these functions and usually gives a great indication that `PyFHD` will work with real data. With testing functions like `histogram` I had the benefit of creating as many tests as I wanted given I was replicating the IDL function, this is the best case scenario. With functions that were translated from `FHD`, in general we'd run the functions in `FHD` and save the input variables and the output variables. The `sav` files would then be read in by a testing function and generally converted to HDF5 for faster and easier reading in the future for the same test. 

For creating a test in `PyFHD` we have utilised the pytest framework which is a powerful testing framework that searches for any function and/or directory that has the word `test` at the start or end of the function name or directory name. A useful feature of pytest are `fixtures` which are heavily used throughout the testing of PyFHD. Fixtures allow you to make functions which perform routines needed for a function, in the case of `test_histogram.py` I used a fixture to spread the data directory across all functions without having to contunally copy and paste the path. When making a test for `PyFHD` the `numpy.testing` package was particualry useful as it contains functions like `np.testing.assert_allclose` which allow you to test the differences between arrays upto an absolute and/or relative precision. See below for a test I made for the histogram function which utilises a basic pytest fixture:

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

Pytest fixtures have the ability to group other fixtures too, take `test_cal_auto_ratio_divide` as an example, I set up multiple pytest fixtures with multiple parameters, and then used pytest fixtures to automatically generate groups of tests without duplicating code for each test. You should also notice the use of the `numpy.testing` library as well. The below code generates 6 tests using the fixtures `tag` and `run` primarily. This test also shows how you can skip certain groups if you need to.

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

It's testing that has enabled us to be sure that `PyFHD` actually does the same things as `FHD` and even in some cases detect bugs on the `FHD` side. Furthermore, once you have set up the tests, evry time you change the code and re-run the test you can be sure you haven't broken existing functionality accidentally giving you more confidence that the changes you have made will improve `PyFHD`. This section wasn't about teaching you how to test, but to show why we do it. If you want to learn more about testing check out the numpy testing functions [here](https://numpy.org/doc/stable/reference/routines.testing.html) and also check out pytest [here](https://docs.pytest.org/en/7.4.x/).

### The Need for Speed

### Adding your new function to the Changelog

### Pull Request for your new function

## IDL to Python Translation Guide

