from PyFHD.io.pyfhd_io import recarray_to_dict
import numpy as np
from scipy.io import readsav
from pathlib import Path
import numpy.testing as npt
from colorama import Fore
from colorama import Style
from PyFHD.io.pyfhd_io import save

def get_data(data_dir, data_filename, *args):
    """
    This function is designed to read npy or sav files in a 
    data directory inside test_fhd_*. Ensure the data file
    has been made with the scripts inside the scripts directory.
    Use splitter.py to put the files and directories in the right 
    format if you have used histogram runner and rebin runner.
    Paths are expected to be of data_dir/data/function_name/[data,expected]_filename.npy
    data_dir is given by pytest-datadir, it should be the directory where the test file is in.

    Parameters
    ----------
    data_dir : Path
        This should be the dir passed through from pytest-datadir
    function_name : String
        The name of the function we're testing
    data_filename : String
        The name of the file for the input
    expected_filename : String
        The name of the file name for the expected result
    *args : List
        If given, is expected to be more filenames
    
    Returns
    -------
    input : 
        The data used for input of the function being tested
    expected : 
        The expected result of the function
    """
    # Put as Paths and read the files
    input_path = Path(data_dir, data_filename)
    if input_path.suffix == '.sav':
        input = readsav(input_path, python_dict=True)
    else:
        input = np.load(input_path, allow_pickle=True)
    if len(args) > 0:
        return_list = [input]
        for file in args:
            path = Path(data_dir, file)
            if path.suffix == '.sav':
                output = readsav(path, python_dict=True)
            else:
                output = np.load(path, allow_pickle=True)
            return_list.append(output)
        return return_list
    # Return the input and expected
    return input
    

def get_data_items(data_dir, data_with_item_path, *args):
    """
    Takes all the path inputs from tests and processes them so they're ready for use.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory
    data_with_item_path : Path
        Path to the data that contains only an item
    *args : Paths
        Give more paths to more data with items that need to be extracted

    Returns
    -------
    return_list
        Variable(s) required to do the test
    """
    # Retrieve the files and their contents
    data = get_data(data_dir, data_with_item_path)
    # Get the key, then use the key to get the item
    key = list(data.item().keys())[0]
    item = data.item().get(key)
    # Process the args list if there is one
    if len(args) > 0:
        # Add to return_list
        return_list = [item]
        for path in args:
            data = get_data(data_dir, path)
            key = list(data.item().keys())[0]
            item_in_data = data.item().get(key)
            return_list.append(item_in_data)
        return return_list
    #Return them
    return item

def get_data_sav(data_dir, sav_file, *args):
    """
    Takes all the path inputs from tests and processes them so they're ready for use.
    More specifically takes in sav files

    Parameters
    ----------
    data_dir : Path
        Path to the data directory
    sav_file : Path
        Path to the sav file, which will load a python dictionary
    args: Paths
        If given, is expected to be more filenames
    """
    data = get_data(data_dir, sav_file)
    key = list(data.keys())[0]
    data = data[key]
    if len(args) > 0:
        # Add to return_list
        return_list = [data]
        for path in args:
            data = get_data(data_dir, path)
            key = list(data.keys())[0]
            data = data[key]
            return_list.append(data)
        return return_list
    return data

def get_savs(data_dir, sav_file, *args):
    """
    Takes in the path for many sav files and reads them without
    reading their keys. Assumes the sav files here have more than one key.
    If you use one sav_path only then the function acts as a wrapper for scipy's readsav.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory
    sav_file : Path
        Path to the sav file, which will load a python dictionary
    args: Paths
        If given, is expected to be more filenames
    """
    data = readsav(Path(data_dir, sav_file), python_dict=True)
    if len(args) > 0:
        data = [data]
        for file in args:
            new_data = readsav(Path(data_dir, file), python_dict=True)
            data.append(new_data)
    return data

def try_assert_all_close(actual : np.ndarray, target : np.ndarray, name : str, tolerance = 1e-8):
    """
    Uses the numpy testing assert_all_close but uses a try and except wrapper around it to print
    the error instead of doing an AssertionError which stops the running of the program. This is helpful
    when doing testing with expected precision errors, but wanting to avoid stopping the program or constantly 
    setting the tolerances on multiple assert statements.

    Parameters
    ----------
    actual : np.ndarray
        The array we calculated
    target : np.ndarray
        The array we actually want to calculate
    name : str
        The name of the variable we are testing
    tolerance : float, optional
        This is the tolerance for the error in absolute values, by default 1e-8
    """
    try :
        npt.assert_allclose(actual, target, atol = tolerance)
        print(Fore.GREEN + Style.BRIGHT + "Test Passed for {}".format(name) + Style.RESET_ALL)
    except AssertionError as error:
        print(Fore.RED + Style.BRIGHT + "Test Failed for {}:".format(name) + Style.RESET_ALL + "{}".format(error) + Style.RESET_ALL)

def convert_to_h5(test_path: Path, save_path: Path, *args) -> None:
    """
    For every file specified as an arg, read the file from the test_path into a python dictionary.
    If it's a dict or recarray that contaisn recarrays, convert all the recarrays using recarray_to_dict.
    The files can be .npy or .sav files. The python dict will then be written into a HDF5 file for testing 
    purposes.

    This function was made to convert many of the .npy and .sav files into something that can be read and written more
    easily by other packages other than numpyt or scipy.

    Parameters
    ----------
    test_path : Path
        The path to a directory with all the files inside it
    save_path : Path
        The path to the file for saving the HDF5
    *args : list[str]
        A list of file names to be read in, can be .npy or .sav files
    """
    to_save = {}
    # Process the file differently depending on whether its IDL or numpy files
    for file in args:
        if (file.endswith('.sav')):
            var = readsav(Path(test_path, file), python_dict = True)
            # Convert to nested dictionaries
            var = recarray_to_dict(var)
        elif(file.endswith('.npy')):
            var = np.load(Path(test_path, file), allow_pickle=True).item()
        for key in var:
            to_save[key] = var[key]
    save(save_path, to_save, "to_save")

def sav_file_vis_arr_swap_axes(sav_file_vis_arr : np.ndarray):
    """After saving arrays from IDL like `vis_arr` and `vis_model_arr` into
    and IDL .sav file, and subsequently loading in via scipy.io.readsav,
    they come out in a shape/format unsuitable for PyFHD. Use this function
    to reshape into shape = (n_pol, n_freq, n_baselines)

    Parameters
    ----------
    sav_file_vis_arr : np.ndarray
        Array as read in by scipy.io.readsav, if `n_pol = 2` should have `shape=(2,)`

    Returns
    -------
    np.ndarray
        Returns the array with `shape=(n_pol, n_freq, n_baselines)`
    """

    n_pol = sav_file_vis_arr.shape[0]

    vis_arr = np.empty((n_pol, sav_file_vis_arr[0].shape[1],
                               sav_file_vis_arr[0].shape[0]),
                               dtype=sav_file_vis_arr[0].dtype)

    for pol in range(n_pol):
        vis_arr[pol, :, :] = sav_file_vis_arr[pol].transpose()

    return vis_arr

def print_types(dictionary, dict_name, indent_level = 1):
    """
    When generating the tests, I'd find it useful to see the types of all the keys and value pairs inside
    the dictionary I'm manipulating. The Debug mode is helpful for this too, but this can be easily used 
    inside a notebook if experimenting in there too. 

    Parameters
    ----------
    dictionary : dict
        The dictionary to print the types of
    dict_name : string
        The name of the dict
    indent_level : int
        Sets the indent levels for printing as it's a recursive function, by default 1
    """
    for key in dictionary.keys():
        # Print this if it's a NumPy array
        if type(dictionary[key]) == np.ndarray:
            print(f"{dict_name}[{key}] : {dictionary[key].dtype} {dictionary[key].shape}\n{indent_level * 2 * ' '}Inside Type: {type(dictionary[key][0])}")
            if type(dictionary[key][0]) == np.ndarray:
                print(f"{indent_level * 2 * ' '}NumPy Array Dtype: {dictionary[key][0].dtype}")
        # Recursively call the function on another sub dict
        elif type(dictionary[key]) == dict:
            print(f"{dict_name}[{key}]  : {type(dictionary[key])}")
            print_types(dictionary[key], dict_name=f"  {key}", indent_level=indent_level + 2)
        # If it's an object, might be useful to print the value
        elif type(dictionary[key]) == object:
            print(f"{dict_name}[{key}]  : {type(dictionary[key])}")
            print(dictionary[key])
        # Otherwise just print it out
        else:
            print(f"{dict_name}[{key}]  : {type(dictionary[key])}")