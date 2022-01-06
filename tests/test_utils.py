import numpy as np
from scipy.io import readsav
from pathlib import Path

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

