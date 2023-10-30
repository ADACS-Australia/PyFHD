import numpy as np
import h5py
from logging import RootLogger
from pathlib import Path

def dtype_picker(dtype: type) -> type:
    """
    Picks the double precision type for the given dtype for saving the hdf5 file to ensure everything
    is saved without losing information.

    Parameters
    ----------
    dtype : type
        The numpy dtype of an array

    Returns
    -------
    type
        The corresponding double precision type
    """
    if np.issubdtype(dtype, np.integer):
        return np.int64
    elif np.issubdtype(dtype, np.floating):
        return np.float64
    elif np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        # Should never get here, this should throw an error
        return None

def dict_to_group(group: h5py.Group, to_convert: dict, logger: RootLogger) -> None:
    """
    Converts a dictionary to a HDF5 group. This is called in the event a dictionary is found inside
    a dictionary that is being saved in a HDF5 file. Creates a subgroup for the hdf5 file with everything
    turning into individual datasets.

    Parameters
    ----------
    group : h5py.Group
        The created group to save the dictionary in
    to_convert : dict
        The dictionary to save into the group
    logger : RootLogger
       PyFHD's Logger
    """
    for key in to_convert:
        match to_convert[key]:
            case dict():
                subgroup = group.create_group(key)
                dict_to_group(subgroup, to_convert[key])
            case np.ndarray():
                group.create_dataset(key, shape = to_convert[key].shape, data = to_convert[key], dtype = dtype_picker(to_convert[key].dtype), compression = 'gzip')
            case list():
                group.create_dataset(key, shape = len(to_convert[key]), data = to_convert[key], compression = 'gzip')
            case _:
                try:
                    group.create_dataset(key, shape = (1), data = to_convert[key])
                except ValueError:
                    logger.error(f"key, type(to_convert[key])")

# Create the save function here
def save(file_name: Path, to_save: np.ndarray | dict, dataset_name: str, logger: RootLogger, to_chunk: dict[str, dict] = {}) -> None:
    """
    Saves a numpy array or dictionary into a hdf5 file using h5py, with compression applied to all arrays/datasets. 
    An array will be saved as a single dataset, while a dictionary will be saved where each key will be a dataset
    unless the key points a dictionary in which case a group will be created and `dict_to_group` called to turn each
    key in that sub dict into a dataset (or another group if it's another sub dictionary). This function should be
    kept as general as possible, if something needs formatting for saving, format it before calling this function.
    If you are converting a sav file to hdf5 with this function, use `recarray_to_dict` which converts the sav output
    from readsav into a proper python dictionary (rather than recarrays or weird array shapes, objects arrays etc.)

    Parameters
    ----------
    file_name : Path
        The file to save as hdf5 should be /path/to/file_name.h5 (or .hdf5)
    to_save : np.ndarray | dict
        The dictionary or numpy array to save into the hdf5 file
    dataset_name : str
        Used in the case that the to_save variable is an array, this name will 
        be used as the key for the dataset in the hdf5 file.
    logger : RootLogger
        PyFHD's Logger
    to_chunk : dict[str, dict], optional
        A dictionary where each key-value pair represents a key in the to_save dictionary, and the value is a dictionary
        which should contain two key-value pairs, `shape` which should be the `shape` of the array and `chunk` which tells
        hdf5 how to chunk the dataset when it's being read/written. If you're not sure how to `chunk` the dataset, set `chunk` 
        to True which enables h5py to guess the chunk size for you. By default {}

    See Also
    --------
    PyFHD.io.pyfhd_io.load : Load a HDF5 file
    """
    with h5py.File(file_name, "w") as h5_file:
        match to_save:
            case np.ndarray():
                logger.info(f"Writing the {dataset_name} array to {file_name}.h5")
                h5_file.create_dataset(dataset_name, to_save.shape, data = to_save, dtype = dtype_picker(to_save.dtype), compression = 'gzip')
            case dict():
                logger.info(f"Writing the {dataset_name} dict to {file_name}.h5, each key will be a dataset, if the key contains a dict then it will be a group.")
                for key in to_save:
                    match to_save[key]:
                        case dict():
                            group = h5_file.create_group(key)
                            dict_to_group(group, to_save[key], logger)
                        case np.ndarray():
                            if key in to_chunk:
                                h5_file.create_dataset(key, shape=to_chunk[key]['shape'], data = to_save[key], dtype = dtype_picker(to_save[key].dtype), chunks = to_chunk[key]['chunk'], compression = 'gzip')
                            else:
                                h5_file.create_dataset(key, shape = to_save[key].shape, data = to_save[key], dtype = dtype_picker(to_save[key].dtype), compression = 'gzip')
                        case list():
                            h5_file.create_dataset(key, shape = len(to_save[key]), data = to_save[key])
                        case _:
                            try:
                                h5_file.create_dataset(key, shape = (1), data = to_save[key])
                            except ValueError:
                                print(key, type(to_save[key]))
            case _:
                logger.warning("Not a dict or numpy array, PyFHD won't write other types at this time, refer to PyFHD.io.pyfhd_io.save to see what is supported")

def group_to_dict(group: h5py.Group) -> dict:
    """
    When loading a hdf5 file into a dictionary, this turns a group into a dictionary, 
    and then returns the dictionary.

    Parameters
    ----------
    group : h5py.Group
        A h5py group to turn into a dictionary

    Returns
    -------
    return_dict: dict
        The group turned into a dictionary
    """
    return_dict = {}
    for key in group:
        match group[key]:
            case h5py.Dataset():
                return_dict[key] = group[key][:]
                if isinstance(return_dict[key], np.ndarray) and return_dict[key].size == 1:
                    return_dict[key] = return_dict[key][0]
            case h5py.Group():
                return_dict[key] = group_to_dict(group[key])
    return return_dict

# Create the load function
def load(file_name: Path, logger: RootLogger, lazy_load: bool = False) -> dict | np.ndarray | h5py.File:
    """
    Loads a HDF5 file into PyFHD, it reads the HDF5 into an array if the 
    HDF5 file contains a single dataset, while a HDF5 which contains multiple
    datasets will load them into a dictionary. 

    Parameters
    ----------
    file_name : Path
        The /path/to/the/hdf5.h5
    logger : RootLogger
        PyFHD's Logger
    lazy_load : bool, optional
        Set to true if you wish to lazy load the file, currently the only file that will be
        supported to do this in PyFHD will be the beam/psf file, but support for other files can
        be done easily enough, by default False
        

    Returns
    -------
    return_dict | array | h5_file: dict | np.ndarray | h5py.File
        Returns a dict in the case the HDF5 file contains multple datasets, 
        An array if the HDF5 contains one dataset or h5py File object if the 
        file is lazy loaded to conserve memory. 
    """
    h5_file = h5py.File(file_name, "r")
    if lazy_load:
        return h5_file
    try:
        if (len(h5_file.keys()) == 1):
            # Assume that it contains only one numpy array, in which case read the array
            key = list(h5_file.keys())[0]
            logger.info(f"Loading {key} from {file_name} into an array")
            array = h5_file[key][:]
            return array
        else:
            return_dict = {}
            logger.info(f"Loading {file_name} into a dictionary")
            for key in h5_file:
                match h5_file[key]:
                    case h5py.Dataset():
                        return_dict[key] = h5_file[key][:]
                        if isinstance(return_dict[key], np.ndarray) and return_dict[key].size == 1:
                            return_dict[key] = return_dict[key][0]
                    case h5py.Group():
                        return_dict[key] = group_to_dict(h5_file[key])
            return return_dict
    finally:
        if not lazy_load:
            h5_file.close()


def recarray_to_dict(data: np.recarray | dict) -> dict:
    """
    Turns a record array into a dict, but does it as a deep convert. This was needed due to scipy's readsav
    returning an inception like experience of record arrays. This would mean to access values from something 
    like the obs structure for a test, the code had to be obs[0]['baseline_info'][0]['tile_a'], which was became 
    untenable as the full python translation won't require these leaving us two codebases for IDL compatible and
    Python compatible. Instead, this function turns all record arrays into dictionaries, which are easier to understand
    and are faster.

    This was made specifically to work with the readsav function, to get compatibility with general recarrays remove the
    zero index, as readsav for some reason adds a single dimension to all recarrays.

    This was updated later to also take a dictionary which may contain record arrays too.

    This was also updated later to turn object arrays into multidimensional arrays if they can be one. In the
    case the object array couldn't be turned into a multidimensional array it was turned into a list

    Parameters
    ----------
    data : np.recarray or dict
        A record array or dictionary maybe containing nested record arrays

    Returns
    -------
    data: dict
        A potentially nested dictionaries of dictionaries
    """
    # Convert the original record array into a dictionary
    if (type(data) == np.recarray):
        data = {name.lower():data[name] for name in data.dtype.names}
    # For every key, if it's a record array, recursively call the function
    for key in data:
        # Every now and then you do get object arrays that contain only one element or arrays that contain only one element
        # These are not useful so I will extract the element out
        if (type(data[key]) == np.ndarray and data[key].size == 1):
            data[key] = data[key][0]
        # Sometimes the recarray is in a standard numpy object array and other times its not for some reason...
        if (type(data[key]) == np.recarray):
            data[key] = recarray_to_dict(data[key])
        elif (type(data[key]) == np.ndarray and type(data[key][0]) == np.recarray):
            data[key] = recarray_to_dict(data[key][0])
        # You can also get object arrays which themselves contain numpy arrays, it's best to turn these
        # into multidimensional arrays. If they can't turn into multidimensional arrays due to them 
        # being different types or not of the same size then it will convert the numpy object array 
        # into a list of objects instead.
        elif (type(data[key]) == np.ndarray and data[key].dtype == object and type(data[key][0]) == np.ndarray):
            try:
                # If it's not an object array, numpy will stack the axes, which isn't desired here
                # as we want to maintain the multidimensional nature of the data. So we'll create an 
                # array of the desired size using the shape of the first element.
                if (data[key][0].dtype != object):
                    new_array = np.empty([data[key].size, *data[key][0].shape], dtype = data[key][0].dtype)
                    for idx in range(new_array.shape[0]):
                        new_array[idx] = data[key][idx]
                    data[key] = new_array
                else:
                    # For an object array you can flatten it, and stack all inner arrays together until it's not an object array
                    # Crucially this assumes the array not as an object array can fit in memory! If you're doing the beam_ptr
                    # conversion take this into consideration
                    while data[key].dtype == object:
                        data[key] = np.vstack(data[key].flatten()).reshape(list(data[key].shape) + list(data[key].flat[0].shape))
            except ValueError:
                data[key] = list(x for x in data[key])
    return data