import numpy as np
import h5py
from logging import Logger
from pathlib import Path
from typing import Any

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

@np.vectorize
def _is_complex(value: Any) -> bool:
    """
    Finds if a value is complex, this works regardless of the array type
    unlike np.iscomplex or np.iscomplexobj which can't handle object arrays. 
    This being vectorized also allows us to check this for any complex type, 
    whether it be the python complex type or a numpy complex type

    Parameters
    ----------
    value : Any
        The value to check in a NumPy array

    Returns
    -------
    bool
        True if value is a complex, False otherwise
    """
    return np.iscomplexobj(value)

@np.vectorize
def _is_string(value: Any) -> bool:
    """
    Finds if a value is a string or not, works regardless of the array type.
    There is no string check available for object arrays

    Parameters
    ----------
    value : Any
        A value to check

    Returns
    -------
    bool
        True if value is a str, False otherwise
    """
    return isinstance(value, str)

@np.vectorize
def _is_none(value: Any) -> bool:
    """
    Checks for a none object and is vectorized to work across any numpy array
    even if it's an object array.

    Parameters
    ----------
    value : Any
        A value to be checked if None

    Returns
    -------
    bool
        True if value is None, otherwise False
    """
    return value is None

@np.vectorize
def _decode_byte_arr(value: np.bytes_) -> str:
    """
    Decodes a byte string into a string

    Parameters
    ----------
    value : np.bytes_
        Value to decode

    Returns
    -------
    str
        The decoded value
    """
    return value.decode()

def format_array(array: np.ndarray) -> np.ndarray:
    """
    Find any `None` values in an object array and replaces them with empty
    strings if we're dealing with a string array, or `NaN`s if we're 
    dealing with a Number array. If complex, the NaN will be nan + nanj.
    If a string array is found, convert the string array to a bytes array,
    in all other cases leave the array alone as it should be ready to save
    into a HDF5 file.

    Parameters
    ----------
    array : np.ndarray
        The array to find None in and if so convert from object array

    Returns
    -------
    array: np.ndarray
        Array without None objects and in the correct dtype
    """
    # Got an error with the vectorized functions on empty arrays
    if array.size == 0 or array.dtype != object:
        if np.issubdtype(array.dtype, np.str_):
            return array.astype(np.bytes_)
        else:
            return array
    if np.any(_is_string(array)):
        # This avoids the np.where deprecation warning
        # Also replaces any None values in place, no copies of the array are made
        array[array == None] = '' 
        array = array.astype(bytes)
    else:
        try:
            if array.dtype == object:
                # Replace any Nones with NaN's in place, no copies made
                array[array == None] = np.nan
                if np.any(_is_complex(array)):
                    # Set the type to complex128 to be sure its double precision complex
                    array = array.astype(np.complex128)
                    # Replace with complex NaNs in place
                    array[np.isnan(array.real)] = np.nan*0j
                else:
                    # Ensure it's a float array if we do have
                    array = array.astype(np.float64)
        except TypeError:
            # Sometimes we deal with structured/record arrays like
            # astropy's FITS_rec, let's leave them alone as we intend 
            # on saving them raw
            pass
    return array

def save_dataset(h5py_obj: h5py.File | h5py.Group,  key: str, value: Any, to_chunk: dict[str, dict], logger: Logger | None) -> bool:
    """
    A general function for saving a dataset inside a HDF5 File or Group. It's used exclusively for saving
    a dictionary into a HDF5 file, hence why we take a `key` and `value` pair. The `to_chunk` parameter is
    explained in the `save` function, please look there for explanation. In the case of finding a None object
    an Empty Dataset is saved and the is_none is returned as True, so the attribute associated with the key
    can also be set to True to indicate to PyFHD later that the value is meant to be None when reading in the 
    dataset again.

    Parameters
    ----------
    h5py_obj : h5py.File | h5py.Group
        A h5py object that has access to the `create_dataset` and `create_group` methods
    key : str
        The key from the dictionary we're saving
    value : Any
        The value from the dictionary
    to_chunk : dict[str, dict]
        A dictionary where each key-value pair represents a key in the to_save dictionary, and the value is a dictionary
        which should contain two key-value pairs, `shape` which should be the `shape` of the array and `chunk` which tells
        hdf5 how to chunk the dataset when it's being read/written. If you're not sure how to `chunk` the dataset, set `chunk` 
        to True which enables h5py to guess the chunk size for you. By default {}
    logger : Logger | None
        PyFHD's Logger

    Returns
    -------
    is_none : bool
        True if the value is None, False otherwise

    See Also
    --------
    PyFHD.io.pyfhd_io.save : Save a HDF5 file
    PyFHD.io.pyfhd_io.dict_to_group : Converts a dictionary to a h5py Group Object
    """
    is_none = False
    # Match the type
    match value:
        case dict():
            group = h5py_obj.create_group(key)
            # dict_to_group will be recursively called if there is another dict
            # in this dict
            dict_to_group(group, value, to_chunk, logger)
        case np.ndarray():
            # Find and replace all None objects
            value = format_array(value)
            # If we want it to be chunked do that, always compress it
            if key in to_chunk:
                h5py_obj.create_dataset(key, shape=to_chunk[key]['shape'], data = value, dtype = dtype_picker(value.dtype), chunks = to_chunk[key]['chunk'], compression = 'gzip')
            else:
                h5py_obj.create_dataset(key, shape = value.shape, data = value, dtype = dtype_picker(value.dtype), compression = 'gzip')
        case list():
            # Was easier to convert to a NumPy array to get vectorization
            value = np.array(value)
            value = format_array(value)
            h5py_obj.create_dataset(key, data = value)
        case Path():
            # If we find a Path object, convert it to a string
            value = str(value)
            h5py_obj.create_dataset(key, shape = (1), data = value)
        case None:
            is_none = True
            # In the case we get something that is none, create empty dataset
            h5py_obj.create_dataset(key, dtype = "b")
        case _:
            try:
                # Store the value in a single size dataset, used for ints, floats, strings etc
                h5py_obj.create_dataset(key, shape = (1), data = value)
            except ValueError:
                if logger is not None:
                    logger.error(f"Failed to save {key}, the type of key was {type(value)}")
    return is_none

def dict_to_group(group: h5py.Group, to_convert: dict, to_chunk: dict[str, dict], logger: Logger | None) -> None:
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
    logger : Logger
       PyFHD's Logger
    """
    for key in to_convert:
        group.attrs[key] = save_dataset(group, key, to_convert[key], to_chunk, logger)

def save(file_name: Path, to_save: np.ndarray | dict, dataset_name: str, logger: Logger | None = None, to_chunk: dict[str, dict] = {}) -> None:
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
    logger : Logger, optional
        PyFHD's Logger, by default None (in case you don't want to use the logger for testing)
    to_chunk : dict[str, dict], optional
        A dictionary where each key-value pair represents a key in the to_save dictionary, and the value is a dictionary
        which should contain two key-value pairs, `shape` which should be the `shape` of the array and `chunk` which tells
        hdf5 how to chunk the dataset when it's being read/written. If you're not sure how to `chunk` the dataset, set `chunk` 
        to True which enables h5py to guess the chunk size for you. By default {}

    See Also
    --------
    PyFHD.io.pyfhd_io.load : Load a HDF5 file
    PyFHD.io.pyfhd_io.dict_to_group : Converts a dictionary to a h5py Group Object
    PyFHD.io.pyfhd_io.recarray_to_dict : Turns any record arrays into dicts, also formats object arrays into the correct dtype array
    PyFHD.io.pyfhd_io.save_dataset : Saves a single dataset based off a dictionary key-value pair
    PyFHD.io.pyfhd_io.format_array : Finds any None is an array and replaces them appropriately
    """
    # Create a custom vectorized function to check for complex numbers
    # is_complex_vectorized = np.vectorize(is_complex)
    with h5py.File(file_name, "w") as h5_file:
        match to_save:
            case np.ndarray():
                if logger:
                    logger.info(f"Writing the {dataset_name} array to {file_name}.h5")
                h5_file.create_dataset(dataset_name, to_save.shape, data = to_save, dtype = dtype_picker(to_save.dtype), compression = 'gzip')
                h5_file.attrs[dataset_name] = False
            case dict():
                if logger:
                    logger.info(f"Writing the {dataset_name} dict to {file_name}, each key will be a dataset, if the key contains a dict then it will be a group.")
                for key in to_save:
                    # We're using the attributes as a mask, where if True then we know
                    # the dataset is representing a None object. 
                    h5_file.attrs[key] = save_dataset(h5_file, key, to_save[key], to_chunk, logger)
            case _:
                if logger:
                    logger.warning("Not a dict or numpy array, PyFHD won't write other types at this time, refer to PyFHD.io.pyfhd_io.save to see what is supported")

def load_dataset(h5py_obj: h5py.File | h5py.Group, key: str, dataset: h5py.Dataset) -> Any:
    """
    Loads a single dataset from a HDF5 File or Group, the key here is the dataset name from the
    file or group and is only used to check the attributes of said file or group. If the attribute
    associated with the key is True, then we assume the value saved is an empty dataset and we should
    return None. If this is False, load the value and check if this value should be a single value. There
    are special checks for byte arrays, if there is byte arrays, PyFHD assumes these are meant to be strings.

    Parameters
    ----------
    h5py_obj : h5py.File | h5py.Group
        A HDF5 file or group
    key : str
        The dataset name
    dataset : h5py.Dataset
        The dataset we are loading

    Returns
    -------
    Any
        The value stored in the HDF5 Dataset

    See Also
    --------
    PyFHD.io.pyfhd_io.load : Load a HDF5 file
    """
    # If the corresponding attribute is True set the current
    # key to None as its an empty dataset
    if h5py_obj.attrs[key]:
        return None
    else:
        value = dataset[:]
        if value.dtype.kind == 'S':
            value = _decode_byte_arr(value)
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode()
        return value

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
                return_dict[key] = load_dataset(group, key, group[key])
            case h5py.Group():
                return_dict[key] = group_to_dict(group[key])
    return return_dict

def load(file_name: Path, logger: Logger | None = None, lazy_load: bool = False) -> dict | np.ndarray | h5py.File:
    """
    Loads a HDF5 file into PyFHD, it reads the HDF5 into an array if the 
    HDF5 file contains a single dataset, while a HDF5 which contains multiple
    datasets will load them into a dictionary. Any groups will be convered to
    sub dictionaries using `group_to_dict` 

    Parameters
    ----------
    file_name : Path
        The /path/to/the/hdf5.h5
    logger : Logger
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

    See Also
    --------
    PyFHD.io.pyfhd_io.save : Save a HDF5 file
    PyFHD.io.pyfhd_io.group_to_dict : Converts a h5py Group object to a dictionary
    """
    h5_file = h5py.File(file_name, "r")
    if lazy_load:
        return h5_file
    try:
        if (len(h5_file.keys()) == 1):
            # Assume that it contains only one numpy array, in which case read the array
            key = list(h5_file.keys())[0]
            if logger:
                logger.info(f"Loading {key} from {file_name} into an array")
            array = h5_file[key][:]
            return array
        else:
            return_dict = {}
            if logger:
                logger.info(f"Loading {file_name} into a dictionary")
            for key in h5_file:
                match h5_file[key]:
                    case h5py.Dataset():
                        return_dict[key] = load_dataset(h5_file, key, h5_file[key])
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
        # We found a single array with only None
        elif (type(data[key]) == np.ndarray and isinstance(data[key][0], type(None))):
            # Get all the None values and turn them into NaNs
            none_values = np.where(data[key] == None)
            if np.size(none_values) > 0:
                data[key][none_values] = np.nan
            # If all of the values were None, then set the array dtype to float64 
            # (as we don't know what dtype it actually was), probably only relevant for testing
            if np.size(none_values) == np.size(data[key]):
                data[key] = data[key].astype(np.float64)
        # Assume we found a string array since it's bytes, convert to a string list
        elif (type(data[key]) == np.ndarray and isinstance(data[key].flat[0], bytes)):
            data[key] = [x.decode().strip() for x in data[key]]
        # Found only bytes, assume it's a string, convert the string
        elif isinstance(data[key], bytes):
            data[key] = data[key].decode()
        # You can also get object arrays which themselves contain numpy arrays, it's best to turn these
        # into multidimensional arrays. If they can't turn into multidimensional arrays due to them 
        # being different types or not of the same size then it will convert the numpy object array 
        # into a list of objects instead.
        elif (type(data[key]) == np.ndarray and data[key].dtype == object and type(data[key][0]) == np.ndarray):
            try:
                # Get all the None values and turn them into NaNs
                none_values = np.nonzero(_is_none(data[key]))
                if np.size(none_values) > 0:
                    data[key][none_values] = np.nan
                # If all of the values were None, then set the array dtype to float64 
                # (as we don't know what dtype it actually was), probably only relevant for testing
                if (np.size(none_values) // len(data[key].shape)) == np.size(data[key]):
                    data[key] = data[key].astype(np.float64)
                # If it's not an object array, numpy will stack the axes, which isn't desired here
                # as we want to maintain the multidimensional nature of the data. So we'll create an 
                # array of the desired size using the shape of the first element.
                elif (data[key][0].dtype != object):
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