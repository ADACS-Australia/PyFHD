import numpy as np
import h5py
from logging import RootLogger
from pathlib import Path

def dtype_picker(dtype: type) -> type:
    """
    TODO: _summary_

    Parameters
    ----------
    dtype : type
        _description_

    Returns
    -------
    type
        _description_
    """
    if np.issubdtype(dtype, np.integer):
        return np.int64
    elif np.issubdtype(dtype, np.floating):
        return np.float64
    elif np.issubdtype(dtype, np.complexfloating):
        return np.complex128

def dict_to_group(group: h5py.Group, to_convert: dict, logger: RootLogger) -> None:
    """
    TODO: _summary_

    Parameters
    ----------
    group : h5py.Group
        _description_
    to_convert : dict
        _description_
    logger : RootLogger
        _description_
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
    TODO: _summary_

    Parameters
    ----------
    file_name : Path
        _description_
    to_save : np.ndarray | dict
        _description_
    dataset_name : str
        _description_
    logger : RootLogger
        _description_
    to_chunk : dict[str, dict], optional
        _description_, by default {}
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
    TODO: _summary_

    Parameters
    ----------
    group : h5py.Group
        _description_

    Returns
    -------
    dict
        _description_
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
def load(file_name: Path, logger: RootLogger):
    # TODO: Add the ability to return certain things as lazy loaded 
    h5_file = h5py.File(file_name, "r")
    try:
        if (len(h5_file.keys()) == 1):
            # Assume that it contains only one numpy array, in which case read the array
            key = list(h5_file.keys())[0]
            logger.info(f"Loading {key} from {file_name}")
            array = h5_file[key][:]
            return array
        else:
            return_dict = {}
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
        h5_file.close()