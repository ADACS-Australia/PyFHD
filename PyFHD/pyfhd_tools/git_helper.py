from subprocess import check_output
import numpy as np
import importlib_resources
import os

def get_commandline_output(command_list):
    """
    Takes a command line entry separated into list entries, and returns the
    output from the command line as a string

    Parameters
    ----------
    command_list : list of strings
        list of strings that when combined form a coherent command to input into
        the command line

    Returns
    -------
    output : string
        the output result of running the command

    """
    output = check_output(command_list,universal_newlines=True).strip()
    return output

def make_gitdict():
    """
    Makes a dictionary containing key git information about the repo by running
    specific commands on the command line

    Returns
    -------
    git_dict : dictionary
        A dictionary containing git information with keywords: describe, date,
        branch

    """

    git_dict = {
        'describe': get_commandline_output(["git", "describe", "--always"]),
        'date': get_commandline_output(["git", "log", "-1", "--format=%cd"]),
        'branch': get_commandline_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }

    return git_dict

def retrieve_gitdict():
    """
    Attempt to recover the git information that was created during pip install. If found, return the git dictionary. If not, return False

    Returns
    -------
    git_dict : dictionary
        A dictionary containing git information with keywords: describe, date,
        branch

    """

    file_path = importlib_resources.files('PyFHD').joinpath('PyFHD_gitinfo.npz')

    ##If things have been pip installed correctly
    if os.path.isfile(file_path):
        git_dict = np.load(file_path, allow_pickle=True)
    else:
        git_dict = False

    return git_dict
