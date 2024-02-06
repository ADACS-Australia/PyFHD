import numpy as np
import importlib_resources
import os


def retrieve_gitdict():
    """
    Attempt to recover the git information that was created during pip install.
    If found, return the git dictionary. If not, return False

    Returns
    -------
    git_dict : dictionary
        A dictionary containing git information with keywords: describe, date,
        branch

    """

    file_path = importlib_resources.files("PyFHD").joinpath("PyFHD_gitinfo.txt")

    # If things have been pip installed correctly
    if os.path.isfile(file_path):
        with open(file_path) as infile:
            lines = infile.read().split("\n")

            git_dict = {
                "describe": lines[0],
                "date": lines[1],
                "branch": lines[2],
            }
    else:
        git_dict = False

    return git_dict
