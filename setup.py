from setuptools import setup
import setuptools
import numpy as np
from PyFHD.pyfhd_tools.git_helper import make_gitdict
import os
from setuptools.command.build_py import build_py as _build_py

class GitInfo(setuptools.Command):
  '''A custom command to create a json file containing PyFHD git information.'''

  description = 'Create the file "PyFHD/PyFHD_gitinfo.json" containing git information '
  user_options = []

  def initialize_options(self):
    '''Set default values for options (this has to be included for
    setuptools.Command to work)'''
    # Each user option must be listed here with their default value.
    self.git_info = True

  def finalize_options(self):
    '''Post-process options (this has to be included for
    setuptools.Command to work)'''
    if self.git_info:
        print('Creating file PyFHD/PyFHD_gitinfo.npz')

  def run(self):
    '''Write the PyFHD git npz file.'''

    ##Find where we are running the pip install from, and add in a sensible
    ##place to save the git dictionary
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'PyFHD', 'PyFHD_gitinfo.npz')

    git_dict = make_gitdict()
    np.savez(save_path, **git_dict)


class BuildPyCommand(_build_py):
  '''Custom build command to run the gitinfo command during build'''

  def run(self):
    self.run_command('gitinfo')
    _build_py.run(self)

setup(
    name = "PyFHD",
    version = '1.0',
    author = "ADACS - Astronomy Data and Computing Services",
    url = "https://github.com/ADACS-Australia/PyFHD",
    python_requires=">=3.7",
    packages = ['PyFHD',
                'PyFHD.beam_setup',
                'PyFHD.calibration',
                'PyFHD.data_setup',
                'PyFHD.gridding',
                'PyFHD.pyfhd_tools',
                'PyFHD.use_idl_fhd'],
    description = 'Python Fast Holograhic Deconvolution: A Python package that does efficient image deconvolution for general radio astronomy, fast-mode Epoch of Reionization analysis, and simulation.',
    long_description = open("README.md").read(),
    long_description_content_type = 'text/markdown',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={'gitinfo': GitInfo,
              'build_py': BuildPyCommand,
              },
    entry_points = {
        'console_scripts' : ['pyfhd = PyFHD.pyfhd:main'],
    },
    package_data={'PyFHD': ['templates/*', 'PyFHD_gitinfo.npz']},
    
)
