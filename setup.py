from setuptools import setup

setup(
    name = "PyFHD",
    version = 1.0,
    author = "ADACS - Astronomy Data and Compute Services",
    url = "https://github.com/ADACS-Australia/PyFHD",
    python_requires=">=3.7",
    packages = ['PyFHD'],
    description = 'Python Fast Holograhic Deconvolution: A Python package that does efficient image deconvolution for general radio astronomy, fast-mode Epoch of Reionization analysis, and simulation.',
    long_description = open("README.md").read(),
    long_description_content_type = 'text/markdown',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts' : ['pyfhd = PyFHD.pyfhd:main'],
    }
)

#TODO: Add More Fields and complete Setup.py