# The Output Directory

This is the default output directory of PyFHD, unless you provide a description via the configuration file, 
by default when PyFHD starts it's run it will create a directory.

The directory name will be in the format:

`pyfhd_<hour>_<minutes>_<seconds>_<day>_<month>_<year>`

For example let's say I start a run on 31st December 2999 with 5 seconds left till midnight, 23:55:00 (Maybe you fell into a Cyro Tube before running PyFHD?), then the directory created should be

`pyfhd_23_55_00_31_12_2999`

If you provided a description via the configuration file, then the direcotry will be:

`pyfhd_<description>_<hour>_<minutes>_<seconds>_<day>_<month>_<year>`

So for example if I wanted to describe the run in terms of what I am doing, which is some gridding straight after coming out of a cryo tube just before midnight of the 30th century then the directory will be:

`pyfhd_some_gridding_after_cryo_sleep_23_55_00_31_12_2999`

Setting a description will hopefully help you find your run later, just in case you fall into another cryo tube for another 1000 years. When providing a description PyFHD will remove all spaces and replace them with an underscore and all letters will be lowercase, this will keep the output directory consistent amongst many runs and hopefully across many machines.

## Inside of a PyFHD directory

Inside the directory is where you'll find all outputs of the run, the exact structure is still being figured out, once we know you'll know.

In general though, you will find the log as a text file, many hdf5 files storing the output of the run, and potentially some images.