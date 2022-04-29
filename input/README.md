# The Input directory

Inside here should be the main input for a PyFHD run, the uvfits files which should be named by their observation id (obs_id). There should be two files for each observation to start a run, the `uvfits` and the metadata file `metafits`. So for example if the observation id I'm interested in is `1061316296`, then inside here should be the two files:

`1061316296.uvfits`

`1061316296.metafits`

## Other files that may be in here

Depending on the options for the PyFHD run you have chosen there may be other files you will need, I'd advise putting them in here.

These other files may include the text files needed for the bandpass during calibration or perhaps catalog sources, I will leave it upto you to decide on how to store these inputs inside this directory.

In general I would try to stick to putting things inside a directory called catalogs for catalogs and a bandpass directory for the bandpass files so the main input directory will only contain the observation fits files.