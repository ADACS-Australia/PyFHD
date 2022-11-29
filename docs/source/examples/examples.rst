Examples
===========

Preamble - directory structure and using ``FHD`` outputs
----------------------------------------------------------

``FHD`` creates a specific directory structure and naming convention that must be respected for us to use the ``IDL`` outputs. In each of the examples below, if ``FHD`` outputs are required, their expected locations and names will be explained.

Those familiar with ``FHD`` should understand the keyword ``version``; this means ``FHD`` outputs will be output to a subdir called ``fhd`` + ``_`` + version. The equivalent in ``PyFHD`` is the ``--description`` option, which will make a subdir called ``pyfhd`` + ``_`` + description.

When running calibration, ``PyFHD`` will (currently) call ``FHD``, and output everything into a subsubdir, which will start with ``fhd``, so we know the outputs have come from the ``IDL`` code. The following (incomplete) command:

.. code-block:: bash

    pyfhd 1088282552 \
        --output-path /where/be/outputs/ \
        --description my_first_run

will result in a directory structure like this:

.. code-block:: bash

    /where/be/outputs/
    └── pyfhd_my_first_run
      ├── fhd_pyfhd_my_first_run
      │ ├── 1088282552_variables.sav
      │ ├── Healpix
      │ ├── metadata
      │ └── vis_data
      └── gridding_outputs

The ``Healpix``, ``metadata``, and ``vis_data`` subdirs are generated (amongst other things) by ``FHD``. ``fhd_pyfhd_my_first_run`` and ``gridding_outputs`` are created by ``PyFHD``. If you don't want to use ``PyFHD`` as a wrapper to run calibration, but still want to run gridding on ``FHD`` outputs, either run ``FHD`` to create/output into ``fhd_pyfhd_my_first_run``, or just symlink in the ``Healpix``, ``metadata``, and ``vis_data``. Otherwise ``PyFHD`` won't be able to find the ``FHD`` outputs.

.. warning::
    
    If you don't use ``PyFHD`` to run ``FHD``, the ``1088282552_variables.sav`` file won't be saved. ``PyFHD`` needs some extra information that isn't saved by default in ``FHD``. Adding the line 

    .. code-block:: idl

        save, bi_use, weights_flag, variance_flag, model_return, preserve_visibilities, filename=file_path_fhd + '_variables.sav'

    into your ``FHD`` run should produce the needed file (this is what ``PyFHD`` does internally).

Once ``PyFHD`` is fully Pythonic, this file structure faffing about will be handled internally to the code. Please bear with us for now.

Running calibration (uses IDL)
-------------------------------------------

Gridding IDL calibration outputs
-------------------------------------------

.. note::

   When performing gridding, the gridding kernel object is often large and complex. As such, reading and converting from the native ``IDL`` ``.sav`` binary format should only be done once, and saved into a numpy ``.npz``. An example ``python`` code snippet to do exactly this is:

   .. code-block:: python

      from scipy.io import readsav
      import numpy as np
      sav_dict = readsav('gauss_beam_pointing-2.sav', python_dict=True)
      np.savez('gauss_beam_pointing-2.npz', **sav_dict)

   Be aware this can take hours. TODO work out a way to share the converted kernels.
   

In this example, calibration should already have been run using ``FHD``. We will then take the calibrated visibilities/model and grid them into two groups: even and odd time steps. This is the first step towards creating a power spectrum (:math:`\varepsilon`\ *ppsilon* uses the difference between the even and odd to estimate the noise).

.. code-block:: bash

   pyfhd \
       '1088281328' \
       --input-path /path/to/data/ \
       --calibrate-visibilities=False \
       --output-path /current/working/directory/ \
       --description my_first_run \
       --grid-psf-file /path/to/beams/gauss_beam_pointing-2.npz \
       --ps-kspan=200 \
       --grid_IDL_outputs

For this command to work, the raw data (which ``FHD`` needs to work out some metadata-type things) should exist as specified above as::

    /path/to/data/1088281328.uvfits

The following ``FHD`` outputs must also exist, in these locations:

.. code-block:: bash

    /current/working/directory
    └── pyfhd_my_first_run
      └── fhd_pyfhd_my_first_run
        ├── 1088281328_variables.sav
        ├── metadata
        | ├── 1088281328_obs.sav
        | └── 1088281328_params.sav
        └── vis_data
          ├── 1088281328_vis_XX.sav
          ├── 1088281328_vis_YY.sav
          ├── 1088281328_vis_model_XX.sav
          ├── 1088281328_vis_model_YY.sav
          └── 1088281328_flags.sav 

Other than specifying file paths, the other necessary arguments have the following effect:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Argument
     - Meaning
   * - -\-calibrate-visibilities=False
     - Default is to run calibration, so switch it off
   * - -\-grid-psf-file
     - A converted ``FHD`` ``psf`` object to use as a gridding kernel
   * - -\-ps-kspan=200
     - Set the width of the gridded visibilities (wavelengths)
   * - -\-grid_IDL_outputs
     - Switches on gridding using ``FHD`` outputs

Once run, this will produce the following outputs:

.. code-block:: bash

   /current/working/directory
   └── pyfhd_my_first_run
     └── gridding_outputs
         ├── 1088281328_gridded_uv_cube_even_XX.h5
         ├── 1088281328_gridded_uv_cube_even_YY.h5
         ├── 1088281328_gridded_uv_cube_odd_XX.h5
         └── 1088281328_gridded_uv_cube_odd_YY.h5

These files contain the gridded data sets, with each frequency slice being a separate ``hdf5`` data object within the relevant file.

Image gridded outputs and project to Healpix (uses IDL)
----------------------------------------------------------
Assuming we have run ``PyFHD`` to grid some visibilities (as detailed in `Gridding IDL calibration outputs`_ above), in this example we will use ``FHD`` to image and project them to Healpix. These outputs can then be input into :math:`\varepsilon`\ *ppsilon*. The example command is:

.. code-block:: bash

   pyfhd \
       '1088281328' \
       --input-path /path/to/data/ \
       --calibrate-visibilities=False \
       --output-path /current/working/directory/ \
       --description my_first_run \
       --grid-psf-file /path/to/beams/gauss_beams_pointing-2.sav \
       --ps-kspan=200 \
       --IDL_healpix_gridded_outputs

Note that unlike in the `Gridding IDL calibration outputs`_ example, this time we point ``--grid-psf-file`` towards an ``IDL`` save file. This is because ``FHD`` needs to access the ``psf`` object within, and ``IDL`` cannot read the ``numpy`` format. This command will write a number of ``.pro`` files to launch ``FHD``, with a small amount of extra code to read in the gridded ``hdf5`` files. For those interested, the template is in ``PyFHD/PyFHD/templates/vis_model_freq_split_read_python.pro``.

Once this code is run, the following outputs are created:

.. code-block:: bash

   /current/working/directory
   └── fhd_pyfhd_my_first_run
     └── Healpix
         ├── 1088281328_even_cubeXX.sav
         ├── 1088281328_even_cubeYY.sav
         ├── 1088281328_odd_cubeXX.sav
         └── 1088281328_odd_cubeYY.sav

Both grid and image/project to Healpix
----------------------------------------
It is straight forward to run the gridding and imaging/healpix projection (detailed in examples `Gridding IDL calibration outputs`_ and `Image gridded outputs and project to Healpix (uses IDL)`_ above) in a single command:

.. code-block:: bash

   pyfhd \
       '1088281328' \
       --input-path /path/to/data/ \
       --calibrate-visibilities=False \
       --output-path /current/working/directory/ \
       --description my_first_run \
       --grid-psf-file /path/to/beams/gauss_beam_pointing-2.npz \
                       /path/to/beams/gauss_beams_pointing-2.sav \
       --ps-kspan=200 \
       --grid_IDL_outputs \
       --IDL_healpix_gridded_outputs

The important thing to note is that we supply both the ``.npz`` and ``.sav`` format beams to the ``--grid-psf-file``, which keeps both ``Python`` and ``IDL`` happy.