.. _FHD: https://github.com/EoRImaging/FHD
.. _FHD installation page: https://github.com/EoRImaging/FHD#installation

*************
Installation
*************

Dependencies
##############

As ``PyFHD`` is still a work in progress, it relies on the original ``IDL`` `FHD`_ code for large areas of functionality. Please refer to the `FHD installation page`_. ``FHD`` itself has a number of dependencies, which must all be added to an environment variable ``IDL_PATH``. An example in use on the ``OzStar`` cluster is given below for reference.

.. code-block:: bash

   export IDL_PATH='<IDL_DEFAULT>':${IDL_PATH}:+/fred/oz048/jline/ADACS/test_PyFHD/FHD_code/FHD
   export IDL_PATH=:${IDL_PATH}:+/fred/oz048/jline/ADACS/test_PyFHD/FHD_code/coyote
   export IDL_PATH=${IDL_PATH}:+/fred/oz048/jline/ADACS/test_PyFHD/FHD_code/fhdps_utils
   export IDL_PATH=${IDL_PATH}:+/apps/skylake/software/compiler/gcc/6.4.0/healpix/3.50/src/idl/
   export IDL_PATH=${IDL_PATH}:+/fred/oz048/jline/ADACS/test_PyFHD/FHD_code/eppsilon/


Installing ``PyFHD``
#######################

Clone (and move into) the ``PyFHD`` repo::

   $ git clone https://github.com/ADACS-Australia/PyFHD && cd PyFHD

Then just pip install the repo::

   $ pip install -r requirements.txt .

.. warning:: ``PyFHD`` currently relies on a number of ``IDL`` templates. Running ``pip install`` sets up a number of internal paths to efficiently find these templates, and so is *essential* for ``PyFHD`` to run correctly.

That's it! You're good to go.

.. note:: Once ``PyFHD`` is feature-complete, we aim to make this both a ``conda`` install and a ``pip`` install.