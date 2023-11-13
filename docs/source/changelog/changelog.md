# Change Log

## Unreleased

## Version 1.0

### Breaking Changes!
* Much of the new translation has slightly different shapes and sizes to FHD in some regard, PyFHD is becoming less compatible with FHD day by day, in order to use any sav files from FHD with PyFHD, users will need to use functions such as `recarray_to_dict` and `sav_file_vis_arr_swap_axes` to make any variables from the sav files compatible with PyFHD.
* Much of PyFHD setup has changed and many bugs have been fixed, this will break previous PyFHD configurations, not that there were many (if any).

### New Features
* Calibration has been translated and tested, and runs completely in PyFHD without breaking 🎉 
* Calibration has been integrated into the main python pipeline, and actually runs all the way through using observations `1088716176` and `1088716276`. Their scientific validity needs to be checked.
* Many of the functions in between the `beam_setup` and `vis_calibrate` and `visibility_grid` have been translated and tested (or in need of more testing, e.g. `vis_flag`).
* There is a pull request template now, it's the one you're reading

### Bug Fixes
* `pyfhd_setup` and `pyfhd.yaml` configuration has been changed to fix bugs like many of the configuration options having underscores instead of dashes in the `yaml` which broke the config in `main`.
* Bugs from FHD have been fixed such as `vis_baseline_hist` and the auto-correlations do seem to work better for `PyFHD`
* Bugs from the initial `obs` translation have been fixed, the flagging now seems to be consistent with FHD in the cases where FHD does not have a bug (note [`fhd_struct_init_obs` and `fhd_struct_init_meta` bug](https://github.com/EoRImaging/FHD/issues/311))

### Test Changes
* Tests for every single `calibration_utils` and additional tests for `vis_calibrate_subroutine` have been added, the entire PyFHD test collection sits at 281 tests and counting.

### Dependency Changes
* `myst-parser` has been added to the dependencies to use markdown files in the docs site.

### Version Changes
* Set the version to 1.0 ready for the official release later this year.

### Translation Changes
* Calibration was translated and tested alongside the functions that glue the initial data extraction part of FHD to the calibration and from calibration to gridding