=========
Changelog
=========

Upcoming changes
================

These changes have not yet been released but can be found on the main branch. Feel free to use and test these, but be aware that they may be not entirely functional or not work for certain input stellar models, ...

StORM now correctly assumes that input models include the spherical component of the centrifugal acceleration in their equation of hydrostatic support, matching what existing stellar evolution codes do. Previously StORM assumed that the models did not include this term. If you have been using rotating stellar models without passing ``--disable-symmetric`` to the deform command, StORM will have effectively applied the spherical component of the deformation twice. This has been fixed now.

Breaking changes
----------------

*None so far*

Features
--------

- Plain text GYRE stellar models are now supported. You can set the stellar model format with the ``--format`` option to the :doc:`input command <reference>`. Support for all versions the format (including the ones for the GSM format) has been added.
- Parallelisation is an optional feature now (and disabled by default). This is to improve backtraces when testing and profiling. Most users will want to have the feature enabled. This is the case for the pre-built binaries, and the commands for the source installation will do so as well.
- Commands are now saved in a history file. This only applies to interactive sessions (i.e. not when piping commands). The `location of the history file <https://docs.rs/directories/latest/directories/struct.ProjectDirs.html#method.data_local_dir>`_ depends on the OS.
- Rotation rates can now be set as the equatorial velocity.

Bug fixes
---------

- Fix crash on perturbing without any modes found.
- Fix radial mode eigenfunctions being slightly wrong. This was caused by switching between the reduced order system and the full system after bracketing. The discretization error of both systems is slightly different, and hence the locations of the solutions will be slightly different. This causes the computed eigenfunctions to be computed for the incorrect frequency. This caused some incorrect mode identifications as well.
- Fix bracketing from converging slowly in the presence of large determinants or non-monotonic behaviour within the bracket. While the modes obtained in such situations are most likely not accurate, StORM will at least finish quickly.
- Fix incorrect eigenfunctions for models with discontinuities.
- Non-deformed computations should be more consistent, as now the proper form of hydrostatic support is used. The impact of this change depends on the rotation rate. As an example, models rotating at ~15% of the critical rotation rate show frequency changes to low-order radial modes on the order of 0.2%.
- Deformed computations now disable the spherically symmetric component of the deformation if the rotation rate is sourced from the model file, as the input model should already contain these corrections. If the rotation rate is set through ``set-rotation-constant`` or ``set-rotation-overlay``, the sperically symmetric component is included in the perturbation, since in that case the model would not include the relevant rotation corrections.
