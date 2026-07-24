=========
Changelog
=========

Upcoming changes
================

These changes have not yet been released but can be found on the main branch. Feel free to use and test these, but be aware that they may be not entirely functional or not work for certain input stellar models, ...

Breaking changes
----------------

*None so far*

Features
--------

- Plain text GYRE stellar models are now supported. You can set the stellar model format with the ``--format`` option to the :doc:`input command <reference>`. Support for all versions the format (including the ones for the GSM format) has been added.
- Parallelisation is an optional feature now (and disabled by default). This is to improve backtraces when testing and profiling. Most users will want to have the feature enabled. This is the case for the pre-built binaries, and the commands for the source installation will do so as well.
- Commands are now saved in a history file. This only applies to interactive sessions (i.e. not when piping commands). The `location of the history file <https://docs.rs/directories/latest/directories/struct.ProjectDirs.html#method.data_local_dir>`_ depends on the OS.

Bug fixes
---------

- Fix crash on perturbing without any modes found.
- Fix radial mode eigenfunctions being slightly wrong. This was caused by switching between the reduced order system and the full system after bracketing. The discretization error of both systems is slightly different, and hence the locations of the solutions will be slightly different. This causes the computed eigenfunctions to be computed for the incorrect frequency. This caused some incorrect mode identifications as well.
- Fix bracketing from converging slowly in the presence of large determinants or non-monotonic behaviour within the bracket. While the modes obtained in such situations are most likely not accurate, StORM will at least finish quickly.
