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

Bug fixes
---------

- Fix crash on perturbing without any modes found.
