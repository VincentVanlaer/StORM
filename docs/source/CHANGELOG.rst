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

Bug fixes
---------

- Fix crash on perturbing without any modes found.
