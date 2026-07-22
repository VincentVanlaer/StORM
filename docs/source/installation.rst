============
Installation
============

StORM can be installed in two ways:

- Prebuilt binaries, which contain all dependencies needed.
- From source. Requires various dependencies to be installed first.

Prebuilt binaries
=================

.. caution::

   For the prebuilt binaries, only linux installs have been tested. We welcome contributions to extend this documentation with instructions for other platforms.

Prebuilt binaries of StORM are provided, and require no dependencies.

1. Download the prebuilt binary for your operating system and architecture:

   - `Linux (x86-64) <linux/storm>`_.
   - `MacOS (universe) <macos/storm>`_.
   - `Windows (x86-64) <windows/storm>`_.

   If your operating system (or architecture) isn't listed here, use the from-source installation instructions below.
2. Make the binary executable by

   - On Linux: running ``chmod +x storm``
   - On MacOS: running ``chmod +x storm && xattr -d com.apple.quarantine storm``. MacOS does not allow you to execute unvalidated files downloaded through your browser. The second command removes the "this file has been downloaded from the internet" information.

3. Place it in a folder that is in ``PATH``, or refer to the binary with its full path

From source
===========

To compile from source, you must install the following dependencies first:

- `A recent rust compiler <https://www.rust-lang.org/tools/install>`_.
- cmake, which is used to compile the HDF5 library. If you are on Linux, you can find cmake in your package manager. For other platforms, have a look at https://cmake.org/download/#latest.

Download `the source archive <storm.tar.gz>`_ and extract it.
This will create a new directory ``storm``, which you can rename if you want.
Open a terminal and enter this directory.
You can compile StORM by running ``cargo build --release`` or run it directly with ``cargo run --release --bin storm``.
The resulting binary can be found at ``target/release/storm``.
You can run it directly from there, or place it somewhere in your ``PATH``.

If you are familiar with nix, you can also run ``nix-shell`` to get all the necessary dependencies, or ``nix-build`` to directly build the StORM binaries.
