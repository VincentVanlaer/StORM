---
weight: 300
draft: false
title: "Installation"
icon: "rocket_launch"
---

{{< alert context="warning" text="Only Linux installs have been tested thus far" />}}

StORM can be installed in two ways:

- Prebuilt binaries, which contains all dependencies
- From source. Requires various dependencies to be installed first.

## Prebuilt binaries

Prebuilt binaries of StORM are provided, and require no dependencies.
Currently, only prebuilt binaries for Linux-based operating systems are supported.

1. Download the prebuilt binary below
1. Make the binary executable by running `chmod +x storm`
1. Place it in a folder that is in `PATH`, or refer to it directly

<div class="col-sm-2 pt-2">
	<a class="ms-auto" href="../storm">
		<div class="card h-100 my-1 text-center card-title">
			<div class="card-body py-2">
                {{< icons/icon vendor=bootstrap name=tux size=5em >}}
                <br>
                <i class="material-icons align-middle">download</i>
			</div>
		</div>
    </a>
</div>

## From source

To compile from source, you must install the following dependencies first:

- [A recent rust compiler](https://www.rust-lang.org/tools/install).
- cmake, which is used to compile the HDF5 library. If you are on Linux, you can find cmake in your package manager. For other platforms, have a look at https://cmake.org/download/#latest.

Download [the source archive](storm.tar.gz) and extract it.
This will create a new directory `storm`, which you can rename if you want.
Open a terminal and enter this directory.
You can compile StORM by running `cargo build --release` or run it directly with `cargo run --release --bin storm`.
The resulting binary can be found at `target/release/storm`.
You can run it directly from there, or place it somewhere in your `PATH`.

If you are familiar with nix, you can also run `nix-shell` to get all the necessary dependencies, or `nix-build` to directly build the StORM binaries.
