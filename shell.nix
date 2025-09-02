{ pkgs ? (import ./nix/pins.nix { }).pkgs
, fenix ? (import ./nix/pins.nix { }).fenix
, gyre ? (import ./nix/pins.nix { }).gyre
}:
let
  rust-toolchain = fenix.combine [
    fenix.stable.cargo
    fenix.stable.clippy
    fenix.stable.rustc
    fenix.stable.rustfmt
    fenix.stable.rust-analyzer
    fenix.stable.rust-src
    fenix.stable.rust-std
    fenix.targets.x86_64-unknown-linux-musl.stable.rust-std
    fenix.targets.x86_64-pc-windows-gnu.stable.rust-std
    fenix.targets.x86_64-apple-darwin.stable.rust-std
    fenix.targets.aarch64-apple-darwin.stable.rust-std
  ];
  package = (import ./default.nix { inherit rust-toolchain; });

  rustPlatform =
    pkgs.makeRustPlatform {
      cargo = rust-toolchain;
      rustc = rust-toolchain;
    };

  iai-callgrind-runner = pkgs.callPackage ./nix/iai-callgrind.nix { inherit rustPlatform; };
  cargo-export = pkgs.callPackage ./nix/cargo-export.nix { inherit rustPlatform; };
  cargo-zigbuild = pkgs.cargo-zigbuild.override { inherit rustPlatform; };
  bench = pkgs.writeScriptBin "bench" /* bash */ ''
    cur=`jj st | grep "(@)" | cut -f 6 -d " "`
    # baseline
    jj new $1
    cargo export target/benchmarks -- bench --bench=wall-clock 2>/dev/null
    cargo bench -q --bench=instruction-count 2>/dev/null 1>/dev/null

    # actual
    jj new $2
    cargo bench -q --bench=wall-clock -- compare -s 100 -d test-data/generated/benches/ --gnuplot target/benchmarks/wall_clock 2>/dev/null
    cargo bench -q --bench=instruction-count 2>/dev/null

    jj edit $cur
  '';
in

package.overrideAttrs (attrs: {
  nativeBuildInputs =
    with pkgs;
    attrs.nativeBuildInputs
    ++ [
      cmake
      (python3.withPackages (p: [
        p.numpy
        p.scipy
        p.matplotlib
        p.mypy
        p.pyqt6
        p.h5py
        p.tqdm
      ]))
      maxima
      bacon
      nodePackages.browser-sync
      gyre
      # Benchmark
      valgrind
      libclang
      iai-callgrind-runner
      cargo-export
      cargo-zigbuild
      gnuplot
      bench
      # Docs
      hugo
      go
    ];

  RUST_BACKTRACE = 1;
})
