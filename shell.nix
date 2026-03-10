{
  pkgs ? (import ./nix/pins.nix { }).pkgs,
  fenix ? (import ./nix/pins.nix { }).fenix,
  gyre ? (import ./nix/pins.nix { }).gyre,
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

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rust-toolchain;
    rustc = rust-toolchain;
  };

  iai-callgrind-runner = pkgs.callPackage ./nix/iai-callgrind.nix { inherit rustPlatform; };
  cargo-export = pkgs.callPackage ./nix/cargo-export.nix { inherit rustPlatform; };
  cargo-zigbuild = pkgs.cargo-zigbuild.override { inherit rustPlatform; };
  bench = pkgs.writeScriptBin "bench" /* bash */ ''
    cur=`jj st | grep "(@)" | cut -f 6 -d " "`
    log='test-data/generated/bench-log'
    # baseline
    jj new $1
    rm $log
    touch $log
    direnv exec . cargo export target/benchmarks -- bench --bench=wall-clock 2>>$log 1>>$log
    direnv exec . cargo bench -q --bench=instruction-count 2>>$log 1>>$log
    jj abandon

    # actual
    jj new $2
    direnv exec . cargo bench -q --bench=wall-clock -- compare -s 100 -d test-data/generated/benches/ --gnuplot target/benchmarks/wall_clock 2>>$log
    direnv exec . cargo bench -q --bench=instruction-count 2>>$log
    jj abandon

    jj edit $cur
  '';

  gyre-90 = pkgs.runCommand "gyre-90" { } ''
    mkdir -p $out/bin
    ln -s ${gyre.gyre-90}/bin/gyre $out/bin/gyre-90
  '';
  gyre-81 = pkgs.runCommand "gyre-81" { } ''
    mkdir -p $out/bin
    ln -s ${gyre.gyre-81}/bin/gyre $out/bin/gyre-81
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
        p.sphinxcontrib-katex
        p.sphinx-autobuild
        p.myst-parser
        p.furo
      ]))
      maxima
      bacon
      nodePackages.browser-sync
      gyre.gyre-90
      # Only the main executable
      gyre-90
      gyre-81
      # Benchmark
      valgrind
      libclang
      iai-callgrind-runner
      cargo-export
      cargo-zigbuild
      gnuplot
      bench
      # Docs
      nodejs
    ];

  RUST_BACKTRACE = 1;
})
