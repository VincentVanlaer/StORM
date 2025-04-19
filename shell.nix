{
  pkgs ? import <nixpkgs> { },
  fenix ? (import ./nix/pins.nix { inherit pkgs; }).fenix,
  gyre ? (import ./nix/pins.nix { inherit pkgs; }).gyre,
}:
let
  rust-toolchain = fenix.combine [
    fenix.latest.cargo
    fenix.latest.clippy
    fenix.latest.rustc
    fenix.latest.rustfmt
    fenix.latest.rust-analyzer
    fenix.latest.rust-src
    fenix.latest.rust-std
    fenix.targets.x86_64-unknown-linux-musl.latest.rust-std
    fenix.latest.miri
  ];
  package = (import ./default.nix { inherit rust-toolchain; });
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
    ];

  shellHook = ''
    export CC="${pkgs.musl.dev}/bin/musl-gcc -static -Os"
  '';

  RUST_BACKTRACE = 1;
})
