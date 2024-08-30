{
  inputs = {
    nixpkgs.url = "nixpkgs";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, fenix, ... }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs; [
        cmake
        gcc
        pkg-config
        gfortran
        (python3.withPackages (p: [ p.numpy p.scipy p.matplotlib p.mypy p.pyqt6 p.h5py ]))
        gnuplot
        openssl.dev
        hdf5.dev
        maxima
        (fenix.packages.${system}.latest.withComponents ["cargo" "clippy" "rustc" "rustfmt" "rust-analyzer" "rust-src" "rust-std"])
      ];
    };
  };
}
