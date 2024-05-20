{
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [ cmake gcc pkg-config gfortran (python3.withPackages (p: [ p.numpy p.scipy p.matplotlib p.mypy p.pyqt6 p.h5py ])) gnuplot openssl maxima];
        };
      });
}
