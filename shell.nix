{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  nativeBuildInputs = [ pkgs.cmake pkgs.gcc pkgs.pkg-config pkgs.gfortran (pkgs.python3.withPackages (p: [ p.numpy p.scipy p.matplotlib p.mypy p.pyqt6 ])) pkgs.gnuplot ];
  buildInputs = [ pkgs.openssl pkgs.fontconfig ];
}
