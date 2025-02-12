{
  pkgs ? import <nixpkgs> { },
  rust-toolchain ? (import ./nix/pins.nix { inherit pkgs; }).fenix.latest.toolchain,
}:
let
  buildRustPackage =
    (pkgs.makeRustPlatform {
      cargo = rust-toolchain;
      rustc = rust-toolchain;
    }).buildRustPackage;
in
pkgs.callPackage ./nix/package.nix { inherit buildRustPackage; }
