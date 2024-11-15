{
  pkgs ? import <nixpkgs> {},
  buildRustPackage ? let
    fenix-src = pkgs.fetchzip {
      url = "https://github.com/nix-community/fenix/archive/664e2f335aa5ae28c8759ff206444edb198dc1c9.tar.gz";
      sha256 = "sha256-vau17dcGvfEWX9DLWuSPC0dfE0XcDe9ZNlsqXy46P88=";
    };
    fenix = import fenix-src {inherit pkgs;};
    toolchain = fenix.latest.toolchain;
  in
    (pkgs.makeRustPlatform {
      cargo = toolchain;
      rustc = toolchain;
    })
    .buildRustPackage,
}:
pkgs.callPackage ./package.nix {inherit buildRustPackage;}
