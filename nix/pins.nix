{
  pkgs ? import (fetchTarball {
    url = "https://github.com/nixos/nixpkgs/archive/60e405b241edb6f0573f3d9f944617fe33ac4a73.zip";
    sha256 = "sha256:0d6sdvj5s57s0nkv4wc7bh4hv7w1v0h4p2jhbzdic2s1swfkilsb";
  }) { },
}:
{
  pkgs = pkgs;
  fenix = import (pkgs.fetchzip {
    url = "https://github.com/nix-community/fenix/archive/a074d1bc9fd34f6b3a9049c5a61a82aea2044801.zip";
    sha256 = "sha256-GhWGWyGUvTF7H2DDGlQehsve1vRqIKAFhxy6D82Nj3Q=";
  }) { inherit pkgs; };

  gyre =
    (import (pkgs.fetchzip {
      url = "https://github.com/VincentVanlaer/nix-gyre/archive/dc05b5672236bf79420d0e04b1e92cf61d382e90.zip";
      sha256 = "sha256-5m8lOQhGMlLcitPbudBt8FVwxVQSdSxVtJhcNi145Uw=";
    }) { inherit pkgs; }).gyre-80;
}
