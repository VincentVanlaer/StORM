{ pkgs }:
{
  fenix = import (pkgs.fetchzip {
    url = "https://github.com/nix-community/fenix/archive/a074d1bc9fd34f6b3a9049c5a61a82aea2044801.zip";
    sha256 = "sha256-GhWGWyGUvTF7H2DDGlQehsve1vRqIKAFhxy6D82Nj3Q=";
  }) { inherit pkgs; };
}
