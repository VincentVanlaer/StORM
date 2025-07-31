{
  pkgs ? import (fetchTarball {
    url = "https://github.com/nixos/nixpkgs/archive/32a4e87942101f1c9f9865e04dc3ddb175f5f32e.zip";
    sha256 = "sha256:1jvflnbrxa8gjxkwjq6kdpdzgwp0hs59h9l3xjasksv0v7xlwykz";
  }) { },
}:
{
  pkgs = pkgs;
  fenix = import (pkgs.fetchzip {
    url = "https://github.com/nix-community/fenix/archive/c3c27e603b0d9b5aac8a16236586696338856fbb.zip";
    sha256 = "sha256-zky3+lndxKRu98PAwVK8kXPdg+Q1NVAhaI7YGrboKYA=";
  }) { inherit pkgs; };

  gyre =
    (import (pkgs.fetchzip {
      url = "https://github.com/VincentVanlaer/nix-gyre/archive/dc05b5672236bf79420d0e04b1e92cf61d382e90.zip";
      sha256 = "sha256-5m8lOQhGMlLcitPbudBt8FVwxVQSdSxVtJhcNi145Uw=";
    }) { inherit pkgs; }).gyre-80;
}
