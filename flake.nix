{
  inputs = {
    nixpkgs.url = "nixpkgs";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    fenix,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    f = fenix.packages.${system};
    toolchain = f.combine [f.latest.cargo f.latest.clippy f.latest.rustc f.latest.rustfmt f.latest.rust-analyzer f.latest.rust-src f.latest.rust-std f.targets.x86_64-unknown-linux-musl.latest.rust-std f.latest.miri];
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs; [
        cmake
        pkg-config
        (python3.withPackages (p: [p.numpy p.scipy p.matplotlib p.mypy p.pyqt6 p.h5py p.tqdm]))
        hdf5.dev
        maxima
        toolchain
        bacon
        nodePackages.browser-sync
      ];

      shellHook = ''
        export CC="${pkgs.musl.dev}/bin/musl-gcc -static -Os"
      '';

      RUST_BACKTRACE = 1;
    };
  };
}
