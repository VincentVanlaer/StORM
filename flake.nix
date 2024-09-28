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
    toolchain = fenix.packages.${system}.latest.withComponents ["cargo" "clippy" "rustc" "rustfmt" "rust-analyzer" "rust-src" "rust-std"];
    buildRustPackage =
      (pkgs.makeRustPlatform {
        cargo = toolchain;
        rustc = toolchain;
      })
      .buildRustPackage;
    cargo-show-asm = buildRustPackage {
      pname = "cargo-show-asm";
      version = "0.2.36";

      src = pkgs.fetchCrate {
        pname = "cargo-show-asm";
        version = "0.2.36";
        hash = "sha256-Ptv8txt7YXPh1QvFwsoRbBQgeLBGn6iVqst1GHU+JJw=";
      };

      cargoHash = "sha256-GkhFbRhEJ+7ikqkNPydKx3Ty8KRsGts5UnwA8bl8Po8=";
      buildFeatures = ["disasm"];

      nativeBuildInputs = [
        pkgs.installShellFiles
      ];

      postInstall = ''
        installShellCompletion --cmd cargo-asm \
          --bash <($out/bin/cargo-asm --bpaf-complete-style-bash) \
          --fish <($out/bin/cargo-asm --bpaf-complete-style-fish) \
          --zsh  <($out/bin/cargo-asm --bpaf-complete-style-zsh)
      '';
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs; [
        cmake
        gcc
        pkg-config
        gfortran
        (python3.withPackages (p: [p.numpy p.scipy p.matplotlib p.mypy p.pyqt6 p.h5py]))
        gnuplot
        openssl.dev
        hdf5.dev
        maxima
        toolchain
        cargo-show-asm
        cargo-watch
        llvm
        nodePackages.browser-sync
      ];
    };
  };
}
