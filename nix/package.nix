{ buildRustPackage
, hdf5
, pkg-config
, autoPatchelfHook
, cmake
, lib
}:
buildRustPackage {
  pname = "storm";
  version = "0.0.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../src
      ../benches
      ../vendor
    ];
  };

  nativeBuildInputs = [
    pkg-config
    cmake
    autoPatchelfHook
  ];
  buildInputs = [ hdf5.dev ];
  doCheck = false;
  auditable = false;

  cargoLock = {
    lockFile = ../Cargo.lock;
  };
}
