{ buildRustPackage
, hdf5
, pkg-config
, autoPatchelfHook
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
    ];
  };

  nativeBuildInputs = [
    pkg-config
    autoPatchelfHook
  ];
  buildInputs = [ hdf5.dev ];
  doCheck = false;
  auditable = false;

  cargoLock = {
    lockFile = ../Cargo.lock;
  };
}
