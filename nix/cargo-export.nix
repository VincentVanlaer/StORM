{ fetchFromGitHub
, rustPlatform
}:
rustPlatform.buildRustPackage rec {
  pname = "cargo-export";
  version = "0.3";

  src = fetchFromGitHub {
    owner = "bazhenov";
    repo = "cargo-export";
    tag = "v${version}";
    hash = "sha256-Q3DCkkJ7GBlaXUet8nJVb6Nc/Eb7sx4VyejBAU1QsHU=";
  };

  doCheck = false;

  cargoHash = "sha256-UwLxijc60U/s0SJlBCByl1ruiw8MqskZ7Sgxyo56xIM=";
}

