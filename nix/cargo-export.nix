{
  fetchFromGitHub,
  rustPlatform,
}:
rustPlatform.buildRustPackage rec {
  pname = "iai-callgrind-runner";
  version = "0.3";

  src = fetchFromGitHub {
    owner = "bazhenov";
    repo = "cargo-export";
    tag = "v${version}";
    hash = "sha256-Q3DCkkJ7GBlaXUet8nJVb6Nc/Eb7sx4VyejBAU1QsHU=";
  };

  doCheck = false;

  cargoHash = "sha256-6TLAjYeSrBFnCnVa8NykAt4fRs5g6c9vwbjKDz802mU=";
}

