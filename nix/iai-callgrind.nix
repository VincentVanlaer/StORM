{
  fetchFromGitHub,
  rustPlatform,
}:
rustPlatform.buildRustPackage rec {
  pname = "iai-callgrind-runner";
  version = "0.14.0";

  src = fetchFromGitHub {
    owner = "iai-callgrind";
    repo = "iai-callgrind";
    tag = "v${version}";
    hash = "sha256-NUFbA927Iye8DnmBWAQNiFmEen/a0931XlT+9gAQSV4=";
  };

  buildAndTestSubdir = "iai-callgrind-runner";
  doCheck = false;

  cargoHash = "sha256-gMpMQ2XUizSXbk/uva2+m61RHkg+x2rx9PYvcmwBVlI=";
}
