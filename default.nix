# import ./package.nix {}
let
  lock = builtins.fromJSON (builtins.readFile ./flake.lock);
  gitignore = fetchTarball {
    url = "https://github.com/hercules-ci/gitignore.nix/archive/${lock.nodes.gitignore-src.locked.rev}.tar.gz";
    sha256 = lock.nodes.gitignore-src.locked.narHash;
  };
in
(import (
  fetchTarball {
    url = "https://github.com/NixOS/flake-compat/archive/${lock.nodes.flake-compat.locked.rev}.tar.gz";
    sha256 = lock.nodes.flake-compat.locked.narHash;
  }
) { src = ./.; }).defaultNix.outputs.packages.${builtins.currentSystem}
