{
  sources ? import ./nix/sources.nix,
  pkgs ? import sources.nixpkgs {},
  fenix ? import sources.fenix {},
  gitignore ? import sources."gitignore.nix" {},
  nixgl ? import sources.nixGL {},
  rust-toolchain ? fenix.combine [
    fenix.complete.toolchain
    fenix.targets.wasm32-unknown-unknown.latest.rust-std
  ],
  naersk ? pkgs.callPackage sources.naersk {
    cargo = rust-toolchain;
    rustc = rust-toolchain;
  },
}:
let
  libraries = with pkgs; [
    openssl
    wasmtime
    stdenv.cc.cc.lib
  ];

  callPackage = pkgs.lib.callPackageWith {
    inherit sources pkgs fenix rust-toolchain naersk gitignore nixgl;
    inherit (gitignore) gitignoreSource;
  };
in
{
  inherit pkgs libraries rust-toolchain;
}
