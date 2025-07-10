{
  _workspace ? import ../. {},
  _parent ? import ../shell.nix {},
  pkgs ? _workspace.pkgs,
  embasee ? _workspace.embasee,
  # libraries ? embasee.libraries,
}:
let
  libraries = embasee.libraries; # _workspace.libraries ++ 
in
pkgs.mkShell {
  # inputsFrom = [embasee.app];
  inputsFrom = [ _parent ];
  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath libraries}";
  packages = with pkgs; [
  ] ++ libraries;
}
