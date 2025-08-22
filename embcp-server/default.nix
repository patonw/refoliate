{
  sources,
  pkgs,
  fenix,
  rust-toolchain,
  naersk,
}:
let
  libraries = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    openssl
    onnxruntime
  ];
in
{
  bin = naersk.buildPackage {
    name = "embcp-server";
    src = ../.;
    cargoBuildOptions = opts: opts ++ [ "--package embcp-server" ];

    nativeBuildInputs = with pkgs; [
      pkg-config
      cmake
      makeWrapper
    ];

    buildInputs = with pkgs; [
    ] ++ libraries;

    postFixup = ''
      if [[ -f $out/bin/embcp-server ]]; then
        wrapProgram $out/bin/embcp-server \
          --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath libraries}
      fi
    '';

    LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libraries}";
    ORT_ENV_SYSTEM_LIB_LOCATION = "${pkgs.onnxruntime}/lib/libonnxruntime.so";
  };
}

