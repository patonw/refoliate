{
  sources,
  pkgs,
  fenix,
  rust-toolchain,
  naersk,
}:
let
  mypython = pkgs.python312.withPackages(ps: with ps; [
    polars
    umap-learn
    scikit-learn
    hdbscan
    numpy
    pyarrow
  ]);
  libraries = with pkgs; [
    mypython
    stdenv.cc.cc.lib
    xorg.libxcb
    libxkbcommon
    fontconfig
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    xorg.libX11.dev
    libGL
    zlib
    openssl
    onnxruntime
  ];
in
{
  inherit libraries mypython;
  bin = naersk.buildPackage {
    name = "embasee";
    src = ../.;
    cargoBuildOptions = opts: opts ++ [ "--package embasee" ];

    nativeBuildInputs = with pkgs; [
      pkg-config
      cmake
      makeWrapper
    ];

    buildInputs = with pkgs; [
    ] ++ libraries;

    postFixup = ''
      if [[ -f $out/bin/embasee ]]; then
        wrapProgram $out/bin/embasee \
          --prefix PATH : ${pkgs.lib.makeBinPath [ mypython ]} \
          --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath libraries}
      fi
    '';

    ORT_ENV_SYSTEM_LIB_LOCATION = "${pkgs.onnxruntime}/lib/libonnxruntime.so";
  };
}

