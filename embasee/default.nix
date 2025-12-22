{
  sources,
  pkgs,
  fenix,
  gitignoreSource,
  nixgl,
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
    wayland
  ];
  nixGL = nixgl.auto.nixGLDefault; # Necessary for running glutin on non-Nixos distros
  embasee = naersk.buildPackage {
    name = "embasee";
    src = gitignoreSource ./.;

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

    LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libraries}";
    ORT_ENV_SYSTEM_LIB_LOCATION = "${pkgs.onnxruntime}/lib/libonnxruntime.so";
  };
in
rec {
  inherit libraries mypython;
  bin = pkgs.writeShellApplication {
    name = "embasee";
    runtimeInputs = [nixGL];
    text = ''
      export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}
      nixGL ${embasee}/bin/embasee "$@"
    '';
  };

  desktop = pkgs.makeDesktopItem {
    # Desktop launcher only
    name = "Embasee";
    desktopName = "Embasee Embedding Explorer";
    exec = "${nixGL}/bin/nixGL ${embasee}/bin/embasee";
  };

  app = pkgs.buildEnv {
    # all launchers
    name = "embasee";
    paths = [
      bin
      desktop
    ];
  };
}

