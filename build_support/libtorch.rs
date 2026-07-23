use std::env;
use std::path::PathBuf;

/// Adds an rpath only when the developer explicitly selected a LibTorch root.
/// `tch` itself supports `LIBTORCH_USE_PYTORCH`; that mode intentionally does
/// not guess a Python installation path from a build script.
pub fn configure(keep_libtorch_linked: bool) {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");

    let Some(root) = env::var_os("LIBTORCH") else {
        return;
    };
    let library_dir = PathBuf::from(root).join("lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_dir.display());

    if keep_libtorch_linked && env::consts::OS == "linux" {
        println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg-bins=-ltorch");
        println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
    }
}
