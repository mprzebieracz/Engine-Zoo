use std::env;
use std::path::PathBuf;

/// Adds the selected LibTorch directory to executable runtime lookup.
/// `tch` itself supports `LIBTORCH_USE_PYTORCH`; that mode intentionally does
/// not guess a Python installation path from a build script.
pub fn configure(keep_libtorch_linked: bool) {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");

    let Some(root) = env::var_os("LIBTORCH") else {
        return;
    };
    let library_dir = PathBuf::from(root).join("lib");
    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_dir.display());

    if keep_libtorch_linked && env::consts::OS == "linux" {
        println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
        // `libtorch.so` depends on this library, but Linux's default
        // `--as-needed` link behaviour otherwise omits it from the final
        // executable. That makes `tch::Cuda::is_available()` report false
        // even when the selected LibTorch build has CUDA support.
        println!("cargo:rustc-link-arg-bins=-l:libtorch_cuda.so");
        println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
    }
}
