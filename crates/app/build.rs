fn main() {
    let libtorch = std::env::var("LIBTORCH").unwrap_or_else(|_| {
        format!(
            "{}/libs/libtorch",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    let rpath = format!("-Wl,-rpath,{libtorch}/lib");
    // Test harnesses and bins: runnable without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg={rpath}");
    // Linux linkers may drop libtorch.so because no symbol is referenced
    // directly. Keep the dependency there so CUDA registrations are loaded.
    // Apple ld does not understand these GNU linker flags; libtorch's normal
    // dependency graph is sufficient on macOS.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg-bins=-ltorch");
        println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
    }
}
