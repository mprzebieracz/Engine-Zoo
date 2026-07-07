fn main() {
    let libtorch = std::env::var("LIBTORCH").unwrap_or_else(|_| {
        format!(
            "{}/libs/libtorch",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    // Make the produced binaries runnable without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{libtorch}/lib");
    // Force a DT_NEEDED entry for libtorch.so (whose own deps pull in
    // libtorch_cuda.so): with the default --as-needed it gets dropped because
    // no symbol is referenced directly, and Cuda::is_available() then reports
    // false at runtime. The standard tch-rs workaround, minus the
    // --copy-dt-needed-entries flag that rust-lld doesn't support.
    println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg-bins=-ltorch");
    println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
}
