fn main() {
    let libtorch = std::env::var("LIBTORCH").unwrap_or_else(|_| {
        format!(
            "{}/libs/libtorch",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    let rpath = format!("-Wl,-rpath,{libtorch}/lib");
    // Test harnesses: runnable without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg={rpath}");
}
