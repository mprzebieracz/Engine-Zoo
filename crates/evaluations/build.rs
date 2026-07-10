fn main() {
    let libtorch = std::env::var("LIBTORCH").unwrap_or_else(|_| {
        format!(
            "{}/libs/libtorch",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch}/lib");
    // Keep libtorch.so in DT_NEEDED so its CUDA libraries and kernel
    // registrations are loaded before tch creates CUDA tensors.
    println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg-bins=-ltorch");
    println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
}
