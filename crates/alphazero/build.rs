include!("../../build_support/libtorch.rs");

fn main() {
    configure(false);

    #[cfg(feature = "raw-tensorrt")]
    build_raw_tensor_rt_shim();
}

/// Compiles and links the raw TensorRT runtime shim (`native/raw_trt_runtime.cpp`).
///
/// Defaults prefer the TensorRT 10.15 headers vendored under
/// `third_party/tensorrt-10.15.1` and the matching pip `libnvinfer.so.10` from
/// the project's Torch-TensorRT venv (BuilderFlag.FP16). Override with
/// `TENSORRT_INCLUDE_DIR` / `TENSORRT_LIB_DIR` when needed.
#[cfg(feature = "raw-tensorrt")]
fn build_raw_tensor_rt_shim() {
    use std::env;
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=native/raw_trt_runtime.cpp");
    println!("cargo:rerun-if-changed=native/raw_trt_runtime.h");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("alphazero is nested under the workspace")
        .to_path_buf();
    let vendored_headers = workspace.join("third_party/tensorrt-10.15.1/include");
    let venv_libs = PathBuf::from(
        "/home/mati/venvs/engine-zoo-trt-py313/lib/python3.13/site-packages/tensorrt_libs",
    );

    let cuda_root = env::var_os("CUDA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_cuda_root);
    let cuda_include = env::var_os("CUDA_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda_root.join("targets/x86_64-linux/include"));
    let cuda_lib = env::var_os("CUDA_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda_root.join("targets/x86_64-linux/lib"));
    let tensorrt_include = env::var_os("TENSORRT_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if vendored_headers.join("NvInfer.h").is_file() {
                vendored_headers
            } else {
                PathBuf::from("/usr/include")
            }
        });
    let tensorrt_lib = env::var_os("TENSORRT_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if venv_libs.join("libnvinfer.so.10").is_file() {
                venv_libs
            } else {
                PathBuf::from("/usr/lib")
            }
        });

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("native/raw_trt_runtime.cpp")
        .include(&tensorrt_include)
        .include(&cuda_include)
        .warnings(true)
        .compile("raw_trt_runtime");

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-search=native={}", tensorrt_lib.display());

    // Pip packages ship only the ABI-versioned soname; create a local
    // `libnvinfer.so` linker name so `-lnvinfer` resolves.
    if tensorrt_lib.join("libnvinfer.so.10").is_file()
        && !tensorrt_lib.join("libnvinfer.so").is_file()
    {
        let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
        let link_name = out.join("libnvinfer.so");
        let _ = std::fs::remove_file(&link_name);
        std::os::unix::fs::symlink(tensorrt_lib.join("libnvinfer.so.10"), &link_name)
            .unwrap_or_else(|error| panic!("symlink libnvinfer.so: {error}"));
        println!("cargo:rustc-link-search=native={}", out.display());
    }

    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if tensorrt_lib.join("libnvinfer.so.10").is_file() {
        // Keep the binary runnable without requiring LD_LIBRARY_PATH.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", tensorrt_lib.display());
    }
}

#[cfg(feature = "raw-tensorrt")]
fn default_cuda_root() -> std::path::PathBuf {
    for candidate in ["/opt/cuda", "/usr/local/cuda"] {
        let path = std::path::PathBuf::from(candidate);
        if path.is_dir() {
            return path;
        }
    }
    std::path::PathBuf::from("/usr/local/cuda")
}
