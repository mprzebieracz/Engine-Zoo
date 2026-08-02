include!("../../build_support/libtorch.rs");

fn main() {
    configure(false);

    #[cfg(feature = "raw-tensorrt")]
    build_raw_tensor_rt_shim();
}

#[cfg(feature = "raw-tensorrt")]
fn build_raw_tensor_rt_shim() {
    println!("cargo:rerun-if-changed=native/raw_trt_runtime.cpp");
    println!("cargo:rerun-if-changed=native/raw_trt_runtime.h");
    for variable in [
        "TENSORRT_ROOT",
        "TENSORRT_INCLUDE_DIR",
        "TENSORRT_LIB_DIR",
        "TENSORRT_PYTHON",
        "CUDA_ROOT",
        "CUDA_INCLUDE_DIR",
        "CUDA_LIB_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let (tensorrt_include, tensorrt_lib) = tensor_rt_paths();
    let (cuda_include, cuda_lib) = cuda_paths();

    require_file(&tensorrt_include.join("NvInfer.h"), "TensorRT header");
    require_directory(&tensorrt_lib, "TensorRT library directory");
    require_file(&cuda_include.join("cuda_runtime_api.h"), "CUDA header");
    require_directory(&cuda_lib, "CUDA library directory");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("native/raw_trt_runtime.cpp")
        .include(tensorrt_include)
        .include(cuda_include)
        .warnings(true)
        .compile("raw_trt_runtime");

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-search=native={}", tensorrt_lib.display());

    #[cfg(unix)]
    add_pip_link_name_if_needed(&tensorrt_lib);

    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

#[cfg(feature = "raw-tensorrt")]
fn tensor_rt_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    use std::env;
    use std::path::PathBuf;

    if let Some(root) = env::var_os("TENSORRT_ROOT") {
        return paths_from_root(PathBuf::from(root), "TensorRT");
    }

    match (
        env::var_os("TENSORRT_INCLUDE_DIR"),
        env::var_os("TENSORRT_LIB_DIR"),
    ) {
        (Some(include), Some(library)) => {
            return (PathBuf::from(include), PathBuf::from(library));
        }
        (Some(_), None) | (None, Some(_)) => {
            panic!(
                "raw-tensorrt requires both TENSORRT_INCLUDE_DIR and \
                 TENSORRT_LIB_DIR when either is set"
            );
        }
        (None, None) => {}
    }

    if let Some(python) = env::var_os("TENSORRT_PYTHON") {
        return discover_tensor_rt_with_python(&python);
    }

    panic!(
        "raw-tensorrt could not find TensorRT; set TENSORRT_ROOT, set both \
         TENSORRT_INCLUDE_DIR and TENSORRT_LIB_DIR, or set TENSORRT_PYTHON"
    );
}

#[cfg(feature = "raw-tensorrt")]
fn discover_tensor_rt_with_python(
    python: &std::ffi::OsStr,
) -> (std::path::PathBuf, std::path::PathBuf) {
    use std::process::Command;

    const DISCOVERY: &str = r#"
from pathlib import Path
import tensorrt

package = Path(tensorrt.__file__).resolve().parent
include_candidates = [package / "include", package.parent / "tensorrt" / "include"]
lib_candidates = [package / "libs", package.parent / "tensorrt_libs", package]

include = next((path for path in include_candidates if (path / "NvInfer.h").is_file()), None)
library = next((path for path in lib_candidates if path.is_dir() and any(path.glob("*nvinfer*"))), None)
if include is None or library is None:
    raise SystemExit("TensorRT Python package does not contain headers and libraries")

print(include)
print(library)
"#;

    let output = Command::new(python)
        .args(["-c", DISCOVERY])
        .output()
        .unwrap_or_else(|error| panic!("failed to run TENSORRT_PYTHON: {error}"));
    if !output.status.success() {
        panic!(
            "TENSORRT_PYTHON discovery failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout = String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("TENSORRT_PYTHON returned non-UTF-8 paths: {error}"));
    let mut paths = stdout.lines();
    let include = paths
        .next()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| panic!("TENSORRT_PYTHON did not return an include path"));
    let library = paths
        .next()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| panic!("TENSORRT_PYTHON did not return a library path"));

    (include.into(), library.into())
}

#[cfg(feature = "raw-tensorrt")]
fn cuda_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    use std::env;
    use std::path::PathBuf;

    match (env::var_os("CUDA_INCLUDE_DIR"), env::var_os("CUDA_LIB_DIR")) {
        (Some(include), Some(library)) => return (include.into(), library.into()),
        (Some(_), None) | (None, Some(_)) => {
            panic!("raw-tensorrt requires both CUDA_INCLUDE_DIR and CUDA_LIB_DIR");
        }
        (None, None) => {}
    }

    if let Some(root) = env::var_os("CUDA_ROOT") {
        return paths_from_root(PathBuf::from(root), "CUDA");
    }

    #[cfg(target_os = "linux")]
    for candidate in ["/opt/cuda", "/usr/local/cuda"] {
        let root = PathBuf::from(candidate);
        if root.is_dir() {
            return paths_from_root(root, "CUDA");
        }
    }

    panic!(
        "raw-tensorrt could not find CUDA; set CUDA_ROOT or both \
         CUDA_INCLUDE_DIR and CUDA_LIB_DIR"
    );
}

#[cfg(feature = "raw-tensorrt")]
fn paths_from_root(
    root: std::path::PathBuf,
    product: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let include = root.join("include");
    let library = [root.join("lib"), root.join("lib64")]
        .into_iter()
        .find(|path| path.is_dir())
        .unwrap_or_else(|| panic!("{product} root {} has no lib or lib64", root.display()));

    (include, library)
}

#[cfg(all(feature = "raw-tensorrt", unix))]
fn add_pip_link_name_if_needed(tensorrt_lib: &std::path::Path) {
    use std::env;
    use std::path::PathBuf;

    if tensorrt_lib.join("libnvinfer.so").is_file()
        || tensorrt_lib.join("libnvinfer.dylib").is_file()
    {
        return;
    }

    let versioned = std::fs::read_dir(tensorrt_lib)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libnvinfer.so."))
        });
    let Some(versioned) = versioned
    else {
        return;
    };

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let link_name = output.join("libnvinfer.so");
    if !link_name.exists() {
        std::os::unix::fs::symlink(&versioned, &link_name)
            .unwrap_or_else(|error| panic!("creating TensorRT linker name: {error}"));
    }

    println!("cargo:rustc-link-search=native={}", output.display());
}

#[cfg(feature = "raw-tensorrt")]
fn require_file(path: &std::path::Path, description: &str) {
    assert!(path.is_file(), "missing {description}: {}", path.display());
}

#[cfg(feature = "raw-tensorrt")]
fn require_directory(path: &std::path::Path, description: &str) {
    assert!(path.is_dir(), "missing {description}: {}", path.display());
}
