include!("../../build_support/libtorch.rs");

fn main() {
    configure(true);
    configure_tch_version();
}

fn configure_tch_version() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets manifest directory");
    let lockfile = std::path::Path::new(&manifest_dir).join("../../Cargo.lock");
    let contents = std::fs::read_to_string(&lockfile).expect("workspace Cargo.lock is readable");
    let version = dependency_version(&contents, "tch").expect("Cargo.lock contains tch");

    println!("cargo:rerun-if-changed={}", lockfile.display());
    println!("cargo:rustc-env=ENGINE_BENCH_TCH_VERSION={version}");
}

fn dependency_version<'a>(lockfile: &'a str, dependency: &str) -> Option<&'a str> {
    lockfile.split("[[package]]").find_map(|package| {
        let mut name = None;
        let mut version = None;

        for line in package.lines() {
            let line = line.trim();
            name = name.or_else(|| line.strip_prefix("name = \"")?.strip_suffix('"'));
            version = version.or_else(|| line.strip_prefix("version = \"")?.strip_suffix('"'));
        }

        if name == Some(dependency) {
            version
        }
        else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::dependency_version;

    #[test]
    fn finds_the_resolved_tch_version() {
        let lockfile = r#"
[[package]]
name = "tch"
version = "0.24.0"
"#;

        assert_eq!(dependency_version(lockfile, "tch"), Some("0.24.0"));
    }
}
