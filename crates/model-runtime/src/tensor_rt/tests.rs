use super::*;

#[test]
fn batch_shapes_reject_invalid_ranges() {
    assert!(TensorRtBatchShapes {
        min: 32,
        optimal: 16,
        max: 64
    }
    .validate()
    .is_err());
}

#[test]
fn compiler_protocol_rejects_an_unknown_schema() {
    let path = std::env::temp_dir().join(format!(
        "engine-zoo-compiler-result-{}.json",
        std::process::id()
    ));
    fs::write(
        &path,
        r#"{"schema_version":2,"operation":"inspect-environment","tensorrt_version":"1","cuda_runtime_version":"1","gpu_name":"gpu","compute_capability":"1","python_version":"1"}"#,
    )
    .unwrap();

    let error = read_compiler_result(&path).unwrap_err();
    let _ = fs::remove_file(path);

    assert!(error
        .to_string()
        .contains("unsupported TensorRT compiler result schema"));
}

#[test]
fn exact_fp16_requires_fp16_io() {
    assert_eq!(expected_dtype(TensorRtBuildPrecision::Fp16), "fp16");
    assert_eq!(
        expected_dtype(TensorRtBuildPrecision::MixedFp32IoFp16Tactics),
        "fp32"
    );
}

#[test]
fn application_compiler_disables_timing_cache() {
    let compiler = TensorRtCompiler::application(
        PathBuf::from("python"),
        None,
        PathBuf::from("compile.py"),
        TensorRtBatchShapes {
            min: 1,
            optimal: 1,
            max: 1,
        },
        TensorRtBuildPrecision::Fp16,
    );

    assert_eq!(compiler.timing_cache, TimingCachePolicy::Disabled);

    let process = compiler_process(&compiler);
    assert_eq!(process.program, PathBuf::from("python"));
    assert_eq!(process.args, [OsString::from("compile.py")]);

    let command = process.command();
    assert_eq!(command.get_program(), OsStr::new("python"));
    assert_eq!(
        command.get_args().collect::<Vec<_>>(),
        [OsStr::new("compile.py")]
    );
    assert!(command
        .get_envs()
        .any(|(name, value)| name == OsStr::new("LD_PRELOAD") && value.is_none()));
}

#[cfg(unix)]
#[test]
fn fake_compiler_protocol_reuses_and_rebuilds_torchscript_artifacts() {
    assert_fake_compiler_protocol(CompiledBackendKind::TensorRtTorchScript);
}

#[cfg(unix)]
#[test]
fn fake_compiler_protocol_reuses_and_rebuilds_raw_artifacts() {
    assert_fake_compiler_protocol(CompiledBackendKind::TensorRtRaw);
}

#[cfg(unix)]
fn assert_fake_compiler_protocol(backend: CompiledBackendKind) {
    let fixture = FakeCompilerFixture::new();
    let model = ModelSpec::connect4_basic(1, 4);
    let checkpoint = fixture.root.join("model.safetensors");
    let var_store = nn::VarStore::new(Device::Cpu);
    let _network = Network::new(&var_store.root(), &model).unwrap();
    var_store.save(&checkpoint).unwrap();

    let output = fixture.root.join("model.engine");
    let compiler = TensorRtCompiler::application(
        PathBuf::from("/bin/sh"),
        (backend == CompiledBackendKind::TensorRtRaw).then(|| fixture.exporter.clone()),
        fixture.script.clone(),
        TensorRtBatchShapes {
            min: 1,
            optimal: 2,
            max: 4,
        },
        TensorRtBuildPrecision::Fp32,
    );

    let cold = compiler
        .ensure_artifact(&checkpoint, &model, backend, &output, Device::Cpu)
        .unwrap();
    assert!(!cold.reused);
    assert_eq!(fixture.calls(), expected_calls(backend));
    assert!(!fixture.calls().contains("--timing-cache"));
    assert!(fixture.calls().contains("--min-batch-size 1"));
    assert!(fixture.calls().contains("--opt-batch-size 2"));
    assert!(fixture.calls().contains("--max-batch-size 4"));
    assert!(fixture.calls().contains("--precision fp32"));

    let before_warm = fixture.calls();
    let warm = compiler
        .ensure_artifact(&checkpoint, &model, backend, &output, Device::Cpu)
        .unwrap();
    assert!(warm.reused);
    assert_eq!(
        fixture.calls().strip_prefix(&before_warm).unwrap(),
        "inspect-environment\n"
    );

    fs::remove_file(&output).unwrap();
    fs::write(&output, b"corrupt artifact").unwrap();
    let before_rebuild = fixture.calls();
    let rebuilt = compiler
        .ensure_artifact(&checkpoint, &model, backend, &output, Device::Cpu)
        .unwrap();
    assert!(!rebuilt.reused);
    assert_eq!(
        fixture.calls().strip_prefix(&before_rebuild).unwrap(),
        expected_calls(backend)
    );
    artifact::validate_artifact(&output).unwrap();
}

#[cfg(unix)]
fn expected_calls(backend: CompiledBackendKind) -> &'static str {
    match backend {
        CompiledBackendKind::TensorRtTorchScript => {
            "inspect-environment\ncompile --input --min-batch-size 1 --opt-batch-size 2 --max-batch-size 4 --precision fp32\n"
        }
        CompiledBackendKind::TensorRtRaw => {
            "inspect-environment\nexporter-program\nexport-onnx --input\nbuild-engine --input --min-batch-size 1 --opt-batch-size 2 --max-batch-size 4 --precision fp32\n"
        }
    }
}

#[cfg(unix)]
struct FakeCompilerFixture {
    root: PathBuf,
    script: PathBuf,
    exporter: PathBuf,
    _cleanup: CleanupDirectory,
}

#[cfg(unix)]
impl FakeCompilerFixture {
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;

        let root = artifact::unique_sibling(&std::env::temp_dir().join("engine-zoo-fake"));
        fs::create_dir(&root).unwrap();
        let script = root.join("compiler.sh");
        let exporter = root.join("exporter.sh");
        fs::write(&script, FAKE_COMPILER_SCRIPT).unwrap();
        fs::write(
            &exporter,
            "#!/bin/sh\nprintf 'exporter-program\\n' >> \"$1.calls\"\nexec /bin/sh \"$@\"\n",
        )
        .unwrap();
        fs::set_permissions(&exporter, fs::Permissions::from_mode(0o700)).unwrap();

        Self {
            script,
            exporter,
            _cleanup: CleanupDirectory::new(root.clone(), false),
            root,
        }
    }

    fn calls(&self) -> String {
        fs::read_to_string(self.script.with_extension("sh.calls")).unwrap_or_default()
    }
}

#[cfg(unix)]
const FAKE_COMPILER_SCRIPT: &str = r#"#!/bin/sh
set -eu

log="$0.calls"
operation="$1"
shift

if [ "$operation" = "inspect-environment" ]; then
    result=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--result-json" ]; then result="$2"; shift 2; else shift; fi
    done
    printf 'inspect-environment\n' >> "$log"
    printf '%s\n' '{"schema_version":1,"operation":"inspect-environment","tensorrt_version":"test-trt","cuda_runtime_version":"test-cuda","gpu_name":"test-gpu","compute_capability":"9.0","python_version":"3.test"}' > "$result"
    exit 0
fi

if [ "$operation" = "export-onnx" ]; then
    output=""
    result=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --output) output="$2"; shift 2 ;;
            --result-json) result="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    printf 'export-onnx --input\n' >> "$log"
    printf 'onnx' > "$output"
    printf '%s\n' '{"schema_version":1,"operation":"export-onnx"}' > "$result"
    exit 0
fi

kind="tensor-rt-torch-script"
label="compile"
if [ "$operation" = "build-engine" ]; then
    kind="tensor-rt-raw"
    label="build-engine"
else
    set -- "$operation" "$@"
fi

output=""
result=""
minimum=""
optimal=""
maximum=""
precision=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) output="$2"; shift 2 ;;
        --result-json) result="$2"; shift 2 ;;
        --min-batch-size) minimum="$2"; shift 2 ;;
        --opt-batch-size) optimal="$2"; shift 2 ;;
        --max-batch-size) maximum="$2"; shift 2 ;;
        --precision) precision="$2"; shift 2 ;;
        *) shift ;;
    esac
done
printf '%s --input --min-batch-size %s --opt-batch-size %s --max-batch-size %s --precision %s\n' "$label" "$minimum" "$optimal" "$maximum" "$precision" >> "$log"
printf 'artifact' > "$output"
printf '{"schema_version":1,"operation":"%s","artifact_kind":"%s","input_dtype":"fp32","output_dtype":"fp32","internal_precision":"%s","tensorrt_version":"test-trt","cuda_runtime_version":"test-cuda","gpu_name":"test-gpu","compute_capability":"9.0","python_version":"3.test","profile":{"min":%s,"opt":%s,"max":%s}}\n' "$label" "$kind" "$precision" "$minimum" "$optimal" "$maximum" > "$result"
"#;
