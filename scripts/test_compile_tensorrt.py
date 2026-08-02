import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"test_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RawCompilerTests(unittest.TestCase):
    def test_import_does_not_import_tensorrt(self):
        previous = sys.modules.pop("tensorrt", None)
        try:
            load_script("compile_tensorrt_raw")
            self.assertNotIn("tensorrt", sys.modules)
        finally:
            if previous is not None:
                sys.modules["tensorrt"] = previous

    def test_commands_are_explicit(self):
        raw = load_script("compile_tensorrt_raw")

        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            raw.parser().parse_args(["--input", "model.ts"])

        export = raw.parser().parse_args(
            [
                "export-onnx",
                "--input", "model.ts",
                "--output", "model.onnx",
                "--channels", "63",
                "--precision", "fp16",
            ]
        )
        self.assertEqual(export.operation, "export-onnx")

    def test_exact_fp16_mismatch_fails_closed(self):
        raw = load_script("compile_tensorrt_raw")

        with self.assertRaisesRegex(RuntimeError, "requires fp16 I/O"):
            raw.validate_exact_precision("fp16", "fp32", "fp32")

    def test_mixed_mode_without_fp16_flag_fails(self):
        raw = load_script("compile_tensorrt_raw")

        class BuilderFlag:
            pass

        fake_trt = type("Trt", (), {"BuilderFlag": BuilderFlag, "__version__": "11.0"})

        with self.assertRaisesRegex(RuntimeError, "cannot provide mixed fp16 tactics"):
            raw.configure_precision(fake_trt, object(), "mixed-fp32-io-fp16-tactics")

    def test_result_is_stable_json(self):
        raw = load_script("compile_tensorrt_raw")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            raw.write_result({"z": 1, "schema_version": 1}, path)

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"schema_version": 1, "z": 1}\n',
            )

    def test_export_result_keeps_its_operation(self):
        raw = load_script("compile_tensorrt_raw")
        fake_torch = type(
            "Torch",
            (),
            {"__version__": "1", "version": type("Version", (), {"cuda": "2"})},
        )

        result = raw.onnx_export_result(fake_torch, "fp16", 0.5)

        self.assertEqual(result["operation"], "export-onnx")

    def test_build_result_contains_environment_and_profile(self):
        raw = load_script("compile_tensorrt_raw")
        args = type(
            "Args",
            (),
            {
                "precision": "fp16",
                "min_batch_size": 1,
                "opt_batch_size": 8,
                "max_batch_size": 16,
            },
        )
        environment = {
            "operation": "inspect-environment",
            "tensorrt_version": "11",
            "cuda_runtime_version": "13",
            "gpu_name": "test gpu",
            "compute_capability": "9.0",
            "python_version": "3",
        }

        result = raw.engine_build_result(args, 1.0, "fp16", "fp16", environment)

        self.assertEqual(result["operation"], "build-engine")
        self.assertEqual(result["gpu_name"], "test gpu")
        self.assertEqual(result["compute_capability"], "9.0")
        self.assertEqual(result["profile"], {"min": 1, "opt": 8, "max": 16})

    def test_stdout_contains_only_json(self):
        raw = load_script("compile_tensorrt_raw")
        output = io.StringIO()

        with mock.patch("sys.stdout", output):
            raw.write_result({"schema_version": 1}, None)

        self.assertEqual(json.loads(output.getvalue()), {"schema_version": 1})


class TorchCompilerTests(unittest.TestCase):
    def test_import_does_not_require_torch(self):
        previous_torch = sys.modules.pop("torch", None)
        previous_trt = sys.modules.pop("torch_tensorrt", None)
        try:
            module = load_script("compile_tensorrt")
            self.assertEqual(module.RESULT_SCHEMA_VERSION, 1)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("torch_tensorrt", sys.modules)
        finally:
            if previous_torch is not None:
                sys.modules["torch"] = previous_torch
            if previous_trt is not None:
                sys.modules["torch_tensorrt"] = previous_trt


if __name__ == "__main__":
    unittest.main()
