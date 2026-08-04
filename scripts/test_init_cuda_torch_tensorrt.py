#!/usr/bin/env python3
"""Regression tests for the CUDA/Torch/TensorRT initializer."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("init_cuda_torch_tensorrt.py")
SPEC = importlib.util.spec_from_file_location("cuda_init", SCRIPT)
assert SPEC and SPEC.loader
cuda_init = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cuda_init)


class PythonExecutablePathTests(unittest.TestCase):
    def test_explicit_python_symlink_keeps_virtualenv_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "base-python"
            target.write_text("#! /bin/sh\n", encoding="utf-8")
            target.chmod(0o755)
            venv_python = root / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            os.symlink(target, venv_python)

            configured = cuda_init.explicit_path(
                str(venv_python), option="--cuda-torch-python", executable=True
            )

            self.assertEqual(configured, venv_python.absolute())
            self.assertNotEqual(configured, target.resolve())


if __name__ == "__main__":
    unittest.main()
