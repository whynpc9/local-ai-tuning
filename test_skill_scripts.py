"""Root discovery shim for the portable skill's test suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path


TEST_FILE = (
    Path(__file__).resolve().parent
    / "skills"
    / "local-ai-inference-tuning"
    / "tests"
    / "test_scripts.py"
)
SPEC = importlib.util.spec_from_file_location("local_ai_inference_tuning_tests", TEST_FILE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ScriptUnitTests = MODULE.ScriptUnitTests
OpenAIStreamIntegrationTest = MODULE.OpenAIStreamIntegrationTest
