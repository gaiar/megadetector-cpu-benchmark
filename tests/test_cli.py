import os
import subprocess
import sys
import unittest
from pathlib import Path


class BenchmarkCLITest(unittest.TestCase):
    def test_help_succeeds_without_megadetector(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        # Avoid picking up user site packages to simulate a minimal environment
        env["PYTHONNOUSERSITE"] = "1"
        # Ensure stdout/stderr are captured for assertions
        result = subprocess.run(
            [sys.executable, "-S", "benchmark.py", "--help"],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"CLI help failed with stderr:\n{result.stderr}",
        )
        self.assertIn("MegaDetector CPU Benchmark", result.stdout)
        self.assertNotIn("MegaDetector is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
