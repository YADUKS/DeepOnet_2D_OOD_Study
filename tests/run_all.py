"""Run independent test groups without mixing native FE and ML runtimes."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="2")
for module in ("tests.test_design", "tests.test_protocol", "tests.test_generation", "tests.test_cli", "tests.test_output_names", "tests.test_models", "tests.test_workflow", "tests.test_fe_queries", "tests.test_tuning_report", "tests.test_solver"):
    subprocess.run([sys.executable, "-m", "unittest", module, "-v"], cwd=root, env=env, check=True)
print("All tests passed.")
