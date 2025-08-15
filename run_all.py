import subprocess
import sys

commands = [
    ["venv\\Scripts\\python", "-m", "backend.ingestion.pipeline", "data/raw/beef_picadillo.pdf"],
    ["venv\\Scripts\\python", "-m", "backend.ingestion.pipeline", "data/raw/chicken_curry.pdf"],
    ["venv\\Scripts\\python", "-m", "backend.tests.evaluate_ragas"]
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        # sys.exit(result.returncode)
    else:
        print("Command succeeded.")
