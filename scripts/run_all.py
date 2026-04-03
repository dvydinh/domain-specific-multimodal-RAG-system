"""
Master orchestration script for ingestion and evaluation.

Usage: python -m scripts.run_all
"""

import subprocess
import sys
import os

# Resolve project root (one level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python")

commands = [
    [PYTHON, "-m", "backend.ingestion.pipeline", "data/raw/beef_picadillo.pdf"],
    [PYTHON, "-m", "backend.ingestion.pipeline", "data/raw/chicken_curry.pdf"],
    [PYTHON, "-m", "backend.tests.evaluate_ragas"],
]

if __name__ == "__main__":
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}")
        else:
            print("Command succeeded.")
