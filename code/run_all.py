# Run the analysis in order.
# From the project folder:  python code/run_all.py

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

steps = [
    "1_process_data.py",
    "2_explore.py",
    "3_scalar_models.py",
    "4_network_model.py",
    "5_ekf_model.py",
    "6_compare.py",
    "7_sensitivity.py",
    "8_interventions.py",
]

if __name__ == "__main__":
    for name in steps:
        print("\n====", name, "====")
        runpy.run_path(str(HERE / name), run_name="__main__")
    print("\nfinished. figures/ and results/ are ready.")
