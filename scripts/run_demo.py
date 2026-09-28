"""Run the demo from a source checkout without package installation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vessel_repair.demo import DEFAULT_OUTDIR, run_demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.outdir), indent=2))
