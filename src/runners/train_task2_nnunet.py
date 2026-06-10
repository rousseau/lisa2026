#!/usr/bin/env python
"""Training entrypoint placeholder for RUN_0003a (nnU-Net v2).

This project currently provides evaluation for nnU-Net predictions but does not
ship an integrated Python training launcher equivalent to other runners.
Use the dedicated nnU-Net CLI workflow documented in `results/runs/RUN_0003a/`.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RUN_0003a nnU-Net training launcher (placeholder)."
    )
    parser.add_argument("--config", required=True, help="Path to run config YAML.")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test_dash")
    parser.parse_args()

    print(
        "[ERROR] nnU-Net training runner is not implemented in Python for this repo.\n"
        "Use the nnU-Net v2 CLI workflow (plan_and_preprocess + nnUNetv2_train)\n"
        "as documented in results/runs/RUN_0003a/implementation_plan.md."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
