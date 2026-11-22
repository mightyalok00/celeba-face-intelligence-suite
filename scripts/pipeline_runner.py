"""
CelebA Face AI & Intelligence System - Deep Functional Pipeline Runner
Executes comprehensive validation of data invariants, MCP protocols,
inference math, and calibrated benchmark models.

Usage:
    python scripts/pipeline_runner.py --stage all
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pipeline_verification import (
    verify_functional_benchmarks,
    verify_functional_data,
    verify_functional_inference,
    verify_functional_mcp,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CelebA Face AI Functional Pipeline Suite")
    parser.add_argument(
        "--stage",
        choices=["all", "data_integrity", "mcp_protocol_check", "inference_check", "benchmark_check"],
        default="all",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("  🚀 CelebA Face AI - Deep Functional Pipeline & Quality Suite", flush=True)
    print(f"  Execution Mode: {args.stage.upper()}", flush=True)
    print("=" * 70, flush=True)

    success = True
    if args.stage in ["all", "data_integrity"]:
        success = success and verify_functional_data()
    if args.stage in ["all", "mcp_protocol_check"]:
        success = success and verify_functional_mcp()
    if args.stage in ["all", "inference_check"]:
        success = success and verify_functional_inference()
    if args.stage in ["all", "benchmark_check"]:
        success = success and verify_functional_benchmarks()

    elapsed = round(time.perf_counter() - start_time, 2)
    print("\n" + "=" * 70, flush=True)
    if success:
        print(f"  ✅ ALL FUNCTIONAL PIPELINE VERIFICATIONS PASSED (Completed in {elapsed}s)", flush=True)
    else:
        print(f"  ❌ PIPELINE FAILED (Completed in {elapsed}s)", flush=True)
    print("=" * 70 + "\n", flush=True)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
