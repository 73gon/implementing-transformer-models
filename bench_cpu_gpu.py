"""
CPU/GPU Benchmark Runner v2 - Subprocess-based

Runs each benchmark configuration in a separate subprocess for:
- Fresh GPU memory state for each config
- Natural timeout handling
- Better isolation and stability

Usage:
    python bench_cpu_gpu_v2.py [--cpu-only] [--cuda-only] [--timeout 600]
"""

import argparse
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# Configuration
SEQ_LENS = [64, 128, 256, 512]
BATCH_SIZES = [8, 16, 32, 64]
NUM_REPEATS = 3
DEFAULT_TIMEOUT = 600  # 10 minutes per config

OUTPUT_DIR = Path("benchmarks")
OUTPUT_FILE = OUTPUT_DIR / "results_cpu_gpu.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="CPU/GPU Benchmark Runner v2")
    parser.add_argument("--cpu-only", action="store_true", help="Run CPU benchmarks only")
    parser.add_argument("--cuda-only", action="store_true", help="Run CUDA benchmarks only")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per config in seconds")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (delete existing results)")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS, help="Sequence lengths to test")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES, help="Batch sizes to test")
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS, help="Number of repeats per config")
    return parser.parse_args()


def load_completed_configs(output_file):
    """Load already completed configurations from CSV."""
    completed = set()
    if not output_file.exists():
        return completed

    import csv

    with open(output_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["device"], int(row["seq_len"]), int(row["batch_size"]), int(row["repeat"]))
            completed.add(key)

    return completed


def run_single_benchmark(device, seq_len, batch_size, repeat, output_file, timeout):
    """Run a single benchmark in a subprocess."""
    cmd = [
        sys.executable,
        "bench_single.py",
        device,
        str(seq_len),
        str(batch_size),
        str(repeat),
        str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            # Extract the timing from stdout
            for line in result.stdout.strip().split("\n"):
                if "->" in line:
                    print(line)
            return True
        else:
            print(f"  -> Failed: {result.stderr[:100] if result.stderr else 'Unknown error'}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  -> Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  -> Exception: {e}")
        return False


def main():
    args = parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Determine devices
    if args.cpu_only:
        devices = ["cpu"]
    elif args.cuda_only:
        devices = ["cuda"]
    else:
        devices = ["cpu", "cuda"]

    # Fresh start?
    if args.fresh and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print("Deleted existing results file.")

    # Load completed configs
    completed = load_completed_configs(OUTPUT_FILE)
    print(f"Found {len(completed)} already completed configurations.")

    # Generate all configs
    all_configs = list(
        product(
            devices,
            args.seq_lens,
            args.batch_sizes,
            range(1, args.repeats + 1),
        )
    )

    total = len(all_configs)
    remaining = [(d, s, b, r) for d, s, b, r in all_configs if (d, s, b, r) not in completed]

    print(f"\nTotal configurations: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Timeout per config: {args.timeout}s")
    print("-" * 60)

    if not remaining:
        print("All configurations already completed!")
        return

    # Run remaining configs
    success_count = 0
    fail_count = 0

    for i, (device, seq_len, batch_size, repeat) in enumerate(remaining, 1):
        progress = total - len(remaining) + i
        print(f"\n[{progress}/{total}] {device.upper()} | seq_len={seq_len} | batch={batch_size} | repeat={repeat}")

        start_time = time.time()
        success = run_single_benchmark(device, seq_len, batch_size, repeat, OUTPUT_FILE, args.timeout)
        elapsed = time.time() - start_time

        if success:
            success_count += 1
            print(f"      Completed in {elapsed:.1f}s")
        else:
            fail_count += 1
            print(f"      Failed after {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
