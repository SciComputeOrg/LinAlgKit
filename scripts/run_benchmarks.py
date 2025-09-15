#!/usr/bin/env python3
"""
Run C++ benchmarks and export results to CSV, and optionally plot.
Requires: the C++ benchmarks built at ./bin/benchmarks/matrix_benchmarks
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # optional
except Exception:
    plt = None


def run_benchmarks(exe: Path, output_csv: Path) -> None:
    # Google Benchmark supports --benchmark_out
    cmd = [str(exe), "--benchmark_out=%s" % str(output_csv), "--benchmark_out_format=csv"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Wrote:", output_csv)


def plot_csv(csv_path: Path, out_png: Path) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    # Very simple plot: runtime vs benchmark name
    names, times = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name")
            real_time = row.get("real_time")
            if not name or not real_time:
                continue
            try:
                names.append(name)
                times.append(float(real_time))
            except ValueError:
                pass
    if not names:
        print("No data to plot")
        return
    plt.figure(figsize=(10, 6))
    plt.barh(names, times)
    plt.xlabel("Time (ns)")
    plt.title("LinAlgKit Benchmarks")
    plt.tight_layout()
    plt.savefig(out_png)
    print("Saved plot:", out_png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", type=Path, default=Path("build"))
    p.add_argument("--exe", type=Path, default=None, help="Path to benchmark executable")
    p.add_argument("--csv", type=Path, default=Path("benchmark_results.csv"))
    p.add_argument("--plot", type=Path, default=None, help="Path to output PNG plot (optional)")
    args = p.parse_args()

    exe = args.exe
    if exe is None:
        exe = args.build_dir / "bin" / "benchmarks" / "matrix_benchmarks"
    if not exe.exists():
        print("Benchmark executable not found:", exe)
        sys.exit(1)

    run_benchmarks(exe, args.csv)

    if args.plot is not None:
        plot_csv(args.csv, args.plot)


if __name__ == "__main__":
    main()
