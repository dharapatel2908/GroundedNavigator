"""
run_eval.py
Runs all scenarios from a JSON file and writes results to CSV.

Usage:
    python eval/run_eval.py --scenarios eval/scenarios/sample.json --output results.csv
"""

import argparse
import csv
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_episode


def run_eval(scenarios_path: str, output_csv: str) -> None:
    with open(scenarios_path) as f:
        scenarios = json.load(f)

    results = []
    for i, s in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Scene={s['scene']}  Goal={s['goal']}")
        result = run_episode(scene=s["scene"], goal=s["goal"], verbose=True)
        results.append(result)

    # Write CSV
    fieldnames = ["scene", "goal", "success", "steps", "latency_s"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    n = len(results)
    success_rate = sum(r["success"] for r in results) / n * 100
    avg_steps = sum(r["steps"] for r in results) / n
    avg_latency = sum(r["latency_s"] for r in results) / n

    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY  ({n} episodes)")
    print(f"{'='*50}")
    print(f"  Success rate  : {success_rate:.1f}%")
    print(f"  Avg steps     : {avg_steps:.1f}")
    print(f"  Avg latency   : {avg_latency:.1f}s")
    print(f"  Results saved : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NavAgent eval harness")
    parser.add_argument("--scenarios", default="eval/scenarios/sample.json")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()
    run_eval(args.scenarios, args.output)
