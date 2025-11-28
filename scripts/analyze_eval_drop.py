"""Analyze evaluation trajectories from outputs and quantify mid-run drop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_eval_curve(path: Path) -> list[tuple[int, float]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    curve: list[tuple[int, float]] = []
    for entry in data:
        if "eval_d4rl_score" not in entry:
            continue
        round_idx = int(entry.get("round", -1)) + 1
        curve.append((round_idx, float(entry["eval_d4rl_score"])))
    return curve


def segment_mean(curve: list[tuple[int, float]], upper: int | None, lower: int = 0) -> float | None:
    seg = [score for round_idx, score in curve if lower < round_idx <= (upper if upper is not None else round_idx)]
    return mean(seg) if seg else None


def summarize(path: Path) -> dict[str, float | None]:
    curve = load_eval_curve(path)
    if not curve:
        return {"file": str(path), "early": None, "mid": None, "late": None}
    return {
        "file": str(path),
        "early": segment_mean(curve, 200),
        "mid": segment_mean(curve, 400, lower=200),
        "late": segment_mean(curve, None, lower=400),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval_d4rl_score drops")
    parser.add_argument("pattern", help="Glob pattern under outputs/ (e.g., 'fed_optidice_stable_hybrid_ratio_fedavg*')")
    args = parser.parse_args()

    base = Path("outputs")
    matches = sorted(base.glob(args.pattern))
    if not matches:
        raise SystemExit(f"No outputs match pattern {args.pattern}")

    print(f"Found {len(matches)} run(s) matching '{args.pattern}'\n")
    header = "{:<60} {:>12} {:>12} {:>12}".format("run", "mean<=200", "mean200-400", ">400")
    print(header)
    print("-" * len(header))
    for run_dir in matches:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        summary = summarize(metrics_path)
        print(
            "{:<60} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                run_dir.name,
                summary["early"] or float("nan"),
                summary["mid"] or float("nan"),
                summary["late"] or float("nan"),
            )
        )


if __name__ == "__main__":
    main()
