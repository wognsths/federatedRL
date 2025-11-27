from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


def _format_eval_line(label: str, stats: Mapping[str, float]) -> str:
    metrics = ", ".join(
        f"{key}={float(value):.6f}" for key, value in sorted(stats.items()) if key.startswith("eval_")
    )
    return f"{label} | {metrics}" if metrics else ""


def append_eval_entry(log_path: Path, label: str, stats: Mapping[str, float]) -> None:
    line = _format_eval_line(label, stats)
    if not line:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def write_eval_log(history: Iterable[Mapping[str, float]], log_path: Path, label_key: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with log_path.open("w", encoding="utf-8") as fp:
        for entry in history:
            stats = {k: v for k, v in entry.items() if k.startswith("eval_")}
            if not stats:
                continue
            label_value = int(entry.get(label_key, -1))
            label = f"[{label_key} {label_value}]"
            fp.write(_format_eval_line(label, stats) + "\n")
            count += 1
    return count
