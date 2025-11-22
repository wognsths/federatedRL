from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import typer

from metrics.rounds_to_x import rounds_to_target

app = typer.Typer(pretty_exceptions_enable=False)


def _load_history(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


@app.command()
def summary(
    metrics: List[Path] = typer.Argument(..., exists=True, file_okay=True, dir_okay=True),
    target: float | None = typer.Option(None, help="Optional target metric (uses 'actor_loss')."),
) -> None:
    files: List[Path] = []
    for item in metrics:
        if item.is_dir():
            files.extend(sorted(item.glob("**/metrics.json")))
        else:
            files.append(item)

    if not files:
        raise typer.BadParameter("No metrics files found")

    for path in files:
        history = _load_history(path)
        final_metrics = history[-1] if history else {}
        typer.echo(f"\n{path} :: steps={len(history)} final={final_metrics}")
        if target is not None:
            curve = [entry.get("actor_loss", entry.get("dual_loss", 0.0)) for entry in history]
            hit = rounds_to_target(curve, target)
            typer.echo(f"  rounds_to_target={hit}")


if __name__ == "__main__":
    app()
