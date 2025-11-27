from __future__ import annotations

import os
from pathlib import Path


def ensure_mujoco_env() -> None:
    """Ensure MuJoCo-related environment variables are populated."""

    mujoco_root = Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", "/root/.mujoco/mujoco210"))
    if mujoco_root.exists():
        os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco_root))

    default_paths = ["/usr/lib/x86_64-linux-gnu", str(mujoco_root / "bin")]
    extra = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [path for path in extra.split(":") if path]
    for candidate in default_paths:
        if candidate and Path(candidate).exists() and candidate not in entries:
            entries.insert(0, candidate)
    if entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join(entries)
