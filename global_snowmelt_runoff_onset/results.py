"""Persist manuscript-cited numbers and tables as CSVs.

Convention: every number quoted in the manuscript must exist in a ``results/``
file written by the notebook that computes it — notebook cell outputs and
figure annotations are not provenance. Notebooks call ``save_result_table``
with a small DataFrame right after computing the numbers they print/plot.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _git_short_sha() -> Optional[str]:
    """Best-effort short SHA of the repo HEAD (None outside a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def save_result_table(
    df: pd.DataFrame,
    name: str,
    version: Optional[str] = None,
    results_dir: Union[str, Path] = "results",
) -> Path:
    """Write ``df`` to ``<results_dir>[/<version>]/<name>.csv`` with provenance.

    Appends ``_version``, ``_git_sha``, and ``_written_at`` (UTC ISO) columns
    (underscore-prefixed to avoid collisions with data columns) and creates the
    directory if needed. ``results_dir`` is resolved relative to the notebook's
    working directory, so each evaluation folder keeps its own ``results/``.

    Pass ``version`` for outputs derived from a specific dataset version (they
    land in a ``results/<version>/`` subdirectory and never overwrite another
    version's tables); omit it for version-independent results.
    """
    out_dir = Path(results_dir) / version if version else Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamped = df.copy()
    stamped["_version"] = version
    stamped["_git_sha"] = _git_short_sha()
    stamped["_written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    path = out_dir / f"{name}.csv"
    stamped.to_csv(path, index=False)
    print(f"wrote {len(stamped)} row(s) -> {path}")
    return path
