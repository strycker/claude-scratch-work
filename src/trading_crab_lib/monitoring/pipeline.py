"""Monitoring helpers for pipeline-level validation and health (C4.2–C4.5)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── C4.2: Tactics summary ────────────────────────────────────────────────


def format_tactics_summary(tactics_df: pd.DataFrame) -> str:
    """Format a tactics DataFrame as a count summary by classification.

    Parameters
    ----------
    tactics_df :
        Output of ``classify_tactics()`` with a ``classification`` column.
    """
    if tactics_df.empty or "classification" not in tactics_df.columns:
        return "  (no tactics data)"

    counts = tactics_df["classification"].value_counts()
    total = len(tactics_df)
    lines = ["  Tactics classification summary:"]
    for cls in ["buy_hold", "swing", "stand_aside"]:
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        bar = "#" * int(pct / 5)
        lines.append(f"    {cls:<14}  {n:3d} assets  ({pct:5.1f}%)  {bar}")
    # Any other classifications
    other = [c for c in counts.index if c not in {"buy_hold", "swing", "stand_aside"}]
    for cls in other:
        n = counts[cls]
        pct = n / total * 100 if total else 0
        lines.append(f"    {cls:<14}  {n:3d} assets  ({pct:5.1f}%)")
    lines.append(f"    TOTAL          {total:3d} assets")
    return "\n".join(lines)


# ── C4.3: Step output validation ─────────────────────────────────────────


@dataclass
class StepValidation:
    """Result of validating a step's output."""

    step_num: int = 0
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def summary(self) -> str:
        lines = [f"  Step {self.step_num} output validation:"]
        for name, ok, detail in self.checks:
            status = "OK" if ok else "WARN"
            lines.append(f"    [{status:>4}] {name}: {detail}")
        return "\n".join(lines)


def validate_step_output(
    step_num: int,
    outputs: dict[str, pd.DataFrame | None],
    *,
    max_nan_pct: float = 0.5,
) -> StepValidation:
    """Validate step outputs: check shape, NaN%, and dtypes.

    Parameters
    ----------
    step_num :
        Pipeline step number for reporting.
    outputs :
        ``{name: DataFrame}`` of the step's outputs.
    max_nan_pct :
        Maximum acceptable NaN fraction (0.0–1.0) per column before warning.

    Returns
    -------
    StepValidation with per-output checks.
    """
    validation = StepValidation(step_num=step_num)

    for name, df in outputs.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            validation.checks.append((f"{name} exists", False, "missing or empty"))
            continue

        # Shape check
        validation.checks.append(
            (f"{name} shape", True, f"{df.shape[0]} rows × {df.shape[1]} cols")
        )

        # NaN check
        if hasattr(df, "isna"):
            nan_frac = df.isna().mean()
            bad_cols = nan_frac[nan_frac > max_nan_pct]
            if len(bad_cols) > 0:
                worst = bad_cols.idxmax()
                validation.checks.append(
                    (f"{name} NaN%", False,
                     f"{len(bad_cols)} cols > {max_nan_pct:.0%} NaN (worst: {worst} at {bad_cols[worst]:.1%})")
                )
            else:
                validation.checks.append(
                    (f"{name} NaN%", True, f"all cols below {max_nan_pct:.0%}")
                )

        # Dtype check — warn if all object columns (no numeric)
        if hasattr(df, "select_dtypes"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0 and len(df.columns) > 0:
                validation.checks.append(
                    (f"{name} dtypes", False, "no numeric columns found")
                )

    log.info("Step %d validation:\n%s", step_num, validation.summary())
    return validation


# ── C4.4 + C4.5: Pipeline timing and health summary ─────────────────────


@dataclass
class PipelineHealthSummary:
    """Tracks step timings, plot counts, and warning counts."""

    step_timings: dict[int, float] = field(default_factory=dict)
    steps_run: list[int] = field(default_factory=list)
    steps_failed: list[int] = field(default_factory=list)
    total_warnings: int = 0

    def record_step(self, step_num: int, elapsed: float, *, failed: bool = False) -> None:
        self.step_timings[step_num] = elapsed
        if failed:
            self.steps_failed.append(step_num)
        else:
            self.steps_run.append(step_num)

    def summary(self) -> str:
        total_elapsed = sum(self.step_timings.values())
        lines = [
            "═" * 60,
            "  Pipeline Health Summary",
            "═" * 60,
            f"  Steps completed: {len(self.steps_run)}/{len(self.steps_run) + len(self.steps_failed)}",
        ]
        if self.steps_failed:
            lines.append(f"  Steps FAILED:    {self.steps_failed}")

        lines.append("")
        lines.append("  Step timings:")
        for step_num in sorted(self.step_timings):
            elapsed = self.step_timings[step_num]
            status = "FAIL" if step_num in self.steps_failed else "OK"
            lines.append(f"    Step {step_num}: {elapsed:6.1f}s  [{status}]")
        lines.append(f"    TOTAL: {total_elapsed:6.1f}s")
        lines.append("═" * 60)
        return "\n".join(lines)
