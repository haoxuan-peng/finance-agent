"""Select high-scoring trajectories from an existing rubric evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def select_items(
    items: list[dict[str, Any]], min_score: float = 70.0
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use the unrounded whole-question percentage, including the threshold."""
    if not math.isfinite(min_score) or not 0 <= min_score <= 100:
        raise ValueError("--min-score must be a finite percentage between 0 and 100")
    counts = {
        "total": len(items),
        "selected": 0,
        "below_threshold": 0,
        "evaluation_errors": 0,
        "invalid_scores": 0,
    }
    selected = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation item must be a JSON object")
        if item.get("status") != "ok":
            counts["evaluation_errors"] += 1
            continue
        score = item.get("score")
        percent = score.get("percent") if isinstance(score, dict) else None
        possible = score.get("possible") if isinstance(score, dict) else None
        if (
            isinstance(percent, bool)
            or not isinstance(percent, (int, float))
            or not math.isfinite(percent)
            or not 0 <= percent <= 100
            or isinstance(possible, bool)
            or not isinstance(possible, (int, float))
            or not math.isfinite(possible)
            or possible <= 0
        ):
            counts["invalid_scores"] += 1
            continue
        if percent >= min_score:
            selected.append(dict(item))
        else:
            counts["below_threshold"] += 1
    selected.sort(
        key=lambda item: (
            -item["score"]["percent"],
            str(item.get("model", "")),
            str(item.get("question_id", "")),
        )
    )
    counts["selected"] = len(selected)
    return selected, counts


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.")[:100] or "unknown"


def _source_result(item: dict[str, Any], project_root: Path) -> Path:
    raw = item.get("source_result")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing source_result for {item.get('question_id')}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    # Never copy an arbitrary ancestor directory from a malformed score record.
    if (
        path.name != "result.json"
        or not re.fullmatch(r"q\d+", path.parent.name)
        or path.parent.name != item.get("question_id")
        or "turns" in path.parts
    ):
        raise ValueError(
            f"Expected a matching qNNN/result.json for {item.get('question_id')}: {path}"
        )
    return path


def export_selection(
    eval_path: Path,
    *,
    min_score: float = 70.0,
    output_dir: Path | None = None,
    copy_trajectories: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Write a selection manifest and optionally copy complete question folders.

    Original logs are never changed. Nonempty destinations are rejected so that
    reruns cannot overwrite files or leave stale, no-longer-selected trajectories.
    """
    if eval_path.is_dir():
        eval_path = eval_path / "scores.json"
    eval_path = eval_path.resolve()
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Expected an evaluator scores.json with an 'items' array")
    selected, counts = select_items(payload["items"], min_score)
    project_root = (project_root or Path.cwd()).resolve()
    output_dir = (
        output_dir or eval_path.parent / f"selected_ge_{min_score:g}"
    ).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty or new: {output_dir}")

    missing = []
    copies: dict[Path, Path] = {}
    for item in selected:
        source = _source_result(item, project_root)
        if output_dir == source.parent or output_dir.is_relative_to(source.parent):
            raise ValueError("Output directory cannot be inside a source trajectory")
        item["resolved_source_result"] = str(source)
        item["trajectory_available"] = source.is_file()
        item["copied_trajectory"] = None
        if not source.is_file():
            missing.append(str(source))
        if copy_trajectories:
            # The run name and source hash preserve separate attempts/models even
            # when several paths sanitize to the same directory name.
            digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            run_name = _safe_component(source.parent.parent.name)
            model = _safe_component(str(item.get("model", "unknown")))
            target = (
                output_dir
                / "trajectories"
                / model
                / f"{run_name}__{digest}"
                / source.parent.name
            )
            copies[target] = source.parent
            item["copied_trajectory"] = str(target)

    # Preflight all selected sources before creating any output in copy mode.
    if copy_trajectories and missing:
        examples = "\n".join(missing[:5])
        raise ValueError(
            f"{len(missing)} selected result.json file(s) are missing:\n{examples}\n"
            "Use --project-root for relative source paths, or omit "
            "--copy-trajectories to export only the selection list."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for target, source_dir in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target, symlinks=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_file": str(eval_path),
        "judge_model": payload.get("judge_model"),
        "dataset": payload.get("dataset"),
        "selection_rule": {
            "field": "score.percent",
            "operator": ">=",
            "value": min_score,
        },
        "output_dir": str(output_dir),
        "copy_trajectories": copy_trajectories,
        "counts": {**counts, "missing_trajectories": len(missing)},
        "selected_by_model": dict(Counter(str(item.get("model")) for item in selected)),
        "items": selected,
    }
    (output_dir / "selected.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "model",
        "question_id",
        "percent",
        "earned",
        "possible",
        "question",
        "source_result",
        "copied_trajectory",
        "trajectory_available",
    ]
    with (output_dir / "selected.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "model": item.get("model"),
                    "question_id": item.get("question_id"),
                    "percent": item["score"]["percent"],
                    "earned": item["score"].get("earned"),
                    "possible": item["score"]["possible"],
                    "question": item.get("question"),
                    "source_result": item["resolved_source_result"],
                    "copied_trajectory": item["copied_trajectory"],
                    "trajectory_available": item["trajectory_available"],
                }
            )
    (output_dir / "selected_paths.txt").write_text(
        "".join(item["resolved_source_result"] + "\n" for item in selected),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select rollout trajectories by whole-question evaluation percentage (no API calls)"
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        required=True,
        help="scores.json file or its evaluation directory",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=70.0,
        help="Inclusive minimum percentage, 0–100 (default: 70)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New/empty directory; default: <evaluation-dir>/selected_ge_70",
    )
    parser.add_argument(
        "--copy-trajectories",
        action="store_true",
        help="Also copy the complete selected qNNN directories, including turns",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Resolve relative source_result paths against this root (default: cwd)",
    )
    return parser


def main_sync() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest = export_selection(**vars(args))
    except (OSError, ValueError, shutil.Error) as error:
        parser.exit(1, f"Selection failed: {error}\n")
    counts = manifest["counts"]
    print(
        f"Selected {counts['selected']}/{counts['total']} trajectories with score >= {args.min_score:g}%"
    )
    print(
        f"Skipped: {counts['evaluation_errors']} evaluation errors, {counts['invalid_scores']} invalid scores"
    )
    for model, count in sorted(manifest["selected_by_model"].items()):
        print(f"  {model}: {count}")
    if counts["missing_trajectories"]:
        print(
            f"Warning: {counts['missing_trajectories']} source trajectories are not available locally"
        )
    print(f"Selection output: {manifest['output_dir']}")


if __name__ == "__main__":
    main_sync()
