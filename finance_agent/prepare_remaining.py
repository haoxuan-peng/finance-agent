"""Prepare an ID-preserving question file for retrying unfinished rollouts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .question_files import QUESTION_ID_PATTERN, load_question_file


RUN_DIRECTORY_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[0-9A-Za-z]+"
)


def _is_completed_result(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    answer = result.get("final_answer")
    return (
        result.get("final_error") is None
        and isinstance(answer, str)
        and bool(answer.strip())
    )


def _question_directories(root: Path) -> list[Path]:
    """Find top-level qNNN result directories, excluding per-turn folders."""
    candidates = [root, *root.rglob("*")]
    question_dirs = []
    for candidate in candidates:
        if not candidate.is_dir() or not QUESTION_ID_PATTERN.fullmatch(candidate.name):
            continue
        relative_parts = candidate.relative_to(root).parts
        if "turns" in relative_parts:
            continue
        question_dirs.append(candidate)
    return question_dirs


def _delete_safe_incomplete_run_dirs(
    question_status: dict[Path, bool],
    remaining_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Delete timestamped run dirs only when every contained question is unfinished."""
    candidate_runs = {
        question_dir.parent
        for question_dir, is_completed in question_status.items()
        if not is_completed
        and question_dir.name in remaining_ids
        and RUN_DIRECTORY_PATTERN.fullmatch(question_dir.parent.name)
    }
    deleted = []
    skipped = []
    for run_dir in sorted(candidate_runs):
        if run_dir.is_symlink():
            skipped.append(str(run_dir))
            continue
        direct_question_dirs = [
            child
            for child in run_dir.iterdir()
            if child.is_dir() and QUESTION_ID_PATTERN.fullmatch(child.name)
        ]
        safe_to_delete = bool(direct_question_dirs) and all(
            child.name in remaining_ids
            and question_status.get(child) is False
            and not child.is_symlink()
            for child in direct_question_dirs
        )
        if not safe_to_delete:
            skipped.append(str(run_dir))
            continue
        shutil.rmtree(run_dir)
        deleted.append(str(run_dir))
    return deleted, skipped


def prepare_remaining(
    question_file: Path,
    logs_path: list[Path],
    output_file: Path,
    delete_incomplete_runs: bool = False,
) -> dict[str, Any]:
    questions, question_ids = load_question_file(question_file)
    if question_file.resolve() == output_file.resolve():
        raise ValueError("Output file must not overwrite the original question file")
    known_ids = set(question_ids)
    completed = set()
    unreadable_results = 0
    missing_results = 0
    unknown_ids = set()
    question_status: dict[Path, bool] = {}
    for root in logs_path:
        if not root.is_dir():
            raise ValueError(f"Log directory does not exist: {root}")
        for question_dir in _question_directories(root):
            qid = question_dir.name
            if qid not in known_ids:
                unknown_ids.add(qid)
                continue
            result_file = question_dir / "result.json"
            if not result_file.is_file():
                missing_results += 1
                question_status[question_dir] = False
                continue
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                unreadable_results += 1
                question_status[question_dir] = False
                continue
            if not isinstance(result, dict):
                unreadable_results += 1
                question_status[question_dir] = False
                continue
            is_completed = _is_completed_result(result)
            question_status[question_dir] = is_completed
            if is_completed:
                completed.add(qid)

    remaining = [
        (qid, question)
        for qid, question in zip(question_ids, questions)
        if qid not in completed
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{qid}\t{question}\n" for qid, question in remaining), encoding="utf-8"
    )
    temporary.replace(output_file)
    deleted_run_dirs: list[str] = []
    skipped_deletion_dirs: list[str] = []
    if delete_incomplete_runs:
        deleted_run_dirs, skipped_deletion_dirs = _delete_safe_incomplete_run_dirs(
            question_status,
            {qid for qid, _ in remaining},
        )
    return {
        "total": len(questions),
        "completed": len(completed),
        "remaining": len(remaining),
        "remaining_ids": [qid for qid, _ in remaining],
        "unreadable_results": unreadable_results,
        "missing_results": missing_results,
        "unknown_question_ids": sorted(unknown_ids),
        "deleted_run_dirs": deleted_run_dirs,
        "skipped_deletion_dirs": skipped_deletion_dirs,
        "output_file": str(output_file),
    }


def main_sync() -> None:
    parser = argparse.ArgumentParser(
        description="Find unfinished rollouts and preserve original IDs in a retry TXT"
    )
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument(
        "--logs-path",
        type=Path,
        nargs="+",
        required=True,
        help="Only runs using this same dataset and original question IDs",
    )
    parser.add_argument("--output-file", type=Path, default=Path("data/remaining.txt"))
    parser.add_argument(
        "--delete-incomplete-runs",
        action="store_true",
        help=(
            "Permanently delete timestamped run directories for unfinished questions. "
            "A directory containing any completed or unknown question is kept."
        ),
    )
    args = parser.parse_args()
    try:
        summary = prepare_remaining(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Could not prepare remaining questions: {error}\n")
    print(f"Total questions: {summary['total']}")
    print(f"Completed: {summary['completed']}")
    print(f"Remaining: {summary['remaining']}")
    print(f"Remaining question IDs: {summary['remaining_ids']}")
    print(f"Output: {summary['output_file']}")
    if summary["unreadable_results"]:
        print(
            f"Warning: {summary['unreadable_results']} unreadable results were ignored."
        )
    if summary["missing_results"]:
        print(
            f"Warning: {summary['missing_results']} question directories had no result.json."
        )
    if summary["unknown_question_ids"]:
        print(
            f"Warning: unknown question IDs were ignored: {summary['unknown_question_ids']}"
        )
    if summary["deleted_run_dirs"]:
        print(f"Deleted incomplete run directories: {summary['deleted_run_dirs']}")
    if summary["skipped_deletion_dirs"]:
        print(
            "Warning: unsafe or mixed run directories were not deleted: "
            f"{summary['skipped_deletion_dirs']}"
        )
    print(
        "Retry using this TXT; evaluate the new trajectories against the original full dataset."
    )


if __name__ == "__main__":
    main_sync()
