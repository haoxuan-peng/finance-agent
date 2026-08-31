"""Export a retest dataset from long, low-scoring evaluated trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, TypeGuard

from .evaluate_rollouts import _field


def _single_line(text: str) -> str:
    """The runner reads one question per line; use the same text in both files."""
    return " ".join(text.split())


def load_source_dataset(path: Path) -> dict[str, dict[str, Any]]:
    """Match evaluator row numbering while preserving the complete original rubric."""
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        if path.suffix.lower() == ".csv":
            question_fields = ("question", "query", "prompt")
            answer_fields = ("answer", "reference_answer", "gold_answer")
            rows = enumerate(csv.DictReader(file), start=1)
        elif path.suffix.lower() == ".jsonl":
            question_fields = ("query", "question", "prompt")
            answer_fields = ("reference_answer", "answer", "gold_answer")
            rows = (
                (index, json.loads(line))
                for index, line in enumerate(file, start=1)
                if line.strip()
            )
        else:
            raise ValueError("--dataset must be a CSV or JSONL file")
        for index, raw in rows:
            qid = f"q{index:03d}"
            if not isinstance(raw, dict):
                raise ValueError(f"Dataset record {qid} must be an object")
            rubrics = _field(raw, "rubrics", "rubric")
            if isinstance(rubrics, str):
                try:
                    rubrics = json.loads(rubrics or "[]")
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid rubric JSON for {qid}: {error}"
                    ) from error
            records[qid] = {
                "question": _field(raw, *question_fields),
                "answer": _field(raw, *answer_fields),
                "question_type": _field(raw, "question_type", "type"),
                "query_id": _field(raw, "query_id", "question_id"),
                "query_date": _field(raw, "query_date"),
                "rubrics": rubrics,
            }
    return records


def _valid_number(value: Any) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def select_questions(
    items: list[dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
    *,
    min_turns: int = 40,
    max_score: float = 10.0,
    num_questions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Require both strict thresholds on the same trajectory, then deduplicate."""
    if isinstance(min_turns, bool) or not isinstance(min_turns, int) or min_turns < 0:
        raise ValueError("--min-turns must be non-negative")
    if not _valid_number(max_score) or not 0 <= max_score <= 100:
        raise ValueError("--max-score must be a finite percentage between 0 and 100")
    if num_questions is not None and (
        isinstance(num_questions, bool)
        or not isinstance(num_questions, int)
        or num_questions < 1
    ):
        raise ValueError("--num-questions must be positive")
    counts = {
        "total_trajectories": len(items),
        "evaluation_errors": 0,
        "invalid_metrics": 0,
        "qualifying_trajectories": 0,
        "qualifying_questions": 0,
        "selected_questions": 0,
    }
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation item must be a JSON object")
        if item.get("status") != "ok":
            counts["evaluation_errors"] += 1
            continue
        score = item.get("score") or {}
        trajectory = item.get("trajectory") or {}
        percent = score.get("percent") if isinstance(score, dict) else None
        possible = score.get("possible") if isinstance(score, dict) else None
        turns = trajectory.get("total_turns") if isinstance(trajectory, dict) else None
        if (
            not _valid_number(percent)
            or not 0 <= percent <= 100
            or not _valid_number(possible)
            or possible <= 0
            or isinstance(turns, bool)
            or not isinstance(turns, int)
            or turns < 0
        ):
            counts["invalid_metrics"] += 1
            continue
        if turns > min_turns and percent < max_score:
            candidates.append(item)
    counts["qualifying_trajectories"] = len(candidates)
    candidates.sort(
        key=lambda item: (
            item["score"]["percent"],
            -item["trajectory"]["total_turns"],
            str(item.get("question_id", "")),
            str(item.get("model", "")),
        )
    )

    rows = []
    seen = set()
    for item in candidates:
        qid = item.get("question_id")
        if not isinstance(qid, str) or qid not in dataset:
            raise ValueError(
                f"No dataset record for {qid!r}; use the original eval dataset"
            )
        record = dataset[qid]
        question = record.get("question")
        evaluated_question = item.get("question")
        if not isinstance(question, str) or not _single_line(question):
            raise ValueError(f"Dataset question {qid} is empty or not a string")
        question = _single_line(question)
        if not isinstance(evaluated_question, str) or question != _single_line(
            evaluated_question
        ):
            raise ValueError(
                f"Question text mismatch for {qid}; --dataset must have the same "
                "questions and row order as the original evaluation"
            )
        # Validate every qualifying match, including duplicate model attempts.
        rubrics = record.get("rubrics")
        if (
            not isinstance(rubrics, list)
            or not rubrics
            or any(
                not isinstance(rubric, dict)
                or not isinstance(
                    rubric.get("rubric_text", rubric.get("criteria")), str
                )
                or not rubric.get("rubric_text", rubric.get("criteria", "")).strip()
                for rubric in rubrics
            )
        ):
            raise ValueError(f"Missing or invalid rubrics for {qid}")
        if qid in seen:
            continue
        seen.add(qid)
        rows.append(
            {
                "Question": question,
                "Answer": record.get("answer") or "",
                "Question Type": record.get("question_type") or "",
                "Rubric": json.dumps(rubrics, ensure_ascii=False),
                "query_id": record.get("query_id") or "",
                "query_date": record.get("query_date") or "",
                "source_question_id": qid,
                "source_model": item.get("model", ""),
                "source_total_turns": item["trajectory"]["total_turns"],
                "source_score_percent": item["score"]["percent"],
            }
        )
    counts["qualifying_questions"] = len(rows)
    if num_questions is not None:
        rows = rows[:num_questions]
    counts["selected_questions"] = len(rows)
    return rows, counts


def export_questions(
    eval_path: Path,
    dataset: Path,
    *,
    min_turns: int = 40,
    max_score: float = 10.0,
    num_questions: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if eval_path.is_dir():
        eval_path = eval_path / "scores.json"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Expected an evaluator scores.json with an 'items' array")
    rows, counts = select_questions(
        payload["items"],
        load_source_dataset(dataset),
        min_turns=min_turns,
        max_score=max_score,
        num_questions=num_questions,
    )
    output_dir = (
        output_dir
        or eval_path.parent / f"questions_turns_gt{min_turns}_score_lt{max_score:g}"
    ).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty or new: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "questions.txt"
    csv_path = output_dir / "questions.csv"
    txt_path.write_text(
        "".join(row["Question"] + "\n" for row in rows), encoding="utf-8"
    )
    columns = [
        "Question",
        "Answer",
        "Question Type",
        "Rubric",
        "query_id",
        "query_date",
        "source_question_id",
        "source_model",
        "source_total_turns",
        "source_score_percent",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return {"counts": counts, "txt_path": str(txt_path), "csv_path": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export questions with turns > 40 and whole-question score < 10% (no API calls)"
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        required=True,
        help="scores.json file or its evaluation directory",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Original CSV/JSONL used for this evaluation, in its original row order",
    )
    parser.add_argument(
        "--num-questions",
        "--limit",
        dest="num_questions",
        type=int,
        help="Maximum distinct questions to export (default: all matches)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=40,
        help="Exclusive minimum turns: 40 means at least 41 (default: 40)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=10.0,
        help="Exclusive maximum whole-question percentage (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New/empty output directory (default: a subdirectory of the evaluation)",
    )
    return parser


def main_sync() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = export_questions(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Selection failed: {error}\n")
    counts = result["counts"]
    print(
        f"Matched {counts['qualifying_trajectories']} trajectories with turns > {args.min_turns} "
        f"and score < {args.max_score:g}%; {counts['qualifying_questions']} unique questions."
    )
    print(
        f"Exported {counts['selected_questions']} questions (lowest score, then most turns first)."
    )
    print(
        f"Skipped: {counts['evaluation_errors']} evaluation errors, {counts['invalid_metrics']} invalid metrics."
    )
    if (
        args.num_questions is not None
        and counts["selected_questions"] < args.num_questions
    ):
        print(
            f"Warning: requested {args.num_questions}, but only {counts['selected_questions']} questions qualify."
        )
    print(f"Retest TXT: {result['txt_path']}")
    print(f"Evaluation CSV: {result['csv_path']}")
    print(
        "Use these two files together: exported row 1 maps to new q001, row 2 to q002, etc."
    )


if __name__ == "__main__":
    main_sync()
