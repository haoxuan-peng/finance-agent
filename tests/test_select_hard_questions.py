import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from finance_agent.evaluate_rollouts import load_dataset
from finance_agent.select_hard_questions import (
    export_questions,
    load_source_dataset,
    select_questions,
)


def _item(
    qid="q001",
    question="Question?",
    turns: Any = 41,
    percent: float = 0,
    model="model-a",
):
    return {
        "status": "ok",
        "question_id": qid,
        "question": question,
        "model": model,
        "score": {"percent": percent, "possible": 10},
        "trajectory": {"total_turns": turns, "success": False},
    }


def _record(question="Question?"):
    return {
        "question": question,
        "answer": "Reference answer",
        "question_type": "Research",
        "query_id": "original-id",
        "query_date": "2026-08-01",
        "rubrics": [
            {
                "criteria": "Criterion",
                "points": 2.5,
                "must_have": True,
                "rubric_type": "Analysis",
                "extra_metadata": ["preserve"],
            }
        ],
    }


def _write_inputs(root, items, records):
    evaluation = root / "scores.json"
    evaluation.write_text(json.dumps({"items": items}), encoding="utf-8")
    dataset = root / "dataset.csv"
    with dataset.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Question", "Answer", "Question Type", "Rubric"]
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "Question": record["question"],
                    "Answer": record["answer"],
                    "Question Type": record["question_type"],
                    "Rubric": json.dumps(record["rubrics"]),
                }
            )
    return evaluation, dataset


class SelectHardQuestionsTests(unittest.TestCase):
    def test_both_bounds_are_strict_and_apply_to_same_trajectory(self):
        items = [
            _item("q001", turns=40, percent=0),
            _item("q002", turns=41, percent=10),
            _item("q003", turns=41, percent=9.9999),
            _item("q004", turns=100, percent=0),
            _item("q005", turns=100, percent=30),
            _item("q005", turns=20, percent=0, model="model-b"),
        ]
        records = {f"q{i:03d}": _record() for i in range(1, 6)}
        rows, counts = select_questions(items, records)
        self.assertEqual([row["source_question_id"] for row in rows], ["q004", "q003"])
        self.assertEqual(counts["qualifying_trajectories"], 2)

    def test_deduplicates_before_limit_and_uses_score_then_turns(self):
        items = [
            _item("q001", percent=5),
            _item("q001", percent=0, model="model-b"),
            _item("q002", turns=80),
            _item("q003", turns=60),
        ]
        records = {f"q{i:03d}": _record() for i in range(1, 4)}
        rows, counts = select_questions(items, records, num_questions=2)
        self.assertEqual([row["source_question_id"] for row in rows], ["q002", "q003"])
        self.assertEqual(counts["qualifying_trajectories"], 4)
        self.assertEqual(counts["qualifying_questions"], 3)
        all_rows, _ = select_questions(items, records)
        self.assertEqual(all_rows[-1]["source_model"], "model-b")

    def test_invalid_metrics_and_evaluation_errors_do_not_qualify(self):
        failed = _item()
        failed["status"] = "error"
        bad_possible = _item()
        bad_possible["score"]["possible"] = 0
        items = [
            failed,
            bad_possible,
            _item(percent=float("nan")),
            _item(percent=-1),
            _item(turns=True),
            _item(turns="80"),
        ]
        rows, counts = select_questions(items, {"q001": _record()})
        self.assertEqual(rows, [])
        self.assertEqual(counts["evaluation_errors"], 1)
        self.assertEqual(counts["invalid_metrics"], 5)

    def test_csv_and_txt_roundtrip_preserves_rubrics_and_reindexes_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _record('Question, with "quotes"\nsecond line?')
            second = _record("Another question?")
            evaluation, dataset = _write_inputs(
                root,
                [
                    _item("q002", record["question"], percent=5),
                    _item("q003", second["question"], percent=0),
                ],
                [_record("Unselected"), record, second],
            )
            output = root / "export"
            result = export_questions(evaluation, dataset, output_dir=output)
            self.assertEqual(
                sorted(p.name for p in output.iterdir()),
                ["questions.csv", "questions.txt"],
            )
            questions = (
                Path(result["txt_path"]).read_text(encoding="utf-8").splitlines()
            )
            reloaded = load_dataset(Path(result["csv_path"]))
            self.assertEqual(
                questions, [second["question"], 'Question, with "quotes" second line?']
            )
            self.assertEqual([v["question"] for v in reloaded.values()], questions)
            self.assertEqual(reloaded["q002"]["reference_answer"], "Reference answer")
            self.assertEqual(reloaded["q002"]["rubrics"][0]["points"], 2.5)
            with Path(result["csv_path"]).open(
                encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["source_question_id"] for r in rows], ["q003", "q002"])
            self.assertEqual(json.loads(rows[1]["Rubric"]), record["rubrics"])

    def test_jsonl_blank_lines_keep_original_ids_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "original.jsonl"
            raw = {
                "query": "Third\nquestion?",
                "query_id": "uuid",
                "query_date": "2026-08-01",
                "rubrics": [
                    {
                        "rubric_id": 7,
                        "rubric_text": "Expected answer",
                        "must_have": True,
                    }
                ],
            }
            dataset.write_text(
                json.dumps({"query": "First", "rubrics": []})
                + "\n\n"
                + json.dumps(raw)
                + "\n",
                encoding="utf-8",
            )
            records = load_source_dataset(dataset)
            self.assertNotIn("q002", records)
            self.assertEqual(records["q003"]["question"], raw["query"])
            evaluation = root / "scores.json"
            evaluation.write_text(
                json.dumps({"items": [_item("q003", "Third question?")]}),
                encoding="utf-8",
            )
            result = export_questions(evaluation, dataset)
            reloaded = load_dataset(Path(result["csv_path"]))
            self.assertEqual(reloaded["q001"]["query_id"], "uuid")
            self.assertEqual(reloaded["q001"]["query_date"], "2026-08-01")
            self.assertEqual(reloaded["q001"]["rubrics"][0]["rubric_id"], 7)
            self.assertEqual(reloaded["q001"]["rubrics"][0]["points"], 1.0)

    def test_missing_or_mismatched_question_and_rubrics_fail(self):
        with self.assertRaisesRegex(ValueError, "No dataset record"):
            select_questions([_item()], {})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            select_questions([_item()], {"q001": _record("Wrong question")})
        record = _record()
        record["rubrics"] = []
        with self.assertRaisesRegex(ValueError, "rubrics"):
            select_questions([_item()], {"q001": record})

    def test_cli_warns_when_fewer_questions_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, dataset = _write_inputs(root, [_item()], [_record()])
            output = root / "export"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "finance_agent.select_hard_questions",
                    "--eval-path",
                    str(evaluation),
                    "--dataset",
                    str(dataset),
                    "--num-questions",
                    "100",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("requested 100, but only 1", result.stdout)
            self.assertEqual(
                len((output / "questions.txt").read_text().splitlines()), 1
            )

    def test_no_matches_produces_empty_txt_and_csv_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, dataset = _write_inputs(root, [_item(turns=40)], [_record()])
            result = export_questions(evaluation, dataset)
            self.assertEqual(result["counts"]["selected_questions"], 0)
            self.assertEqual(Path(result["txt_path"]).read_text(), "")
            self.assertIn(
                "Rubric", Path(result["csv_path"]).read_text(encoding="utf-8-sig")
            )

    def test_rejects_invalid_arguments_and_existing_output(self):
        cases: list[dict[str, Any]] = [
            {"min_turns": -1},
            {"max_score": float("nan")},
            {"num_questions": 0},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_questions([], {}, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, dataset = _write_inputs(root, [_item()], [_record()])
            result = export_questions(evaluation, dataset)
            original = Path(result["csv_path"]).read_bytes()
            with self.assertRaisesRegex(ValueError, "empty or new"):
                export_questions(evaluation, dataset)
            self.assertEqual(Path(result["csv_path"]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
