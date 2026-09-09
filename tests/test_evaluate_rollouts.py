import csv
import json
import tempfile
import unittest
from pathlib import Path

from finance_agent.evaluate_rollouts import (
    _add_score_totals,
    _aggregate,
    _extract_json_object,
    _normalize_rubrics,
    _validate_judgement,
    discover_results,
    load_dataset,
    render_html_report,
)


class EvaluateRolloutsTests(unittest.TestCase):
    def test_loads_ff_test_question_ids_by_line_number(self):
        dataset = load_dataset(Path("data/FF_test.jsonl"))

        self.assertEqual(len(dataset), 120)
        self.assertIn(
            "What does Block's guidance look like?", dataset["q044"]["question"]
        )
        self.assertGreater(len(dataset["q044"]["rubrics"]), 0)

    def test_loads_synthetic_csv_fields_and_weighted_points(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "synthetic.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["Question", "Answer", "Question Type", "Rubric"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "Who is the CFO?",
                        "Answer": "Example Person",
                        "Question Type": "Company Person",
                        "Rubric": json.dumps(
                            [
                                {"criteria": "Names the CFO", "points": 2},
                                {"criteria": "Gives the date", "points": 0.5},
                            ]
                        ),
                    }
                )

            dataset = load_dataset(dataset_path)

        self.assertEqual(dataset["q001"]["question"], "Who is the CFO?")
        self.assertEqual(dataset["q001"]["reference_answer"], "Example Person")
        self.assertEqual(dataset["q001"]["question_type"], "Company Person")
        self.assertEqual(
            [rubric["points"] for rubric in dataset["q001"]["rubrics"]],
            [2.0, 0.5],
        )

    def test_score_totals_use_rubric_points(self):
        item = {
            "judgement": {
                "rubric_scores": [
                    {"score": 1, "points": 2.0, "must_have": True},
                    {"score": 0, "points": 0.5, "must_have": False},
                ]
            }
        }

        _add_score_totals(item)

        self.assertEqual(item["score"]["earned"], 2.0)
        self.assertEqual(item["score"]["possible"], 2.5)
        self.assertEqual(item["score"]["percent"], 80.0)
        self.assertEqual(item["score"]["rubrics_passed"], 1)

    def test_discovers_latest_duplicate_and_ignores_turn_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "model-a" / "run-1" / "q001" / "result.json"
            newer = root / "model-a" / "run-2" / "q001" / "result.json"
            turn = (
                root
                / "model-a"
                / "run-2"
                / "q001"
                / "turns"
                / "turn_001"
                / "result.json"
            )
            for path in (older, newer, turn):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            older.touch()
            newer.touch()
            newer_mtime = older.stat().st_mtime_ns + 1_000_000
            newer.touch()
            newer.chmod(0o644)
            import os

            os.utime(newer, ns=(newer_mtime, newer_mtime))

            found = discover_results([root])

            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["model"], "model-a")
            self.assertEqual(found[0]["question_id"], "q001")
            self.assertEqual(found[0]["result_path"], newer)

    def test_parses_fenced_json_and_validates_rubric_order(self):
        rubrics = _normalize_rubrics(
            [
                {"rubric_id": 1, "rubric_text": "First", "must_have": True},
                {"rubric_id": 2, "rubric_text": "Second"},
            ]
        )
        payload = _extract_json_object(
            """```json
{"rubric_scores":[
  {"rubric_id":"2","score":0,"explanation":"missing","evidence":""},
  {"rubric_id":"1","score":1,"explanation":"present","evidence":"quote"}
],"summary":"ok"}
```"""
        )

        judgement = _validate_judgement(payload, rubrics)

        self.assertEqual(
            [item["rubric_id"] for item in judgement["rubric_scores"]], ["1", "2"]
        )
        self.assertEqual([item["score"] for item in judgement["rubric_scores"]], [1, 0])

    def test_aggregate_includes_completion_turns_tools_and_score_buckets(self):
        def item(percent, success, turns, tool_usage):
            return {
                "status": "ok",
                "score": {
                    "earned": percent,
                    "possible": 100,
                    "percent": percent,
                    "rubrics_passed": 1,
                    "rubrics_total": 1,
                    "must_have_earned": 0,
                    "must_have_possible": 0,
                    "must_have_percent": 0,
                },
                "trajectory": {
                    "success": success,
                    "total_turns": turns,
                    "tool_usage": tool_usage,
                },
            }

        items = [
            item(0, True, 10, {"web_search": 2}),
            item(10, False, 20, {"parse_html_page": 4}),
            item(99.9, True, 30, {"web_search": 1}),
            item(100, False, 40, {}),
        ]
        items.append(
            {
                "status": "error",
                "trajectory": {"success": False, "total_turns": 50, "tool_usage": {}},
            }
        )

        summary = _aggregate(items)

        self.assertEqual(summary["agent_successes"], 2)
        self.assertEqual(summary["questions"], 5)
        self.assertEqual(summary["agent_completion_rate_percent"], 40)
        self.assertEqual(summary["trajectory_count"], 5)
        self.assertEqual(summary["average_turns"], 30)
        self.assertEqual(
            summary["tool_call_totals"], {"parse_html_page": 4.0, "web_search": 3.0}
        )
        self.assertEqual(
            summary["average_tool_calls"], {"parse_html_page": 0.8, "web_search": 0.6}
        )
        self.assertEqual(summary["score_distribution"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])

    def test_renders_self_contained_report(self):
        item = {
            "status": "ok",
            "model": "model-a",
            "question_id": "q001",
            "question": "Question?",
            "final_answer": "Answer.",
            "score": {
                "earned": 1,
                "possible": 1,
                "percent": 100,
                "rubrics_passed": 1,
                "rubrics_total": 1,
                "must_have_earned": 1,
                "must_have_possible": 1,
                "must_have_percent": 100,
            },
            "trajectory": {
                "success": True,
                "stop_reason": "done_tool",
                "total_turns": 2,
                "tool_calls_count": 1,
                "tool_usage": {"web_search": 1},
                "input_tokens": 100,
                "output_tokens": 20,
            },
            "judgement": {
                "rubric_scores": [
                    {
                        "rubric_id": "1",
                        "rubric_text": "Criterion",
                        "must_have": True,
                        "points": 1.0,
                        "score": 1,
                        "explanation": "Present",
                        "evidence": "Answer",
                    }
                ]
            },
        }
        payload = {
            "generated_at": "2026-01-01T00:00:00Z",
            "judge_model": "judge",
            "items": [item],
            "summary": _aggregate([item]),
        }

        report = render_html_report(payload)

        self.assertIn("Finance Agent Rubric Evaluation", report)
        self.assertIn("model-a", report)
        self.assertIn("100.0%", report)
        self.assertIn("Score distribution", report)
        self.assertIn("90–100: 1", report)
        self.assertIn("1/1 (100.0%)", report)
        self.assertIn("average turns", report)
        self.assertIn("web_search", report)


if __name__ == "__main__":
    unittest.main()
