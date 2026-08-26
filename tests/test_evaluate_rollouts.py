import tempfile
import unittest
from pathlib import Path

from finance_agent.evaluate_rollouts import (
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
                "must_have_earned": 1,
                "must_have_possible": 1,
                "must_have_percent": 100,
            },
            "trajectory": {
                "success": True,
                "stop_reason": "done_tool",
                "total_turns": 2,
                "tool_calls_count": 1,
                "input_tokens": 100,
                "output_tokens": 20,
            },
            "judgement": {
                "rubric_scores": [
                    {
                        "rubric_id": "1",
                        "rubric_text": "Criterion",
                        "must_have": True,
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


if __name__ == "__main__":
    unittest.main()
