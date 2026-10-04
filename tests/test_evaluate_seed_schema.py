import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from finance_agent.evaluate_rollouts import JudgeClient, load_dataset


class SeedRubricSchemaTests(unittest.TestCase):
    def _write_dataset(self, rubric: object) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "seed.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["Question", "Answer", "Question Type", "Rubric"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Question": "What was revenue?",
                    "Answer": "Revenue was 10.",
                    "Question Type": "Extraction",
                    "Rubric": json.dumps(rubric),
                }
            )
        return path

    def test_loads_wrapped_seed_rubric_and_preserves_context(self) -> None:
        path = self._write_dataset(
            {
                "question_id": "QR01",
                "max_score": 3,
                "difficulty": "Easy",
                "ability": "Extract revenue.",
                "criteria": [
                    {"id": "QR01.1", "points": 2, "criterion": "Revenue is 10."},
                    {"id": "QR01.2", "points": 1, "criterion": "Uses USD."},
                ],
                "tolerance": "Equivalent units are acceptable.",
                "grading_rule": "Score independently.",
                "contradiction_check": "Do not use 2023 revenue.",
                "contradiction_rule": "A contradiction fails that criterion.",
                "extension": "Provenance metadata that is not grading guidance.",
            }
        )

        item = load_dataset(path)["q001"]

        self.assertEqual(item["query_id"], "QR01")
        self.assertEqual(
            item["rubrics"],
            [
                {
                    "rubric_id": "QR01.1",
                    "rubric_text": "Revenue is 10.",
                    "operator": "correctness",
                    "points": 2.0,
                },
                {
                    "rubric_id": "QR01.2",
                    "rubric_text": "Uses USD.",
                    "operator": "correctness",
                    "points": 1.0,
                },
            ],
        )
        self.assertEqual(item["rubric_context"]["difficulty"], "Easy")
        self.assertEqual(
            item["rubric_context"]["contradiction_check"],
            "Do not use 2023 revenue.",
        )
        self.assertNotIn("extension", item["rubric_context"])

    def test_rejects_max_score_mismatch(self) -> None:
        path = self._write_dataset(
            {
                "max_score": 10,
                "criteria": [
                    {"id": "QR01.1", "points": 2, "criterion": "Revenue is 10."}
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "does not match max_score"):
            load_dataset(path)


class JudgeRubricContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_context_as_unscored_judge_guidance(self) -> None:
        judge = object.__new__(JudgeClient)
        judge.max_retries = 1
        captured: dict[str, Any] = {}

        async def request(user_prompt: str, *, json_mode: bool):
            captured.update(json.loads(user_prompt))
            return {
                "rubric_scores": [
                    {
                        "rubric_id": "QR01.1",
                        "score": 1,
                        "explanation": "Correct.",
                        "evidence": "10",
                    }
                ],
                "summary": "Correct.",
            }

        judge._request = request
        rubrics = [
            {
                "rubric_id": "QR01.1",
                "rubric_text": "Revenue is 10.",
                "points": 2.0,
            }
        ]
        context = {"tolerance": "Equivalent units are acceptable."}

        judgement = await judge.grade(
            question="What was revenue?",
            final_answer="10",
            rubrics=rubrics,
            rubric_context=context,
        )

        self.assertEqual(captured["rubric_context"], context)
        self.assertEqual(len(captured["rubrics"]), 1)
        self.assertEqual(judgement["rubric_scores"][0]["score"], 1)


if __name__ == "__main__":
    unittest.main()
