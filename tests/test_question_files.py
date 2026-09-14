import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_library.base import LLMConfig

from finance_agent.get_agent import Parameters
from finance_agent.prepare_remaining import prepare_remaining
from finance_agent.question_files import load_question_file
from finance_agent.run_agent import run_tests_parallel


class QuestionFileTests(unittest.TestCase):
    def test_plain_questions_keep_existing_numbering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.txt"
            path.write_text(
                "\ufeffFirst question\n\n Second question \n", encoding="utf-8"
            )
            self.assertEqual(
                load_question_file(path),
                (["First question", "Second question"], ["q001", "q002"]),
            )

    def test_explicit_ids_are_preserved_and_removed_from_prompt_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remaining.txt"
            path.write_text(
                "q003\tThird question\nq116\tQuestion 116\nq1000\tQuestion 1000\n",
                encoding="utf-8",
            )
            questions, ids = load_question_file(path)
            self.assertEqual(ids, ["q003", "q116", "q1000"])
            self.assertEqual(
                questions, ["Third question", "Question 116", "Question 1000"]
            )

    def test_invalid_files_are_rejected(self):
        cases = [
            "q003\tQuestion\nPlain question\n",
            "q003\tA\nq003\tB\n",
            "q003\t\n",
            "q1\tQuestion\n",
            "\n\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.txt"
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_question_file(path)

    def test_remaining_file_preserves_original_ids_and_ignores_turn_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.txt"
            questions.write_text("First\nSecond\nThird\n", encoding="utf-8")
            logs = root / "logs"
            records = {
                "run/q001/result.json": {"final_error": None, "final_answer": "Done"},
                "run/q002/result.json": {
                    "final_error": {"type": "Error"},
                    "final_answer": "Partial",
                },
                "run/q003/result.json": {"final_error": None, "final_answer": "  "},
                "run/q002/turns/q002/result.json": {
                    "final_error": None,
                    "final_answer": "Intermediate",
                },
                "run/q999/result.json": {
                    "final_error": None,
                    "final_answer": "Other dataset",
                },
            }
            for name, payload in records.items():
                path = logs / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "remaining.txt"
            summary = prepare_remaining(questions, [logs], output)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["remaining_ids"], ["q002", "q003"])
            self.assertEqual(summary["unknown_question_ids"], ["q999"])
            self.assertEqual(output.read_text(), "q002\tSecond\nq003\tThird\n")

    def test_retrying_a_remaining_file_still_preserves_ids_above_999(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "remaining.txt"
            questions.write_text(
                "q116\tQuestion A\nq1000\tQuestion B\n", encoding="utf-8"
            )
            result = root / "logs" / "run" / "q1000" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text(
                json.dumps({"final_error": None, "final_answer": "Done"}),
                encoding="utf-8",
            )
            output = root / "remaining2.txt"
            summary = prepare_remaining(questions, [root / "logs"], output)
            self.assertEqual(summary["remaining_ids"], ["q116"])
            self.assertEqual(output.read_text(), "q116\tQuestion A\n")

    def test_original_question_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.txt"
            questions.write_text("Question\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare_remaining(questions, [root], questions)
            self.assertEqual(questions.read_text(), "Question\n")


class RunnerQuestionIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_passes_original_ids_to_agent_and_records_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            class FakeAgent:
                async def run(self, inputs, question_id):
                    calls.append((question_id, inputs[0].text))
                    return SimpleNamespace(
                        success=True,
                        final_error=None,
                        total_turns=1,
                        final_answer="Done",
                        output_dir=root / question_id,
                        model_dump=lambda mode: {"final_answer": "Done"},
                    )

            parameters = Parameters(model_name="test", llm_config=LLMConfig())
            with (
                patch("finance_agent.run_agent.get_agent", return_value=FakeAgent()),
                patch("sys.stdout", new=io.StringIO()),
            ):
                results = await run_tests_parallel(
                    ["Question A", "Question B"], 2, parameters, ["q116", "q1000"]
                )
            self.assertCountEqual([qid for qid, _ in calls], ["q116", "q1000"])
            self.assertEqual(
                [item["question_id"] for item in results], ["q116", "q1000"]
            )
            self.assertNotIn("q116\t", dict(calls)["q116"])
            written = json.loads((root / "results.json").read_text())
            self.assertEqual(written[0]["question_id"], "q116")


if __name__ == "__main__":
    unittest.main()
