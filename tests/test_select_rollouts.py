import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from finance_agent.select_rollouts import export_selection, select_items


def _item(percent=70.0, model="model-a", qid="q001", source=None):
    return {
        "status": "ok",
        "model": model,
        "question_id": qid,
        "question": "Question, with\nmultiple lines?",
        "final_answer": "Candidate answer",
        "source_result": source or f"logs/finance/{model}/run-1/{qid}/result.json",
        "score": {
            "earned": percent / 10,
            "possible": 10,
            "percent": percent,
            "must_have_percent": 0,
        },
        "trajectory": {"success": False},
        "judgement": {"rubric_scores": [{"score": 1, "points": 7}]},
    }


def _scores(root, items):
    path = root / "evaluations" / "run" / "scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"judge_model": "judge", "items": items}), encoding="utf-8"
    )
    return path


class SelectRolloutsTests(unittest.TestCase):
    def test_inclusive_whole_question_score_and_error_filter(self):
        items = [_item(69.9999), _item(70), _item(100)]
        failed = _item(100)
        failed["status"] = "error"
        items.append(failed)

        selected, counts = select_items(items)

        self.assertEqual([item["score"]["percent"] for item in selected], [100, 70])
        self.assertEqual(counts["selected"], 2)
        self.assertEqual(counts["evaluation_errors"], 1)
        self.assertEqual(counts["below_threshold"], 1)
        # No additional must-have/agent-success restriction was requested.
        self.assertFalse(selected[0]["trajectory"]["success"])

    def test_rejects_invalid_scores_and_thresholds(self):
        items = [_item(value) for value in (float("nan"), float("inf"), -1, 101)]
        invalid = _item()
        invalid["score"]["percent"] = "70"
        items.append(invalid)
        zero = _item(100)
        zero["score"]["possible"] = 0
        items.append(zero)
        selected, counts = select_items(items)
        self.assertEqual(selected, [])
        self.assertEqual(counts["invalid_scores"], 6)
        for threshold in (-1, 101, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                select_items([], threshold)

    def test_copies_complete_trajectories_without_model_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [_item(model="model-a"), _item(model="model-b")]
            for item in items:
                source = root / item["source_result"]
                turn = source.parent / "turns" / "turn_001" / "result.json"
                turn.parent.mkdir(parents=True)
                turn.write_text('{"turn": 1}', encoding="utf-8")
                source.write_text('{"final_answer": "answer"}', encoding="utf-8")
                (source.parent / "agent.log").write_text("log", encoding="utf-8")
            scores = _scores(root, items)

            manifest = export_selection(
                scores.parent, copy_trajectories=True, project_root=root
            )

            output = Path(manifest["output_dir"])
            self.assertEqual(manifest["counts"]["selected"], 2)
            targets = []
            for item in manifest["items"]:
                target = Path(item["copied_trajectory"])
                targets.append(target)
                self.assertEqual((target / "agent.log").read_text(), "log")
                self.assertTrue(
                    (target / "turns" / "turn_001" / "result.json").is_file()
                )
                self.assertEqual(
                    (target / "result.json").read_bytes(),
                    Path(item["resolved_source_result"]).read_bytes(),
                )
                self.assertEqual(item["judgement"], items[0]["judgement"])
            self.assertNotEqual(targets[0], targets[1])
            with (output / "selected.csv").open(
                encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["question"], items[0]["question"])
            self.assertEqual(
                len((output / "selected_paths.txt").read_text().splitlines()), 2
            )
            self.assertEqual(
                json.loads((output / "selected.json").read_text())["counts"][
                    "selected"
                ],
                2,
            )

    def test_missing_sources_allow_manifest_but_fail_copy_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = _scores(root, [_item()])
            output = root / "copy"
            with self.assertRaisesRegex(ValueError, "missing"):
                export_selection(
                    scores, output_dir=output, copy_trajectories=True, project_root=root
                )
            self.assertFalse(output.exists())
            manifest = export_selection(scores, project_root=root)
            self.assertEqual(manifest["counts"]["selected"], 1)
            self.assertEqual(manifest["counts"]["missing_trajectories"], 1)

    def test_refuses_to_overwrite_previous_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = _scores(root, [_item()])
            manifest = export_selection(scores, project_root=root)
            selection = Path(manifest["output_dir"]) / "selected.json"
            original = selection.read_bytes()
            with self.assertRaisesRegex(ValueError, "empty or new"):
                export_selection(scores, project_root=root)
            self.assertEqual(selection.read_bytes(), original)

    def test_rejects_unsafe_or_mismatched_source_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in (
                "result.json",
                "logs/q002/result.json",
                "logs/q001/turns/q001/result.json",
            ):
                with self.subTest(source=source):
                    scores = _scores(root, [_item(source=source)])
                    with self.assertRaisesRegex(ValueError, "matching qNNN"):
                        export_selection(scores, project_root=root)
            scores = _scores(root, [_item()])
            with self.assertRaisesRegex(ValueError, "inside a source"):
                export_selection(
                    scores,
                    output_dir=root / Path(_item()["source_result"]).parent / "export",
                    project_root=root,
                )

    def test_cli_outputs_valid_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = _scores(root, [_item(69)])
            output = root / "selection"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "finance_agent.select_rollouts",
                    "--eval-path",
                    str(scores),
                    "--output-dir",
                    str(output),
                    "--copy-trajectories",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Selected 0/1", result.stdout)
            self.assertEqual(
                json.loads((output / "selected.json").read_text())["items"], []
            )


if __name__ == "__main__":
    unittest.main()
