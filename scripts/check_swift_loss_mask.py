"""Verify actual Qwen/ms-swift token labels without loading model weights."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path


def load_plugin() -> None:
    path = Path(__file__).with_name("swift_per_call_loss.py")
    spec = importlib.util.spec_from_file_location("finance_swift_call_mask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def check_synthetic(template) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Test independent call masks.",
            "parameters": {
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
        },
    }
    # These four cases detect the first-call-only bug in both directions.
    for flags in [[False, True], [True, False], [False, False], [True, True]]:
        messages = [
            {"role": "system", "content": "MASK_SYSTEM_7283"},
            {"role": "user", "content": "MASK_USER_7283"},
            {
                "role": "assistant",
                "content": "<think>MASK_REASONING_7283</think>\nPlanning.",
            },
        ]
        markers = []
        for index, enabled in enumerate(flags):
            marker = f"MASK_CALL_{index}_7283"
            markers.append(marker)
            messages.append(
                {
                    "role": "tool_call",
                    "content": json.dumps(
                        {"name": "probe", "arguments": {"payload": marker}}
                    ),
                    "loss": enabled,
                    "loss_scale": 1.0 if enabled else 0.0,
                }
            )
        for index in range(len(flags)):
            messages.append(
                {"role": "tool_response", "content": f"MASK_OBSERVATION_{index}_7283"}
            )
        messages.append(
            {"role": "assistant", "content": "<think>Recovered.</think>MASK_FINAL_7283"}
        )
        data = {"tools": json.dumps([tool]), "messages": messages}
        encoded = template.encode(copy.deepcopy(data))
        plain = copy.deepcopy(data)
        for message in plain["messages"]:
            message.pop("loss", None)
            message.pop("loss_scale", None)
        baseline = template.encode(plain)
        assert encoded["input_ids"] == baseline["input_ids"], (
            "Plugin changed rendered/tokenized context"
        )
        visible = template.safe_decode(encoded["input_ids"])
        supervised = template.safe_decode(encoded["labels"])
        assert "MASK_FINAL_7283" in supervised
        assert "MASK_REASONING_7283" in supervised
        for index, (marker, enabled) in enumerate(zip(markers, flags)):
            assert marker in visible
            assert (marker in supervised) == enabled, (
                f"Call {index}, flags {flags}: mask failed"
            )
            assert f"MASK_OBSERVATION_{index}_7283" in visible
            assert f"MASK_OBSERVATION_{index}_7283" not in supervised
        assert "MASK_USER_7283" not in supervised
        assert "MASK_SYSTEM_7283" not in supervised
        print(f"Synthetic flags {flags}: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-9B", help="Model ID or local model directory"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--masked-samples", type=int, default=8)
    args = parser.parse_args()
    from swift import get_processor, get_template

    load_plugin()
    template = get_template(
        get_processor(args.model),
        agent_template="qwen3_5",
        template_backend="swift",
        loss_scale="default",
        is_binary_loss_scale=True,
        max_length=None,
    )
    template.set_mode("train")
    check_synthetic(template)
    if args.dataset:
        checked = 0
        with args.dataset.open(encoding="utf-8") as source:
            for line in source:
                data = json.loads(line)
                if not any(
                    m["role"] == "tool_call" and m.get("loss") is False
                    for m in data["messages"]
                ):
                    continue
                encoded = template.encode(copy.deepcopy(data))
                baseline = copy.deepcopy(data)
                for message in baseline["messages"]:
                    message.pop("loss", None)
                    message.pop("loss_scale", None)
                plain = template.encode(baseline)
                assert encoded["input_ids"] == plain["input_ids"], data.get("id")
                assert any(label != -100 for label in encoded["labels"])
                removed = sum(
                    a == -100 and b != -100
                    for a, b in zip(encoded["labels"], plain["labels"])
                )
                assert removed > 0, f"No call tokens were masked: {data.get('id')}"
                # With only 0/1 loss weights, supervised tokens must be their input IDs.
                assert all(
                    label == -100 or label == token
                    for label, token in zip(encoded["labels"], encoded["input_ids"])
                )
                print(
                    f"Real sample {data.get('id')}: {len(encoded['input_ids'])} tokens, {removed} newly ignored labels"
                )
                checked += 1
                if checked >= args.masked_samples:
                    break
        assert checked, "No masked examples found"
    print("Token-label verification passed; no model weights were loaded.")


if __name__ == "__main__":
    main()
