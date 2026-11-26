"""Stream agent trajectories into ms-swift SFT data with auditable call masks.

No model, judge, API, or training dependencies are needed. Tool observations are
data, never instructions. Only explicit validation failures are masked by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


def schema_errors(value: Any, schema: dict, path: str = "arguments") -> list[str]:
    """Validate the simple JSON Schema constraints present in this dataset."""
    types = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    expected = schema.get("type")
    expected = (
        expected if isinstance(expected, list) else [expected] if expected else []
    )
    if expected and not any(t in types and types[t](value) for t in expected):
        return [f"{path}: expected {expected}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: outside enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(schema_errors(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        for bound, compare in [
            ("minimum", lambda a, b: a < b),
            ("maximum", lambda a, b: a > b),
        ]:
            if bound in schema and compare(value, schema[bound]):
                errors.append(f"{path}: violates {bound}")
    return errors


def tool_schemas(raw: Any) -> dict[str, dict]:
    tools = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(tools, list) or not tools:
        raise ValueError("Expected a nonempty tools list or JSON-encoded list")
    schemas = {}
    for tool in tools:
        function = tool.get("function", tool)
        schemas[function["name"]] = function.get("parameters", {})
    return schemas


def call_errors(content: str, schemas: dict[str, dict]) -> tuple[str, list[str]]:
    try:
        call = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        # Such a record also cannot be rendered by the native agent template.
        raise ValueError("Tool call content is not valid JSON") from None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError("Tool call has no string name")
    name = call["name"]
    if name not in schemas:
        return name, ["unknown_tool"]
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            raise ValueError("Tool arguments string is not valid JSON") from None
    if not isinstance(args, dict):
        raise ValueError(
            "Tool arguments must be a JSON object for Qwen3.5 XML rendering"
        )
    errors = schema_errors(args, schemas[name])
    if name in {"edgar_search", "web_search", "price_history"}:
        parsed = {}
        for key in ["start_date", "end_date"]:
            value = args.get(key)
            if value is None:
                continue
            try:
                if not isinstance(value, str) or not re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", value
                ):
                    raise ValueError
                parsed[key] = date.fromisoformat(value)
            except ValueError:
                errors.append(f"{key}: invalid_date")
        if len(parsed) == 2 and parsed["start_date"] > parsed["end_date"]:
            errors.append("start_date: after_end_date")
    return name, errors


def classify_response(name: str, content: str) -> tuple[str | None, str | None]:
    """Classify anchored tool diagnostics, not keywords inside retrieved documents.

    Returns (agent_error | execution_failure | review | None, reason).
    A failed website/API call alone does not prove that the model chose badly.
    """
    short = content.strip().lstrip("\"'")[:2500]
    lower = short.lower()
    if name == "retrieve_information":
        if lower.startswith("error: no documents are available in data storage"):
            return "agent_error", "empty_data_storage"
        if lower.startswith("error: your prompt must include at least one key"):
            return "agent_error", "missing_storage_placeholder"
        if lower.startswith("error: the key "):
            return "agent_error", "invalid_storage_key_or_reference"
        if lower.startswith(
            (
                "error: each item in input_character_ranges",
                "error: each range specification",
            )
        ):
            return "agent_error", "invalid_character_ranges"
        if lower.startswith("error code: 413") and (
            "context" in lower or "tokens" in lower
        ):
            return "agent_error", "tool_input_exceeds_context_window"
    if name == "calculator" and lower.startswith("error: invalid expression"):
        return "agent_error", "invalid_calculator_expression"
    if name in {"edgar_search", "web_search", "price_history"}:
        if re.match(r"(?:invalid |parameter )?(?:start_date|end_date)\b", lower):
            return "agent_error", "invalid_date_range"
    if lower.startswith(
        (
            "cannot connect to host",
            "cannot write to closing transport",
            "connection timeout",
            "timeout error when parsing",
            "request timed out",
        )
    ):
        return "execution_failure", "network_or_timeout"
    if re.match(r"[45]\d\d,\s*message=", short):
        code = int(short[:3])
        if code in {400, 404, 422}:
            return "review", f"http_{code}_cause_uncertain"
        return "execution_failure", f"http_{code}"
    match = re.match(r"(?:pricing data http|error code:)\s*(\d{3})", lower)
    if match:
        code = int(match.group(1))
        if "额度" in short or "订阅" in short:
            return "execution_failure", "service_quota_or_subscription"
        if code in {400, 404, 422}:
            return "review", f"http_{code}_cause_uncertain"
        return "execution_failure", f"http_{code}"
    if lower.startswith("'utf-8' codec can't decode") or lower.startswith(
        "utf-8' codec can't decode"
    ):
        return "execution_failure", "tool_decode_failure"
    if lower.startswith("html output was empty"):
        return "execution_failure", "empty_html"
    if name == "price_history" and lower.startswith("no pricing data returned"):
        return "review", "no_price_data"
    if name != "submit_final_result" and content.strip() in {"", "[]", "EMPTY"}:
        return "review", "empty_result_not_proof_of_bad_call"
    if lower.startswith(("error:", "error code:", "traceback", "exception:")):
        return "review", "unclassified_explicit_error"
    return None, None


def call_batches(messages: list[dict]) -> list[tuple[list[int], list[int]]]:
    """Pair each call batch with results in the recorded positional order."""
    batches = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] != "tool_call":
            if messages[i]["role"] in {"tool", "tool_response"}:
                raise ValueError(f"Orphan tool response at message {i}")
            i += 1
            continue
        calls = []
        while i < len(messages) and messages[i]["role"] == "tool_call":
            calls.append(i)
            i += 1
        responses = []
        while i < len(messages) and messages[i]["role"] in {"tool", "tool_response"}:
            responses.append(i)
            i += 1
        if len(calls) != len(responses):
            raise ValueError(f"Unbalanced call batch: {calls} versus {responses}")
        batches.append((calls, responses))
    return batches


def mask_record(
    record: dict, *, mask_execution_failures: bool = False
) -> tuple[dict, list[dict], Counter]:
    # Change only loss annotations; preserve message content, order and metadata.
    result = {**record, "messages": [dict(m) for m in record["messages"]]}
    messages = result["messages"]
    if not messages or any(
        m.get("role")
        not in {"system", "user", "assistant", "tool_call", "tool_response", "tool"}
        or not isinstance(m.get("content"), str)
        for m in messages
    ):
        raise ValueError("Unsupported role or non-string content")
    schemas = tool_schemas(record["tools"])
    audit = []
    counts = Counter(rows=1, messages=len(messages))
    for calls, responses in call_batches(messages):
        counts["call_batches"] += 1
        masked_in_batch = 0
        for call_index, response_index in zip(calls, responses):
            call, response = messages[call_index], messages[response_index]
            name, invalid = call_errors(call["content"], schemas)
            category, reason = classify_response(name, response["content"])
            reasons = (["schema_or_date_validation"] if invalid else []) + (
                [reason] if reason else []
            )
            should_mask = (
                bool(invalid)
                or category == "agent_error"
                or (
                    mask_execution_failures
                    and category in {"execution_failure", "review"}
                    and reason != "empty_result_not_proof_of_bad_call"
                    and reason != "no_price_data"
                )
            )
            counts["tool_calls"] += 1
            if category:
                counts[f"{category}_responses"] += 1
            if invalid:
                counts["schema_or_date_invalid_calls"] += 1
            if should_mask:
                call["loss"] = False
                # An old positive loss_scale can otherwise override binary loss.
                call["loss_scale"] = 0.0
                counts["newly_masked_tool_calls"] += 1
            if call.get("loss") is False or call.get("loss_scale") == 0:
                masked_in_batch += 1
                counts["masked_tool_calls"] += 1
                counts[f"masked_tool:{name}"] += 1
            if should_mask or category or invalid:
                audit.append(
                    {
                        "id": record.get("id"),
                        "call_index": call_index,
                        "response_index": response_index,
                        "tool": name,
                        "masked": call.get("loss") is False
                        or call.get("loss_scale") == 0,
                        "category": "agent_error" if invalid else category,
                        "reasons": reasons,
                        "validation_details": invalid,
                        "call_content": call["content"],
                        "response_excerpt": response["content"][:2500],
                    }
                )
        if 0 < masked_in_batch < len(calls):
            counts["mixed_mask_batches"] += 1
        counts["batch_max_size"] = max(counts["batch_max_size"], len(calls))
    if counts["masked_tool_calls"]:
        counts["rows_with_masked_calls"] = 1
    return result, audit, counts


def export_masks(
    input_path: Path, output_dir: Path, *, mask_execution_failures: bool = False
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.masked.swift.jsonl"
    audit_path = output_dir / "tool_call_audit.jsonl"
    counts = Counter()
    reasons = Counter()
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    with (
        input_path.open("rb") as source,
        output_path.open("xb") as output,
        audit_path.open("x", encoding="utf-8") as audit,
    ):
        for line_number, line in enumerate(source, 1):
            source_hash.update(line)
            try:
                record, entries, row_counts = mask_record(
                    json.loads(line), mask_execution_failures=mask_execution_failures
                )
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Line {line_number}: {error}; partial output must not be used"
                ) from error
            batch_max = row_counts.pop("batch_max_size", 0)
            counts["batch_max_size"] = max(counts["batch_max_size"], batch_max)
            counts.update(row_counts)
            encoded = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            output.write(encoded)
            output_hash.update(encoded)
            for entry in entries:
                entry["line_number"] = line_number
                audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if entry["masked"]:
                    reasons.update(entry["reasons"])
            if line_number % 1000 == 0:
                print(f"Processed {line_number} records", flush=True)
    summary = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "source_sha256": source_hash.hexdigest(),
        "output_sha256": output_hash.hexdigest(),
        "policy": "mask_agent_errors_and_execution_failures"
        if mask_execution_failures
        else "mask_explicit_agent_errors_only",
        "mask_fields": {"loss": False, "loss_scale": 0.0},
        "requires_external_plugin": "scripts/swift_per_call_loss.py",
        "counts": dict(counts),
        "masked_reasons": dict(reasons),
        "limitations": [
            "Successful execution does not establish semantic correctness or final-answer accuracy.",
            "Timeout/quota/HTTP failures are preserved by default; uncertain failures are in the audit.",
            "Preceding assistant reasoning is retained and supervised unless already masked.",
            "The dataset has not been tokenized locally; run scripts/check_swift_loss_mask.py in the training environment.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main_sync() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-execution-failures",
        action="store_true",
        help="Also mask explicit execution failures and uncertain HTTP errors",
    )
    args = parser.parse_args()
    try:
        summary = export_masks(
            args.input,
            args.output_dir,
            mask_execution_failures=args.mask_execution_failures,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Masking failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_sync()
