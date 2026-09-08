#!/usr/bin/env python3
"""Extract questions from a JSON, JSONL, or CSV dataset into a TXT file."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行不是有效 JSON: {error}") from error


def iter_json(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, list):
        yield from data
        return

    if isinstance(data, dict):
        for container_key in ("data", "items", "questions"):
            if isinstance(data.get(container_key), list):
                yield from data[container_key]
                return
        yield data
        return

    raise ValueError("JSON 顶层必须是对象或数组")


def iter_csv(path: Path) -> Iterable[Any]:
    # newline="" is required so the csv module can correctly parse quoted,
    # multi-line questions. utf-8-sig also transparently removes a UTF-8 BOM.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV 文件缺少表头")
        yield from reader


def find_field(record: dict[str, Any], requested_field: str | None) -> str:
    """Find an explicit field or auto-detect query/question case-insensitively."""
    fields_by_lowercase = {
        key.lower(): key for key in record if isinstance(key, str)
    }

    if requested_field:
        if requested_field in record:
            return requested_field
        matched_field = fields_by_lowercase.get(requested_field.lower())
        if matched_field:
            return matched_field
        raise ValueError(f"找不到字段 {requested_field!r}")

    for candidate in ("query", "question"):
        matched_field = fields_by_lowercase.get(candidate)
        if matched_field:
            return matched_field

    raise ValueError("找不到题目字段，请使用 --field 指定字段名")


def extract_queries(
    records: Iterable[Any], requested_field: str | None
) -> Iterable[str]:
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条记录不是键值对象")

        try:
            field = find_field(record, requested_field)
        except ValueError as error:
            raise ValueError(f"第 {index} 条记录：{error}") from error

        query = record.get(field)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"第 {index} 条记录缺少非空字符串字段 {field!r}")

        # Collapse embedded newlines/tabs so every query occupies exactly one line.
        yield " ".join(query.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 JSON、JSONL 或 CSV 题目集中提取题目，每题输出一行。"
    )
    parser.add_argument("input", type=Path, help="输入的 .json、.jsonl 或 .csv 文件")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="输出 TXT 文件（默认：输入文件同目录、同名 .txt）",
    )
    parser.add_argument(
        "--field",
        help="题目字段名（默认自动识别 query 或 question，不区分大小写）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".txt")

    suffix = input_path.suffix.lower()
    readers = {
        ".json": iter_json,
        ".jsonl": iter_jsonl,
        ".csv": iter_csv,
    }
    if suffix not in readers:
        raise ValueError("输入文件扩展名必须是 .json、.jsonl 或 .csv")

    records = readers[suffix](input_path)
    queries = list(extract_queries(records, args.field))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for query in queries:
            file.write(query + "\n")

    print(f"已提取 {len(queries)} 道题目 -> {output_path}")


if __name__ == "__main__":
    main()
