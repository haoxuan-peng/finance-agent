#!/usr/bin/env python3
"""Extract queries from a JSON/JSONL dataset into a one-question-per-line TXT file."""

import argparse
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


def extract_queries(records: Iterable[Any], field: str) -> Iterable[str]:
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条记录不是 JSON 对象")

        query = record.get(field)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"第 {index} 条记录缺少非空字符串字段 {field!r}")

        # Collapse embedded newlines/tabs so every query occupies exactly one line.
        yield " ".join(query.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 JSON/JSONL 题目集中提取 query，每题输出一行。"
    )
    parser.add_argument("input", type=Path, help="输入的 .json 或 .jsonl 文件")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="输出 TXT 文件（默认：输入文件同目录、同名 .txt）",
    )
    parser.add_argument("--field", default="query", help="题目字段名（默认：query）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".txt")

    if input_path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("输入文件扩展名必须是 .json 或 .jsonl")

    records = iter_jsonl(input_path) if input_path.suffix.lower() == ".jsonl" else iter_json(input_path)
    queries = list(extract_queries(records, args.field))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for query in queries:
            file.write(query + "\n")

    print(f"已提取 {len(queries)} 道题目 -> {output_path}")


if __name__ == "__main__":
    main()
