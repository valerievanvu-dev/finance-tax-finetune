"""Convert a training-data spreadsheet (exported as CSV) into JSONL, checking every row.

Write examples in Google Sheets or Excel using the columns in data/template.csv,
export as CSV, then run:

    python tools/csv_to_jsonl.py my_examples.csv --out data/train.jsonl

The script refuses to write the file if any row has a problem, and tells you
which row and why. Typical problems: a missing answer, an answer that is not a
number when answer_type is "number", or a duplicate id.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

REQUIRED = ["id", "split", "category", "question", "reasoning", "answer", "answer_type"]
TYPES = {"number", "percent", "choice"}
SPLITS = {"train", "eval"}


def clean_number(value: str) -> float:
    return float(value.replace("$", "").replace(",", "").replace("%", "").strip())


def check_row(i: int, row: dict, seen: set[str]) -> tuple[dict | None, list[str]]:
    problems = []
    for col in REQUIRED:
        if not (row.get(col) or "").strip():
            problems.append(f"missing '{col}'")
    rid = (row.get("id") or "").strip()
    if rid in seen:
        problems.append(f"duplicate id '{rid}'")
    kind = (row.get("answer_type") or "").strip().lower()
    if kind and kind not in TYPES:
        problems.append(f"answer_type must be one of {sorted(TYPES)}")
    split = (row.get("split") or "").strip().lower()
    if split and split not in SPLITS:
        problems.append("split must be 'train' or 'eval'")
    answer = (row.get("answer") or "").strip()
    if kind in {"number", "percent"} and answer:
        try:
            answer = clean_number(answer)
        except ValueError:
            problems.append(f"answer '{answer}' is not a number")
    if kind == "choice" and answer and answer.upper() not in {"A", "B", "C", "D"}:
        problems.append("choice answers must be A, B, C or D")
    if rid:
        seen.add(rid)
    if problems:
        return None, [f"row {i}: {p}" for p in problems]
    tol = (row.get("tolerance") or "").strip()
    return {
        "id": rid,
        "split": split,
        "category": row["category"].strip(),
        "question": row["question"].strip(),
        "context": (row.get("context") or "").strip(),
        "reasoning": row["reasoning"].strip(),
        "answer": answer.upper() if kind == "choice" else answer,
        "answer_type": kind,
        "tolerance": float(tol) if tol else (0.0 if kind == "choice" else 0.001),
        "difficulty": (row.get("difficulty") or "").strip(),
        "source": (row.get("source") or "").strip(),
        "license": (row.get("license") or "").strip(),
        "reviewed_by": (row.get("reviewed_by") or "").strip(),
    }, []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=sorted(SPLITS), help="Only keep rows from this split")
    args = ap.parse_args()

    rows, problems, seen = [], [], set()
    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):  # row 1 is the header
            if not any((v or "").strip() for v in row.values()):
                continue
            rec, errs = check_row(i, row, seen)
            problems += errs
            if rec and (not args.split or rec["split"] == args.split):
                rows.append(rec)

    if problems:
        print("Fix these before converting:")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    unreviewed = sum(1 for r in rows if not r["reviewed_by"])
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {args.out}")
    if unreviewed:
        print(f"Note: {unreviewed} examples have no reviewer yet.")


if __name__ == "__main__":
    main()
