"""Score model answers against a gold answer file, and write an error log.

Usage:
    python scoring/score.py --gold data/eval.jsonl --pred predictions.jsonl --errors analysis/errors.csv

Inputs
  gold  JSONL with: id, category, answer, answer_type ("number", "percent" or "choice"), tolerance
  pred  JSONL with: id, output (the model's full text response)

The model is expected to end its response with a line like "Final answer: 22.47%".
If that line is missing, the script falls back to the last number or letter it can find,
and counts how often that happened, because a missing answer line is a formatting problem
you can fix with training data.

It can also be imported (the fine-tuning notebook does this):
    from score import score
    report = score(gold_rows, {"EV-001": "...model text..."})
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict

FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?P<open>\()?\s*(?P<neg>-)?\s*\$?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<close>\))?"
    r"\s*(?P<unit>%|x\b|million\b|mn\b|billion\b|bn\b|thousand\b|k\b|m\b|b\b)?",
    re.IGNORECASE,
)
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}


def final_answer_text(output: str) -> tuple[str, bool]:
    """Return the text after the last 'Final answer:' and whether that line was found."""
    matches = FINAL_RE.findall(output or "")
    if matches:
        return matches[-1].strip(), True
    lines = [ln.strip() for ln in (output or "").strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


def parse_number(text: str) -> tuple[float | None, bool]:
    """Parse the first number in text. Returns (value, had_percent_sign).

    Handles $, commas, (parentheses) negatives, % and x suffixes, and words like 'million'.
    """
    m = NUMBER_RE.search(text or "")
    if not m:
        return None, False
    value = float(m.group("num").replace(",", ""))
    unit = (m.group("unit") or "").lower()
    if unit in SCALE:
        value *= SCALE[unit]
    negative = bool(m.group("neg")) or bool(m.group("open") and m.group("close"))
    return (-value if negative else value), unit == "%"


def parse_choice(text: str) -> str | None:
    m = re.search(r"\(?\b([A-D])\b\)?", text or "")
    return m.group(1).upper() if m else None


def is_correct(gold: dict, answer_text: str) -> tuple[bool, str]:
    kind = gold.get("answer_type", "number")
    expected = gold["answer"]
    if kind == "choice":
        got = parse_choice(answer_text)
        return got == str(expected).upper(), str(got)

    value, had_pct = parse_number(answer_text)
    if value is None:
        return False, "no number found"
    expected = float(expected)
    tol = float(gold.get("tolerance") or 0.001)

    def close(v: float) -> bool:
        if expected == 0:
            return abs(v) <= tol
        return abs(v - expected) / abs(expected) <= tol

    if kind == "percent":
        # Gold stores 22.47 for 22.47%. Accept "22.47%", "22.47" or "0.2247".
        candidates = [value] if had_pct else [value, value * 100]
        return any(close(c) for c in candidates), f"{value}{'%' if had_pct else ''}"
    return close(value), f"{value:g}"


def score(gold_rows: list[dict], predictions: dict[str, str]) -> dict:
    by_cat = defaultdict(lambda: [0, 0])
    errors, missing_final, missing_pred = [], 0, 0
    correct = 0
    for g in gold_rows:
        output = predictions.get(g["id"])
        if output is None:
            missing_pred += 1
            ok, got, found = False, "no prediction", False
        else:
            text, found = final_answer_text(output)
            ok, got = is_correct(g, text)
        if output is not None and not found:
            missing_final += 1
        by_cat[g.get("category", "all")][1] += 1
        if ok:
            correct += 1
            by_cat[g.get("category", "all")][0] += 1
        else:
            errors.append({
                "id": g["id"],
                "category": g.get("category", ""),
                "expected": g["answer"],
                "got": got,
                "had_final_answer_line": found,
                "model_output_excerpt": (output or "")[-400:].replace("\n", " "),
                "failure_type": "",
                "notes": "",
            })
    total = len(gold_rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_category": {c: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for c, v in sorted(by_cat.items())},
        "missing_final_answer_line": missing_final,
        "missing_predictions": missing_pred,
        "errors": errors,
    }


def print_report(report: dict, label: str = "") -> None:
    title = f"Score{(' — ' + label) if label else ''}"
    print(title)
    print("-" * len(title))
    print(f"Overall: {report['correct']}/{report['total']} = {report['accuracy']:.1%}")
    for cat, v in report["by_category"].items():
        print(f"  {cat:<20} {v['correct']}/{v['total']} = {v['accuracy']:.0%}")
    if report["missing_final_answer_line"]:
        print(f"Responses without a 'Final answer:' line: {report['missing_final_answer_line']}")
    if report["missing_predictions"]:
        print(f"Questions with no prediction: {report['missing_predictions']}")


def write_errors(report: dict, path: str) -> None:
    fields = ["id", "category", "expected", "got", "had_final_answer_line", "model_output_excerpt", "failure_type", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report["errors"])


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--errors", help="Optional CSV path for the error log")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    gold = load_jsonl(args.gold)
    preds = {p["id"]: p["output"] for p in load_jsonl(args.pred)}
    report = score(gold, preds)
    print_report(report, args.label)
    if args.errors:
        write_errors(report, args.errors)
        print(f"Wrote {len(report['errors'])} errors to {args.errors}")


if __name__ == "__main__":
    main()
