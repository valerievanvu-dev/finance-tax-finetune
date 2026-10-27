import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))

from score import final_answer_text, is_correct, parse_number, score  # noqa: E402


@pytest.mark.parametrize("text, value, pct", [
    ("$617,760.00", 617760.0, False),
    ("22.47%", 22.47, True),
    ("2.2x", 2.2, False),
    ("$10 million", 10_000_000, False),
    ("(1,250)", -1250, False),
    ("about 6,000 units", 6000, False),
])
def test_parse_number(text, value, pct):
    v, p = parse_number(text)
    assert v == pytest.approx(value) and p == pct


def test_final_answer_line_is_preferred():
    text, found = final_answer_text("Step 1: 5\nStep 2: 7\nFinal answer: $12.00")
    assert found and text == "$12.00"


def test_fallback_when_line_missing():
    text, found = final_answer_text("The answer is 12")
    assert not found and text == "The answer is 12"


def test_percent_accepts_three_forms():
    gold = {"answer": 22.47, "answer_type": "percent", "tolerance": 0.002}
    assert is_correct(gold, "22.47%")[0]
    assert is_correct(gold, "22.47")[0]
    assert is_correct(gold, "0.2247")[0]
    assert not is_correct(gold, "25%")[0]


def test_number_tolerance():
    gold = {"answer": 19332.80, "answer_type": "number", "tolerance": 0.001}
    assert is_correct(gold, "$19,333")[0]
    assert not is_correct(gold, "$19,500")[0]


def test_choice():
    gold = {"answer": "B", "answer_type": "choice"}
    assert is_correct(gold, "(B) Investing activities")[0]
    assert not is_correct(gold, "A")[0]


def test_seed_reasoning_scores_perfectly():
    """The worked answers in the seed data must score 100% against themselves."""
    rows = [json.loads(l) for f in ("train.jsonl", "eval.jsonl") for l in open(ROOT / "data" / f)]
    preds = {}
    for r in rows:
        ans = f"{r['answer']}%" if r["answer_type"] == "percent" else str(r["answer"])
        preds[r["id"]] = r["reasoning"] + "\nFinal answer: " + ans
    report = score(rows, preds)
    assert report["accuracy"] == 1.0, report["errors"]


def test_cli(tmp_path):
    gold = ROOT / "data" / "eval.jsonl"
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({"id": "EV-004", "output": "Final answer: 6.25%"}) + "\n")
    out = subprocess.run([sys.executable, str(ROOT / "scoring" / "score.py"), "--gold", str(gold), "--pred", str(pred)],
                         capture_output=True, text=True, check=True).stdout
    assert "Overall: 1/10" in out
