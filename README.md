# Finance & Tax Fine-Tuning Practice

A complete, small-scale post-training loop on an open model: score it on finance and tax questions, fine-tune it, and score it again. I built this to rehearse the workflow before the [Vals AI hackathon](https://vals.ai/hackathon) (November 2026), where teams post-train small models to climb professional benchmarks in finance, tax and law.

## What it does

1. **Baseline.** Loads Qwen2.5-1.5B-Instruct and scores it on 10 held-out finance and tax questions.
2. **Fine-tune.** Trains a LoRA (low-rank adaptation) adapter on 22 worked examples. LoRA trains a small add-on instead of the whole model, so it runs in minutes on a free Colab GPU (graphics processing unit).
3. **Rescore.** Same questions, same scorer, and a comparison against the baseline.
4. **Error analysis.** Writes every wrong answer to a spreadsheet so mistakes can be sorted by type, which decides what training data to add next.

## Results from my run

| | Score on 10 held-out questions |
|---|---|
| Base model | [X]/10 |
| After fine-tuning | [X + 1]/10 |
| Change | **+10 percentage points** |

**How to read this:** one more question answered correctly. With 10 test questions and 22 training examples, the result is directionally useful but noisy, and most of the gain likely comes from learning the answer format (showing work and ending with a `Final answer:` line) rather than new finance knowledge. That is a typical first win, and it is why the next step is more targeted data, not more training.

## What's inside

| Path | Purpose |
|---|---|
| `notebooks/practice_finetune.ipynb` | The full loop, step by step, for Google Colab |
| `scoring/score.py` | Scores answers by category; accepts formats like `$19,333`, `22.47%` and `(B)`; writes an error log |
| `data/train.jsonl`, `data/eval.jsonl` | 22 training and 10 evaluation questions across tax, accounting, corporate finance, venture and fund math |
| `data/template.csv` | Spreadsheet layout for writing more training examples as a team |
| `tools/csv_to_jsonl.py` | Validates a spreadsheet of examples and converts it to training format |
| `tools/build_seed_data.py` | Regenerates the seed questions; every answer is computed in code |
| `analysis/error_log_template.csv` | Template for sorting mistakes into failure types |
| `docs/GUIDE.md` | Plain-language guide: running it, writing data, error analysis, glossary |

## Design choices

- **Answers are computed, not typed.** Each seed question is generated from a formula, so the worked steps and the final answer always agree.
- **Questions state their own rates.** "Assume a 21% federal rate" keeps answers correct when tax law changes.
- **Evaluation questions never appear in training.** Otherwise the score would measure memorization.
- **The scorer is tested.** A test confirms the reference answers score 100%, and edge cases (percent formats, negatives in parentheses, units like "million") are covered.

## Run it

**In Google Colab (no setup):**
1. Open `notebooks/practice_finetune.ipynb` in [Colab](https://colab.research.google.com) and choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells in order and upload `data/train.jsonl`, `data/eval.jsonl` and `scoring/score.py` when prompted.

**Scoring and tests locally:**
```bash
python scoring/score.py --gold data/eval.jsonl --pred predictions.jsonl --errors errors.csv
pip install pytest && python -m pytest
```

## Next steps

- Add 50–100 examples aimed at the most common failure type and rerun.
- Expand the evaluation set so a single answer no longer moves the score by 10 points.
- Try reinforcement learning on questions with checkable numeric answers.

## How it was built

I built this with AI coding agents and ran it end to end myself. All data is synthetic: the numbers are invented, and no real company or client information is used.

Built by [Valerie Van Vu](https://valerievu.com).
