# Vals Hackathon, Track 1: Starter Kit

A rehearsal kit for the post-training track. It lets you run one full loop before the event (score a small model, fine-tune it, score it again) and gives you a clean way to build finance and tax training data with a team.

## What's inside

| Folder | What it is | When you use it |
|---|---|---|
| `notebooks/practice_finetune.ipynb` | A Google Colab notebook that scores a small model, fine-tunes it with LoRA (low-rank adaptation) and scores it again | Once before the event, to learn the workflow |
| `scoring/score.py` | Scores model answers against correct answers, by category, and writes an error log | Every time you test a model |
| `data/template.csv` | The spreadsheet layout for writing training examples | When your team writes examples in Google Sheets |
| `data/train.jsonl`, `data/eval.jsonl` | 22 training and 10 evaluation examples across tax, accounting, corporate finance, venture and fund math | Practice data for the notebook |
| `tools/csv_to_jsonl.py` | Checks a spreadsheet of examples and converts it to the training format | After writing examples in a spreadsheet |
| `tools/build_seed_data.py` | Regenerates the seed examples; every answer is computed in code | If you want to add or change seed examples |
| `analysis/error_log_template.csv` | A sheet for sorting the model's mistakes into types | After each scoring run |

## Step 1: Do one practice run (about 30 minutes)

1. Go to [colab.research.google.com](https://colab.research.google.com) and choose **File → Upload notebook**. Pick `notebooks/practice_finetune.ipynb`.
2. Choose **Runtime → Change runtime type → T4 GPU**, then **Save**.
3. Run the cells from top to bottom (**Shift + Enter**). When asked, upload `data/train.jsonl`, `data/eval.jsonl` and `scoring/score.py`.
4. Compare the two scores at step 7. With only 22 examples, expect small or noisy changes. The point is to see every step work once.

If a cell fails on installation, restart the session (**Runtime → Restart session**) and run from step 2. Library versions change often; the notebook already handles the most common renamed setting.

## Step 2: Learn the scoring loop

The model must end each answer with a line like `Final answer: 22.47%`. The scorer reads that line and checks it:

- **Numbers** count as correct within a small tolerance, so `$19,333` matches `19332.80`.
- **Percentages** accept `22.47%`, `22.47` or `0.2247`.
- **Multiple choice** accepts `B` or `(B) Investing activities`.

To score any model's answers from the command line:

```bash
python scoring/score.py --gold data/eval.jsonl --pred predictions.jsonl --errors analysis/errors.csv
```

`predictions.jsonl` needs one line per question: `{"id": "EV-001", "output": "...full model response..."}`.

The scorer also reports how many responses were missing the `Final answer:` line. That is a formatting problem, and it is the cheapest one to fix.

## Step 3: Build training data as a team

Write examples in Google Sheets using the columns in `data/template.csv`:

| Column | What to put |
|---|---|
| `id` | A unique code, such as `TR-101` |
| `split` | `train` or `eval`. Never put the same question in both |
| `category` | `tax`, `accounting`, `corporate_finance`, `venture`, `fund`, `credit` or your own |
| `question` | The full question, with every rate and rule the answer depends on |
| `context` | Optional: a document excerpt the question refers to |
| `reasoning` | The worked solution, step by step, as you would want the model to write it |
| `answer` | Just the answer: a number, a percent value (`22.47`) or a letter |
| `answer_type` | `number`, `percent` or `choice` |
| `tolerance` | How close counts as correct; `0.001` means within 0.1% |
| `source`, `license` | Where it came from, and whether you are allowed to use it |
| `reviewed_by` | Initials of the person who checked the answer |

Then export as CSV and convert:

```bash
python tools/csv_to_jsonl.py my_examples.csv --out data/train.jsonl --split train
```

The converter stops and lists every problem row (missing answer, duplicate id, a non-number where a number belongs) before writing anything.

### Rules that keep training data useful

1. **State every rate inside the question.** "Assume a 21% federal rate" keeps the answer correct even when tax law changes.
2. **Write the reasoning you want copied.** The model learns your steps as well as your answers.
3. **Have a second person check every answer.** One wrong example teaches the model a wrong rule.
4. **Track the source.** Only use material you are allowed to use. Check the hackathon rules on which models may generate training data, because some providers' terms restrict training other models on their outputs.
5. **Keep evaluation questions private.** If they leak into training, your score stops measuring anything.

## Step 4: Error analysis, the skill that wins

After each scoring run, open the error log in Google Sheets and fill in `failure_type` for each mistake, for example:

- `format`: no final answer line, or wrong units
- `arithmetic`: right method, wrong calculation
- `wrong_rule`: misapplied a tax or accounting rule
- `misread`: ignored part of the question
- `used_discount_price_not_cap`: specific to the domain

Count the types. Add training examples that target the most common one, retrain and rescore. Repeat.

## Words you'll hear at the hackathon

- **Base model:** the model before you train it.
- **Fine-tuning / SFT (supervised fine-tuning):** training on example questions with worked answers.
- **LoRA (low-rank adaptation):** training a small add-on to the model instead of all of it; fast and cheap.
- **Epoch:** one pass through all the training data.
- **Overfitting:** the model memorizes the training examples and gets worse on new questions.
- **RL (reinforcement learning) / GRPO (group relative policy optimization):** letting the model try many answers and rewarding correct ones; useful when answers can be checked automatically.
- **Loss:** a training number that should go down; it does not tell you the real score, so always rescore.

## Run the tests

```bash
pip install pytest
python -m pytest
```

All seed answers are computed in code, and a test confirms the worked answers score 100%.
