"""Build notebooks/practice_finetune.ipynb. Run: python tools/build_notebook.py"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Practice fine-tuning run: finance and tax

This notebook walks through one complete post-training loop on a small open model, so the hackathon workflow feels familiar:

1. **Score the base model** on held-out finance and tax questions.
2. **Fine-tune** it with LoRA (low-rank adaptation), which trains a small add-on instead of the whole model.
3. **Score it again** and compare.

**Runtime:** Google Colab with a free T4 GPU (graphics processing unit). Choose *Runtime → Change runtime type → T4 GPU* before running. The whole notebook takes roughly 15–25 minutes.

**What to expect:** with only 22 training examples, this is a rehearsal, not a real improvement. The model will mostly learn the answer format and may overfit. At the hackathon you would use hundreds to thousands of examples.

**Files to upload when prompted:** `train.jsonl`, `eval.jsonl` (from the `data` folder) and `score.py` (from the `scoring` folder).
""")

md("## 1. Install the libraries\nUnsloth makes fine-tuning fast enough for a free GPU. This takes 2–4 minutes. If Colab asks you to restart the session afterwards, do so and continue from step 2.")
code("""
!pip install -q unsloth
""")

md("## 2. Upload the data and scoring script")
code("""
import json, os, sys

try:
    from google.colab import files
    print("Select train.jsonl, eval.jsonl and score.py")
    files.upload()
except ImportError:
    print("Not on Colab: make sure train.jsonl, eval.jsonl and score.py are in this folder.")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

train_rows = load_jsonl("train.jsonl")
eval_rows = load_jsonl("eval.jsonl")
sys.path.insert(0, ".")
from score import score, print_report

print(f"{len(train_rows)} training examples, {len(eval_rows)} evaluation examples")
print("Example question:", train_rows[0]["question"][:200])
""")

md("""
## 3. Load a small model
`Qwen2.5-1.5B-Instruct` is small enough for a free GPU. At the hackathon, swap in whichever model the rules allow (for example a 4B or 8B model on a larger GPU).

`load_in_4bit=True` compresses the model so it fits in memory.
""")
code("""
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
""")

md("""
## 4. Decide how questions and answers are formatted
Every example becomes a short chat: a system instruction, the question, and (for training) the worked answer ending in `Final answer: ...`. The scoring script reads that last line, so teaching the model this exact format is the first easy win.
""")
code("""
SYSTEM_PROMPT = (
    "You are a careful finance and tax analyst. Work through the problem step by step, "
    "then end with a single line in the form 'Final answer: <answer>'. "
    "Give percentages with a % sign and dollar amounts as plain numbers."
)

def build_messages(row, include_answer):
    question = row["question"]
    if row.get("context"):
        question = row["context"] + "\\n\\n" + question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if include_answer:
        answer = str(row["answer"]) + ("%" if row["answer_type"] == "percent" else "")
        messages.append({"role": "assistant", "content": row["reasoning"] + "\\nFinal answer: " + answer})
    return messages

print(tokenizer.apply_chat_template(build_messages(train_rows[0], True), tokenize=False))
""")

md("## 5. Score the base model before training\nThis is your baseline. Write the number down: every change you make later is judged against it.")
code("""
import torch

def generate_answers(rows, max_new_tokens=384):
    FastLanguageModel.for_inference(model)
    outputs = {}
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            build_messages(row, include_answer=False),
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids=prompt, max_new_tokens=max_new_tokens, do_sample=False)
        outputs[row["id"]] = tokenizer.decode(out[0][prompt.shape[1]:], skip_special_tokens=True)
    return outputs

base_outputs = generate_answers(eval_rows)
base_report = score(eval_rows, base_outputs)
print_report(base_report, "base model")
""")

md("### Read a few mistakes\nThis is the most valuable habit at the hackathon: look at *why* the model is wrong before deciding what data to add.")
code("""
for err in base_report["errors"][:3]:
    print(f"--- {err['id']} ({err['category']}) expected {err['expected']}, got {err['got']}")
    print(base_outputs[err["id"]][-600:])
    print()
""")

md("""
## 6. Attach LoRA adapters and fine-tune
- `r=16` sets the size of the add-on. Bigger can learn more but trains slower.
- `max_steps=30` keeps the practice run short. With 22 examples and 8 examples per step, the model sees each example about 11 times, which is a lot, so watch for overfitting.
- `learning_rate=2e-4` is a common starting point for LoRA.
""")
code("""
import inspect
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
FastLanguageModel.for_training(model)

train_ds = Dataset.from_list([
    {"text": tokenizer.apply_chat_template(build_messages(r, include_answer=True), tokenize=False)}
    for r in train_rows
])

config_args = dict(
    dataset_text_field="text",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=3,
    max_steps=30,
    learning_rate=2e-4,
    logging_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)
# The TRL library renamed some arguments across versions; this keeps the cell working on either.
config_params = inspect.signature(SFTConfig).parameters
config_args["max_seq_length" if "max_seq_length" in config_params else "max_length"] = MAX_SEQ_LEN
trainer_args = dict(model=model, train_dataset=train_ds, args=SFTConfig(**config_args))
if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
    trainer_args["processing_class"] = tokenizer
else:
    trainer_args["tokenizer"] = tokenizer

trainer = SFTTrainer(**trainer_args)
stats = trainer.train()
print(f"Training took {stats.metrics['train_runtime']:.0f} seconds; final loss {stats.metrics['train_loss']:.3f}")
""")

md("## 7. Score the fine-tuned model\nSame questions, same scoring. Compare against the baseline from step 5.")
code("""
tuned_outputs = generate_answers(eval_rows)
tuned_report = score(eval_rows, tuned_outputs)
print_report(base_report, "base model")
print()
print_report(tuned_report, "fine-tuned")
print()
print(f"Change: {tuned_report['accuracy'] - base_report['accuracy']:+.1%}")
""")

md("## 8. Save the error log and the adapter\nThe error log opens in Google Sheets. Fill in the `failure_type` column to see which kinds of mistakes are most common.")
code("""
from score import write_errors

write_errors(tuned_report, "errors_after_training.csv")
model.save_pretrained("finance_tax_lora")
tokenizer.save_pretrained("finance_tax_lora")

try:
    from google.colab import files
    files.download("errors_after_training.csv")
except ImportError:
    print("Saved errors_after_training.csv and the finance_tax_lora folder.")
""")

md("""
## What to try next
- **More data beats more training.** Add 50–100 examples for whichever category scored worst, then rerun steps 6–7.
- **Check for overfitting.** If the score drops after training, lower `max_steps` or the learning rate.
- **Try a bigger model.** Change `MODEL_NAME` to a larger instruct model if your GPU has room.
- **Keep eval questions out of training.** Never copy evaluation questions into the training file, or the score stops meaning anything.
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "practice_finetune.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out}")
