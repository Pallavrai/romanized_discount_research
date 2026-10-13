# Builds pipeline.ipynb from the stage scripts, so the notebook and the scripts cannot drift apart.
# Run after editing any script: python scripts/make_notebook.py
import json
import pathlib

HERE = pathlib.Path(__file__).parent
STAGES = [
    ("stage1_setup_inspect.py", "Stage 1: install, download Dakshina, print the Hindi files"),
    ("stage2_build_conditions.py", "Stage 2: build the four conditions (slow cell: IndicXlit)"),
    ("stage3_count_tokens.py", "Stage 3: count tokens with every tokenizer"),
    ("stage4_statistics.py", "Stage 4: statistics and Tables 1 and 2"),
    ("stage5_figure.py", "Stage 5: Figure 1"),
    ("stage6_error_analysis.py", "Stage 6: sentences for hand labelling"),
    ("stage7_placeholders.py", "Stage 7: values for the [DATA] placeholders"),
    ("pack_outputs.py", "Last step: outputs.zip"),
]
INTRO = """# Token cost of human-romanized Hindi

Run the cells from top to bottom on Kaggle or Colab (CPU is enough).

Before you start:
- **Kaggle:** switch on Internet in the notebook settings.
- **Gated tokenizers (Llama 2, Llama 3, Llama 4, Gemma 3):** accept each license on Hugging Face, then save a
  free read token as a notebook secret named `HF_TOKEN`. Without it those four are skipped and logged.

Everything is written to `results/` and `data/`. The last cell packs them into `outputs.zip`.
"""


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


cells = [cell("markdown", INTRO)]
for filename, title in STAGES:
    cells += [cell("markdown", "## " + title), cell("code", (HERE / filename).read_text(encoding="utf-8"))]
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = HERE.parent / "pipeline.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {out} with {len(STAGES)} code cells")
