# Token cost of human-romanized Hindi across tokenizer generations

Code for the paper *Is Romanization Still a Discount? Measuring the Token Cost of Human-Romanized Hindi
Across Tokenizer Generations* by Pallav Rai and
Anand Kumar Rai.

Preprint: [doi.org/10.5281/zenodo.23385568](https://doi.org/10.5281/zenodo.23385568)

```bibtex
@misc{rai2026romanization,
  title     = {Is Romanization Still a Discount? Measuring the Token Cost of Human-Romanized {H}indi Across Tokenizer Generations},
  author    = {Rai, Pallav and Rai, Anand Kumar},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint},
  doi       = {10.5281/zenodo.23385568},
  url       = {https://doi.org/10.5281/zenodo.23385568}
}
```

The pipeline counts how many tokens the same Hindi sentences need in four forms, with open tokenizers
released between 2019 and 2026. No model weights are loaded and no text is generated.

| Code | Form | Source |
| --- | --- | --- |
| DEV | Devanagari | Dakshina, native-script side |
| HUM | Human romanization | Dakshina annotators |
| XLIT | Machine romanization | IndicXlit (Indic-to-English model) |
| ITR | Rule-based romanization | ITRANS, `indic-transliteration` package |

## Run it

**Kaggle or Colab (CPU is enough).** Upload `pipeline.ipynb` and run the cells from top to bottom.

- Kaggle: switch on Internet in the notebook settings.
- Llama 2, Llama 3, Llama 4 and Gemma 3 are gated. Accept each license on Hugging Face and save a free
  read token as a notebook secret named `HF_TOKEN`. Without it those four are skipped and logged.
- The last cell writes `outputs.zip` with everything below.

**On your own machine (tested with Python 3.12).**

```bash
pip install -r requirements.txt
```

```bash
for s in stage1_setup_inspect stage2_build_conditions stage3_count_tokens stage4_statistics stage5_figure stage6_error_analysis stage7_placeholders; do python scripts/$s.py || break; done
```

Stage 1 downloads the 2.0 GB Dakshina archive. Stage 2 is the slow step: it builds a separate
Python 3.8 environment for IndicXlit (about 1 GB) and transliterates every distinct word once.

`pipeline.ipynb` is generated from the scripts. After editing a script, run `python scripts/make_notebook.py`.

## What comes out

| File | Content |
| --- | --- |
| `data/dakshina_hi_4cond.tsv` | One row per sentence: `id, dev, hum, xlit, itr` plus four robustness columns |
| `results/stage2_report.json` | Sentences left after each filtering step, and the IndicXlit settings |
| `results/per_sentence_counts.csv` | Token count for every sentence, condition and tokenizer |
| `results/tokenizer_inventory.csv` | Appendix A: repo ID, commit hash, vocabulary size, year, generation |
| `results/table1_fertility.csv`, `.md` | Table 1: tokens per Devanagari word |
| `results/table2_gap.csv`, `.md` | Table 2: human-versus-machine gap, 95% CI, Wilcoxon p (Holm) |
| `results/discount.csv` | Discount (DEV / condition) with 95% CI: the data behind Figure 1 |
| `results/rankings.csv` | Tokenizer ranks under HUM and under XLIT |
| `results/robustness.csv` | The same measures with other text and transliteration settings |
| `results/word_pair_differences.csv` | The 50 most frequent word spellings that differ between HUM and XLIT |
| `results/summary.json` | Headline numbers, including Kendall's tau |
| `results/figure1_discount.pdf`, `.png` | Figure 1 |
| `results/data_placeholders.md` | Each `[DATA]` placeholder of the draft next to its value |
| `data/error_analysis_top50.csv`, `data/error_analysis_random50.csv` | Sentences for hand labelling, with an empty `cause` column |
| `versions.json`, `requirements.txt` | Python, library, tokenizer and IndicXlit versions of the run |

## Choices fixed before any result was seen

- **Text.** Dakshina's own lowercased, punctuation-free, word-aligned file
  (`hi.romanized.rejoined.aligned.cased_nopunct.tsv`), so that case and punctuation do not differ between
  conditions. The cased, punctuated sentences are kept as a robustness check.
- **Filter.** Sentences with fewer than 5 Devanagari words are dropped. Nothing else is filtered; duplicates
  and sentences with Latin letters, digits or zero-width joiners are counted in `stage2_report.json`.
- **XLIT.** IndicXlit v1.0, beam width 4, the model's best guess without dictionary re-ranking
  (`rescore=False`). The library default with re-ranking is kept as the robustness column `xlit_rescored`.
- **ITR.** ITRANS output, lowercased like the other romanized text. The cased output is the robustness
  column `itr_cased`.
- **Token counts.** No special tokens such as BOS or EOS are added.
- **Fertility.** Corpus totals divided by the Devanagari word count of the same sentences.
- **Generations.** By the year a vocabulary first shipped, not the year of the model: G1 before 2023,
  G2 2023 to 2024, G3 2025 and later.
- **Duplicates.** Tokenizers with identical per-sentence counts on all four conditions count as one.
- **Statistics.** Paired bootstrap over sentences, 10,000 resamples, seed 20261012, percentile intervals.
  Wilcoxon signed-rank tests with Holm correction across tokenizers. Kendall's tau between the fertility
  rankings under HUM and XLIT.

## Reproducibility notes

- IndicXlit runs in its own Python 3.8 environment: `torch 2.4.1`, `fairseq 0.12.2`, `numpy < 1.24`,
  `ai4bharat-transliteration 1.1.3`. The package imports `urduhack` but only calls it for Urdu, so a
  one-function stand-in replaces it and TensorFlow is not installed.
- Each distinct word is transliterated once, in batches. The script then checks 20 sentences against
  IndicXlit's own sentence function and reports any mismatch.
- If IndicXlit cannot be installed, Stage 2 reuses `data/xlit_output.json` from an earlier run, provided it
  was made from exactly the same sentences. `stage2_report.json` records when this happens.
- Mistral's Tekken tokenizer is read from the repository's `tekken.json` through `tiktoken`, the way
  `mistral-common` reads it.
- Tokenizers are loaded with `trust_remote_code=False` and pinned to the commit recorded in
  `results/tokenizer_inventory.csv`.

## Data and licences

- Code: MIT License, see `LICENSE`.
- Dakshina (Roark et al., 2020) is released under CC BY-SA 4.0. This repository redistributes only data derived
  from it: token counts, and a 50-row table of frequent word spellings. The files in `results/` are shared
  under CC BY-SA 4.0 with attribution to Dakshina. The `data/` folder is not committed.
- IndicXlit models are released under the MIT License.
