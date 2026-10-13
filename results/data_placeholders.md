| Where in the draft | Placeholder | Value from this run |
| --- | --- | --- |
| Abstract, Section 3 | [DATA: N] | 8847 |
| Abstract, Introduction | [DATA: K] | 12 |
| Abstract, Introduction | [DATA: earliest year] | 2019 |
| Abstract, 5.2 | [DATA: X]% (HUM vs XLIT, mean over tokenizers) | -11.5 |
| Abstract | [DATA: min]% to [DATA: max]% | -13.75 (mT5) to -8.86 (Sarvam-1) |
| 5.2 | [DATA: Y]% (HUM vs ITR, mean over tokenizers) | -9.43 |
| Abstract | [DATA: A]x (older tokenizers, G1 median) | 1.833x (n=4) |
| Abstract | [DATA: B]x (recent tokenizers, G3 median) | 0.863x (n=2) |
| 5.3 | median discount G1 / G2 / G3 | 1.833x (n=4) / 1.182x (n=6) / 0.863x (n=2) |
| Abstract, 5.3 | [DATA: reverses / does not reverse], tokenizer names | discount below 1.0 for: XLM-R, Mistral Tekken, o200k_base, Sarvam-1, Gemma 3, Llama 4 |
| 5.3 | discount below 1.0 with the whole 95% CI below 1.0 | XLM-R, Mistral Tekken, o200k_base, Sarvam-1, Gemma 3, Llama 4 |
| Abstract, 5.4 | [DATA: tau] (p) | 0.939 (p = 3.22e-07; 95% CI 0.909 to 0.939) |
| 5.4 | most efficient tokenizer under HUM / under XLIT | Gemma 3 / Gemma 3 |
| 5.4 | tokenizers moving two or more places | o200k_base |
| Section 3 | sentences dropped as shorter than 5 words | 1153 |
| Section 3 | source sentences before filtering | 10000 |
| Table 1 | all cells | results/table1_fertility.md |
| Table 2 | all cells | results/table2_gap.md |
| Appendix A | vocab size and commit per tokenizer | results/tokenizer_inventory.csv |
| Figure 1 | chart | results/figure1_discount.pdf |

A plus sign means human romanization costs MORE tokens than the machine version; a minus sign means it costs FEWER, and the sentence in the draft must be reworded.
