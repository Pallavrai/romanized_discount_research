# Stage 7 helper: list every [DATA] placeholder in the draft next to the value from this run.
# Reads results/summary.json and results/stage2_report.json; writes results/data_placeholders.md.
import json
import os

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))

with open("results/summary.json") as f:
    s = json.load(f)
with open("results/stage2_report.json") as f:
    stage2 = json.load(f)

medians = s["discount_hum_median_by_generation"]
per_gen = s["tokenizers_per_generation"]
xlit, itr, tau = s.get("gap_xlit_percent"), s.get("gap_itr_percent"), s.get("kendall_tau_hum_vs_xlit")
missing = "NOT AVAILABLE (IndicXlit failed in Stage 2)"


def median(gen):
    return f"{medians[gen]}x (n={per_gen[gen]})" if gen in medians else "no tokenizer in this generation"


below = s["discount_hum_below_1"]
rows = [
    ("Abstract, Section 3", "[DATA: N]", s["sentences"]),
    ("Abstract, Introduction", "[DATA: K]", s["tokenizers"]),
    ("Abstract, Introduction", "[DATA: earliest year]", s["earliest_year"]),
    ("Abstract, 5.2", "[DATA: X]% (HUM vs XLIT, mean over tokenizers)", f"{xlit['mean']:+}" if xlit else missing),
    ("Abstract", "[DATA: min]% to [DATA: max]%",
     f"{xlit['min']:+} ({xlit['min_tokenizer']}) to {xlit['max']:+} ({xlit['max_tokenizer']})" if xlit else missing),
    ("5.2", "[DATA: Y]% (HUM vs ITR, mean over tokenizers)", f"{itr['mean']:+}" if itr else missing),
    ("Abstract", "[DATA: A]x (older tokenizers, G1 median)", median("G1")),
    ("Abstract", "[DATA: B]x (recent tokenizers, G3 median)", median("G3")),
    ("5.3", "median discount G1 / G2 / G3", " / ".join(median(g) for g in ("G1", "G2", "G3"))),
    ("Abstract, 5.3", "[DATA: reverses / does not reverse], tokenizer names",
     f"discount below 1.0 for: {', '.join(below)}" if below else "does not reverse for any tokenizer"),
    ("5.3", "discount below 1.0 with the whole 95% CI below 1.0", ", ".join(s["discount_hum_ci_entirely_below_1"]) or "none"),
    ("Abstract, 5.4", "[DATA: tau] (p)",
     f"{tau['tau']} (p = {tau['p']:.3g}; 95% CI {tau['ci95'][0]} to {tau['ci95'][1]})" if tau else missing),
    ("5.4", "most efficient tokenizer under HUM / under XLIT",
     f"{s['most_efficient']['hum']} / {s['most_efficient']['xlit']}" if tau else missing),
    ("5.4", "tokenizers moving two or more places", (", ".join(s["moved_2_or_more_places"]) or "none") if tau else missing),
    ("Section 3", "sentences dropped as shorter than 5 words", stage2["dropped_under_5_words"]),
    ("Section 3", "source sentences before filtering", stage2["source_sentences"]),
    ("Table 1", "all cells", "results/table1_fertility.md"),
    ("Table 2", "all cells", "results/table2_gap.md" if xlit or itr else missing),
    ("Appendix A", "vocab size and commit per tokenizer", "results/tokenizer_inventory.csv"),
    ("Figure 1", "chart", "results/figure1_discount.pdf"),
]
lines = ["| Where in the draft | Placeholder | Value from this run |", "| --- | --- | --- |"]
lines += [f"| {where} | {tag} | {value} |" for where, tag, value in rows]
note = ("\nA plus sign means human romanization costs MORE tokens than the machine version; "
        "a minus sign means it costs FEWER, and the sentence in the draft must be reworded.\n")
text = "\n".join(lines) + "\n" + note
with open("results/data_placeholders.md", "w") as f:
    f.write(text)
print(text)
