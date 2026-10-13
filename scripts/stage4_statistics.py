# Stage 4: fertility, Discount, Gap, bootstrap CIs, Wilcoxon + Holm, Kendall's tau.
# Reads results/per_sentence_counts.csv and results/tokenizer_inventory.csv; writes tables to results/.
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))

SEED, RESAMPLES, CHUNK = 20261012, 10_000, 500
ROMAN = ["hum", "xlit", "itr"]

counts = pd.read_csv("results/per_sentence_counts.csv")
inv = pd.read_csv("results/tokenizer_inventory.csv", keep_default_na=False)
inv = inv[(inv["status"] == "ok") & (inv["duplicate_of"] == "")].sort_values(["year", "name"]).reset_index(drop=True)
names = inv["name"].tolist()
EXTRA = ["dev_raw", "hum_raw", "itr_cased", "xlit_rescored"]
have = [c for c in ["dev"] + ROMAN + EXTRA if f"{names[0]}__{c}" in counts.columns]
roman = [c for c in ROMAN if c in have]
machine = [c for c in roman if c != "hum"]

# Paired bootstrap over sentences. Multinomial weights are the same thing as drawing sentences
# with replacement, and one set of resamples is shared by every tokenizer and condition.
cols = ["dev_words"] + [f"{t}__{c}" for t in names for c in have]
X = counts[cols].to_numpy(dtype=float)
n = len(X)
rng = np.random.default_rng(SEED)
boot = np.vstack([rng.multinomial(n, np.full(n, 1 / n), size=CHUNK) @ X for _ in range(RESAMPLES // CHUNK)])
total = dict(zip(cols, X.sum(axis=0)))
resampled = dict(zip(cols, boot.T))


def ratio(numerator, denominator, minus=0.0):
    """Point estimate from corpus totals with a 95% percentile interval from the bootstrap."""
    low, high = np.percentile(resampled[numerator] / resampled[denominator] - minus, [2.5, 97.5])
    return total[numerator] / total[denominator] - minus, low, high


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p))
    out[order] = np.minimum(adjusted, 1)
    return out


assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def wilcoxon(a, b):
    return 1.0 if (a == b).all() else stats.wilcoxon(a, b).pvalue


def to_markdown(table):
    rows = [list(table.columns)] + table.astype(str).values.tolist()
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines[:1] + ["|" + " --- |" * len(table.columns)] + lines[1:]) + "\n"


def save(table, stem, title):
    table.to_csv(f"results/{stem}.csv", index=False)
    with open(f"results/{stem}.md", "w") as f:
        f.write(to_markdown(table))
    print(f"\n=== {title} ===\n{to_markdown(table)}")


meta = inv[["name", "label", "gen", "year"]]

# Table 1: fertility = tokens per Devanagari word.
fertility = meta.copy()
for c in ["dev"] + roman:
    fertility[c] = [ratio(f"{t}__{c}", "dev_words")[0] for t in names]
table1 = fertility[["label", "gen"] + ["dev"] + roman].round(3)
table1.columns = ["Tokenizer", "Gen"] + [c.upper() for c in ["dev"] + roman]
save(table1, "table1_fertility", "Table 1. Fertility (tokens per Devanagari word), lower is cheaper")

# Discount = T(DEV) / T(c). Above 1 means romanization saves tokens.
discount = meta.copy()
for c in roman:
    values = [ratio(f"{t}__dev", f"{t}__{c}") for t in names]
    discount[f"discount_{c}"], discount[f"discount_{c}_low"], discount[f"discount_{c}_high"] = zip(*values)
discount.to_csv("results/discount.csv", index=False)

# Table 2: Gap = T(HUM) / T(machine) - 1, in percent. Positive means human text costs more.
gap = meta.copy()
table2 = pd.DataFrame({"Tokenizer": inv["label"]})
for c in machine:
    values = [ratio(f"{t}__hum", f"{t}__{c}", minus=1.0) for t in names]
    gap[f"gap_{c}"], gap[f"gap_{c}_low"], gap[f"gap_{c}_high"] = (100 * np.array(v) for v in zip(*values))
    gap[f"wilcoxon_holm_{c}"] = holm([wilcoxon(counts[f"{t}__hum"], counts[f"{t}__{c}"]) for t in names])
    table2[f"Gap vs {c.upper()} (%)"] = gap[f"gap_{c}"].map("{:+.2f}".format)
    table2[f"95% CI ({c.upper()})"] = [f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in zip(gap[f"gap_{c}_low"], gap[f"gap_{c}_high"])]
    # a p-value too small for a float prints as 0, so show the float limit instead
    table2[f"Wilcoxon p, Holm ({c.upper()})"] = [f"{p:.2e}" if p > 0 else "<1e-300" for p in gap[f"wilcoxon_holm_{c}"]]
gap.to_csv("results/gap.csv", index=False)
if machine:
    save(table2, "table2_gap", "Table 2. Human-versus-machine gap with 95% bootstrap confidence intervals")

summary = {
    "sentences": n,
    "devanagari_words": int(total["dev_words"]),
    "tokenizers": len(names),
    "tokenizer_labels": inv["label"].tolist(),
    "earliest_year": int(inv["year"].min()),
    "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
    "discount_hum_median_by_generation": discount.groupby("gen")["discount_hum"].median().round(3).to_dict(),
    "tokenizers_per_generation": inv["gen"].value_counts().sort_index().to_dict(),
    "discount_hum_below_1": discount.loc[discount["discount_hum"] < 1, "label"].tolist(),
    "discount_hum_ci_entirely_below_1": discount.loc[discount["discount_hum_high"] < 1, "label"].tolist(),
}
for c in machine:
    g = gap[f"gap_{c}"]
    summary[f"gap_{c}_percent"] = {
        "mean": round(g.mean(), 2), "min": round(g.min(), 2), "max": round(g.max(), 2),
        "min_tokenizer": inv["label"][g.idxmin()], "max_tokenizer": inv["label"][g.idxmax()],
    }

# RQ3: does the ranking of tokenizers change between human and machine romanization?
if "xlit" in roman:
    ranks = meta.copy()
    ranks["rank_hum"] = fertility["hum"].rank(method="min").astype(int)
    ranks["rank_xlit"] = fertility["xlit"].rank(method="min").astype(int)
    ranks["shift"] = ranks["rank_xlit"] - ranks["rank_hum"]
    ranks.to_csv("results/rankings.csv", index=False)
    tau, p = stats.kendalltau(fertility["hum"], fertility["xlit"])
    # The same resamples give an interval for tau. Token totals rank exactly like fertility.
    hum_boot = np.column_stack([resampled[f"{t}__hum"] for t in names])
    xlit_boot = np.column_stack([resampled[f"{t}__xlit"] for t in names])
    tau_low, tau_high = np.percentile([stats.kendalltau(h, x)[0] for h, x in zip(hum_boot, xlit_boot)], [2.5, 97.5])
    summary["kendall_tau_hum_vs_xlit"] = {"tau": round(float(tau), 3), "p": float(p),
                                          "ci95": [round(float(tau_low), 3), round(float(tau_high), 3)]}
    summary["most_efficient"] = {"hum": inv["label"][fertility["hum"].idxmin()], "xlit": inv["label"][fertility["xlit"].idxmin()]}
    summary["moved_2_or_more_places"] = ranks.loc[ranks["shift"].abs() >= 2, "label"].tolist()
    print("=== Rankings by fertility (1 = cheapest) ===\n" + ranks.drop(columns="name").to_string(index=False))

# Robustness: case and punctuation kept, ITRANS before lowercasing, IndicXlit with its dictionary re-ranking.
robust = meta.copy()
robust["discount_hum"] = discount["discount_hum"]
robust["discount_hum_raw_text"] = [ratio(f"{t}__dev_raw", f"{t}__hum_raw")[0] for t in names]
robust["fertility_itr"] = fertility["itr"]
robust["fertility_itr_cased"] = [ratio(f"{t}__itr_cased", "dev_words")[0] for t in names]
if "xlit_rescored" in have:
    robust["gap_xlit_percent"] = gap["gap_xlit"]
    robust["gap_xlit_rescored_percent"] = [100 * ratio(f"{t}__hum", f"{t}__xlit_rescored", minus=1.0)[0] for t in names]
    summary["gap_xlit_rescored_percent_mean"] = round(robust["gap_xlit_rescored_percent"].mean(), 2)
robust.round(3).to_csv("results/robustness.csv", index=False)
print("\n=== Robustness ===\n" + robust.drop(columns="name").round(3).to_string(index=False))

text = json.dumps(summary, indent=2, default=lambda o: o.item())  # .item() turns numpy numbers into plain ones
with open("results/summary.json", "w") as f:
    f.write(text)
print("\n=== Discount (DEV / condition) with 95% CI ===\n" + discount.drop(columns="name").round(3).to_string(index=False))
print("\n=== summary.json ===\n" + text)
