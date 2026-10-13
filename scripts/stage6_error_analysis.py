# Stage 6: export the 50 sentences where HUM and XLIT token counts differ most, for hand labelling.
# Writes data/error_analysis_top50.csv and data/error_analysis_random50.csv, each with an empty "cause" column,
# and results/word_pair_differences.csv with the most frequent spelling differences.
import collections
import os
import sys

import pandas as pd

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))

TOP = 50
df = pd.read_csv("data/dakshina_hi_4cond.tsv", sep="\t", dtype=str, keep_default_na=False)
counts = pd.read_csv("results/per_sentence_counts.csv")
inv = pd.read_csv("results/tokenizer_inventory.csv", keep_default_na=False)
names = inv.loc[(inv["status"] == "ok") & (inv["duplicate_of"] == ""), "name"]
if "xlit" not in df.columns:
    sys.exit("No xlit column: the IndicXlit step in Stage 2 failed, so there is nothing to compare.")
assert (df["id"].astype(int) == counts["id"]).all(), "sentence order differs between the two files"


def differing_words(hum, xlit):
    hum, xlit = hum.split(), xlit.split()
    if len(hum) != len(xlit):
        return "(word counts differ)"
    return "; ".join(f"{h} -> {x}" for h, x in zip(hum, xlit) if h != x)


# Token counts are averaged over the distinct tokenizers, so no single tokenizer picks the sample.
hum = counts[[f"{t}__hum" for t in names]].mean(axis=1)
xlit = counts[[f"{t}__xlit" for t in names]].mean(axis=1)
out = df[["id", "dev", "hum", "xlit"]].copy()
out["dev_words"] = counts["dev_words"]
out["hum_tokens_mean"] = hum.round(2)
out["xlit_tokens_mean"] = xlit.round(2)
out["difference"] = (hum - xlit).round(2)
out["difference_per_word"] = ((hum - xlit) / counts["dev_words"]).round(3)
out["differing_words"] = [differing_words(h, x) for h, x in zip(df["hum"], df["xlit"])]
out["cause"] = ""

top = out.loc[out["difference"].abs().sort_values(ascending=False).index[:TOP]]
# utf-8-sig so that Excel shows Devanagari correctly
top.to_csv("data/error_analysis_top50.csv", index=False, encoding="utf-8-sig")
# The largest differences come from the longest sentences, so a random sample is exported too.
out.sample(TOP, random_state=20261012).to_csv("data/error_analysis_random50.csv", index=False, encoding="utf-8-sig")

# Corpus-wide view of the same question: which word spellings differ most often?
pairs, aligned_words = collections.Counter(), 0
for h, x in zip(df["hum"], df["xlit"]):
    h, x = h.split(), x.split()
    if len(h) == len(x):
        aligned_words += len(h)
        pairs.update((a, b) for a, b in zip(h, x) if a != b)
differing = sum(pairs.values())
common = pd.DataFrame([(a, b, n, round(100 * n / differing, 2)) for (a, b), n in pairs.most_common(TOP)],
                      columns=["hum", "xlit", "count", "percent_of_differing_words"])
common.to_csv("results/word_pair_differences.csv", index=False)
print(f"Words spelled differently by HUM and XLIT: {differing} of {aligned_words} aligned words "
      f"({100 * differing / aligned_words:.1f}%), {len(pairs)} distinct pairs")
print(f"The 10 most frequent pairs cover {common['percent_of_differing_words'].head(10).sum():.1f}% of them:")
print(common.head(10).to_string(index=False))

print(f"\nSaved data/error_analysis_top50.csv and data/error_analysis_random50.csv "
      f"({len(top)} sentences each, tokenizers averaged: {len(names)})")
print(f"HUM costs more in {(top['difference'] > 0).sum()} of them, XLIT costs more in {(top['difference'] < 0).sum()}")
print(f"Median length: {top['dev_words'].median():.0f} words in this sample, {out['dev_words'].median():.0f} in the corpus")
print("\nFirst 5 rows:")
print(top[["id", "dev_words", "difference", "differing_words"]].head(5).to_string(index=False, max_colwidth=110))
