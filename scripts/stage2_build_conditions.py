# Stage 2: build the four conditions (DEV, HUM, XLIT, ITR) for the same Hindi sentences.
# Writes data/dakshina_hi_4cond.tsv and results/stage2_report.json.
import glob
import json
import os
import shutil
import subprocess
import sys
import unicodedata

import pandas as pd
from indic_transliteration import sanscript

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))
os.makedirs("results", exist_ok=True)

ROM = "data/dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined"
MIN_WORDS = 5
# IndicXlit needs fairseq, which only ships ready-made builds up to Python 3.8,
# so it runs in its own environment and the notebook's packages are left alone.
ENV = os.environ.get("XLIT_ENV", "/tmp/xlit_env")
PY = f"{ENV}/bin/python"


def read_aligned(path):
    """One (native, latin) word pair per line; a '</s>' pair ends each sentence."""
    sentences, current = [], []
    with open(path, encoding="utf-8", newline="\n") as f:
        for n, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                sys.exit(f"{path} line {n}: expected 2 tab-separated fields, got {len(fields)}")
            if fields[0] == "</s>":
                sentences.append(current)
                current = []
            else:
                current.append(fields)
    if current:
        sys.exit(f"{path}: last sentence has no closing </s>")
    return sentences


# Main text: Dakshina's own punctuation-free, lowercased, word-aligned file.
# Raw text: the cased, punctuated sentences, kept as a robustness check.
nopunct = read_aligned(ROM + ".aligned.cased_nopunct.tsv")
cased = read_aligned(ROM + ".aligned.tsv")
with open(ROM + ".tsv", encoding="utf-8", newline="\n") as f:
    raw = [line.rstrip("\n").split("\t") for line in f]

# The three files must hold the same sentences in the same order.
assert all(len(r) == 2 for r in raw), "rejoined.tsv: expected 2 fields on every line"
assert len(nopunct) == len(cased) == len(raw), "the three files have different sentence counts"
assert all(len(a) == len(b) for a, b in zip(nopunct, cased)), "aligned files disagree on word counts"
assert all(
    " ".join(w[0] for w in s) == r[0] and " ".join(w[1] for w in s) == r[1] for s, r in zip(cased, raw)
), "aligned.tsv does not join back to rejoined.tsv"


def nfc(text):
    return unicodedata.normalize("NFC", text)


df = pd.DataFrame({
    "id": range(1, len(raw) + 1),
    "dev": [" ".join(w[0] for w in s) for s in nopunct],
    "hum": [" ".join(w[1] for w in s) for s in nopunct],
    "dev_raw": [r[0] for r in raw],
    "hum_raw": [r[1] for r in raw],
})
report = {"source_sentences": len(df)}
for col in ["dev", "hum", "dev_raw", "hum_raw"]:
    fixed = df[col].map(nfc)
    report[f"changed_by_nfc_{col}"] = int((fixed != df[col]).sum())
    df[col] = fixed
lowered = df["hum"].str.lower()
report["changed_by_lowercasing_hum"] = int((lowered != df["hum"]).sum())
df["hum"] = lowered

dev_words = df["dev"].str.split().str.len()
report["dropped_under_%d_words" % MIN_WORDS] = int((dev_words < MIN_WORDS).sum())
df = df[dev_words >= MIN_WORDS].reset_index(drop=True)
dev_words = df["dev"].str.split().str.len()
report["remaining_sentences"] = len(df)
report["remaining_dev_words"] = int(dev_words.sum())
report["counted_but_not_filtered"] = {
    "duplicate_dev_sentences": int(df["dev"].duplicated().sum()),
    "dev_with_latin_letters": int(df["dev"].str.contains("[A-Za-z]").sum()),
    "dev_with_digits": int(df["dev"].str.contains(r"\d").sum()),
    "dev_with_zero_width_joiners": int(df["dev"].str.contains("[‌‍]").sum()),
    "hum_word_count_differs_from_dev": int((df["hum"].str.split().str.len() != dev_words).sum()),
}

df["itr_cased"] = [sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS) for s in df["dev"]]
df["itr"] = df["itr_cased"].str.lower()

# Runs inside the Python 3.8 environment. Transliterates each distinct word once, in batches.
WORKER = r'''
import json, logging, sys, time
from importlib.metadata import version
import fairseq, numpy, torch
from ai4bharat.transliteration import XlitEngine
from ai4bharat.transliteration.transformer import indic2en
from ai4bharat.transliteration.transformer.base_engine import LANG_WORD_REGEXES
from ai4bharat.transliteration.utils import INDIC_TO_LATIN_PUNCT_TRANSLATOR, INDIC_TO_STANDARD_NUMERALS_TRANSLATOR

logging.disable(logging.INFO)  # fairseq logs several lines per batch
src, dst, cache_path = sys.argv[1:4]
BEAM, BATCH, PARITY_N = 4, 256, 20
# column -> (rescore, candidates per word).
# "xlit" is the model's own best guess, the setting in the IndicXlit README example for Indic-to-Roman.
# "xlit_rescored" is the library default: the top 4 guesses re-ranked with a word-frequency dictionary.
VARIANTS = {"xlit": (False, 1), "xlit_rescored": (True, 4)}
WORD = LANG_WORD_REGEXES["hi"]


def prep(s):
    # the same steps IndicXlit applies inside its own sentence function
    return s.lower().strip().translate(INDIC_TO_LATIN_PUNCT_TRANSLATOR).translate(INDIC_TO_STANDARD_NUMERALS_TRANSLATOR)


with open(src, encoding="utf-8") as f:
    sentences = json.load(f)
prepped = [prep(s) for s in sentences]
words = sorted({w for s in prepped for w in WORD.findall(s)})
try:
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}

start = time.time()
columns, variants = {}, {}
for column, (rescore, topk) in VARIANTS.items():
    done = cache.setdefault(column, {})
    todo = [w for w in words if w not in done]
    print("IndicXlit %s: %d distinct words, %d still to do" % (column, len(words), len(todo)), flush=True)
    engine = XlitEngine(src_script_type="indic", beam_width=BEAM, rescore=rescore)
    began = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        best = engine.batch_transliterate_words(chunk, "hi", "en", topk=topk)[0]
        if len(best) == topk * len(chunk):
            best = best[::topk]
        else:
            # batch output is not `topk` per word: use the public one-word call instead
            best = [engine.translit_word(w, "hi", topk=topk)[0] for w in chunk]
        done.update(zip(chunk, best))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        n = i + len(chunk)
        if n == len(todo) or (i // BATCH) % 10 == 9:
            print("  %d/%d words, %.0f words/sec" % (n, len(todo), n / (time.time() - began)), flush=True)
    columns[column] = [WORD.sub(lambda m: done[m.group(0)], p) for p in prepped]
    # Self-check: IndicXlit's own sentence function (one model call per word) must give the same text.
    checked = min(PARITY_N, len(sentences))
    own = [engine.translit_sentence(s, "hi") for s in sentences[:checked]]
    mismatches = [i for i in range(checked) if own[i] != columns[column][i]]
    for i in mismatches:
        print("PARITY MISMATCH in %s\n  own  : %s\n  batch: %s" % (column, own[i], columns[column][i]))
    variants[column] = {
        "rescore": rescore, "candidates_per_word": topk,
        "empty_outputs": sum(1 for w in words if not done[w]),
        "parity_checked_sentences": checked, "parity_mismatches": len(mismatches),
    }

meta = {
    "beam_width": BEAM,
    "model_url": indic2en.MODEL_DOWNLOAD_URL, "model_version": indic2en.XLIT_VERSION,
    "distinct_words": len(words),
    "variants": variants,
    "seconds": round(time.time() - start),
    "versions": {
        "python": sys.version.split()[0], "torch": torch.__version__, "fairseq": fairseq.__version__,
        "numpy": numpy.__version__, "ai4bharat-transliteration": version("ai4bharat-transliteration"),
    },
}
with open(dst, "w", encoding="utf-8") as f:
    json.dump({"sentences": sentences, "columns": columns, "meta": meta}, f, ensure_ascii=False)
print("IndicXlit done:", json.dumps(meta))
'''

# IndicXlit imports urduhack (which pulls in TensorFlow) but only calls it for Urdu.
STUB = (
    "import pathlib, site;"
    "pathlib.Path(site.getsitepackages()[0], 'urduhack.py')"
    ".write_text('def normalize(text):\\n    return text\\n')"
)
SETUP = [
    [sys.executable, "-m", "pip", "install", "-q", "uv"],
    [sys.executable, "-m", "uv", "venv", "--seed", "--python", "3.8", ENV],
    # pip >= 24.1 rejects omegaconf 2.0.x, which fairseq requires
    [PY, "-m", "pip", "install", "-q", "pip<24.1"],
    [PY, "-m", "pip", "install", "-q", "torch==2.4.1", "torchaudio==2.4.1",
     "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    [PY, "-m", "pip", "install", "-q", "numpy<1.24", "fairseq==0.12.2", "pydload", "ujson", "tqdm",
     "indic-nlp-library"],
    [PY, "-m", "pip", "install", "-q", "--no-deps", "ai4bharat-transliteration==1.1.3"],
    [PY, "-c", STUB],
]


def run(cmd):
    """Run a command and stream its output into the notebook cell. Returns the exit code."""
    print("$", " ".join(cmd).splitlines()[0][:110], flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def xlit_env_ready():
    check = [PY, "-c", "import ai4bharat.transliteration, fairseq, urduhack"]
    return os.path.exists(PY) and subprocess.run(check, capture_output=True).returncode == 0


def make_xlit(sentences):
    if not xlit_env_ready():
        shutil.rmtree(ENV, ignore_errors=True)
        if any(run(cmd) for cmd in SETUP):
            return None
    with open("data/xlit_input.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False)
    with open("data/xlit_worker.py", "w", encoding="utf-8") as f:
        f.write(WORKER)
    if run([PY, "data/xlit_worker.py", "data/xlit_input.json", "data/xlit_output.json", "data/xlit_cache.json"]):
        return None
    with open("data/xlit_output.json", encoding="utf-8") as f:
        return json.load(f)


def saved_xlit(sentences):
    """Fallback when IndicXlit cannot run here: reuse output saved by an earlier run on the same sentences."""
    for path in ["data/xlit_output.json"] + glob.glob("/kaggle/input/*/xlit_output.json"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("sentences") == sentences:
                saved["meta"]["source"] = f"reused {path}; IndicXlit did not run in this session"
                print(f"\nUsing saved IndicXlit output from {path}")
                return saved
    return None


columns = ["id", "dev", "hum", "xlit", "itr", "dev_raw", "hum_raw", "itr_cased", "xlit_rescored"]
sentences = df["dev"].tolist()
result = make_xlit(sentences) or saved_xlit(sentences)
if result is None:
    print("\n!!! IndicXlit FAILED and no saved output matches these sentences.")
    print("The file is written WITHOUT the xlit columns. Paste the error above.")
    print("Fallback: put the repository's data/xlit_output.json into data/ here (on Kaggle, add it as a dataset) and rerun this cell.")
    report["xlit"] = "FAILED"
    columns = [c for c in columns if not c.startswith("xlit")]
else:
    for column, sentences in result["columns"].items():
        df[column] = [nfc(s).lower() for s in sentences]
    report["xlit"] = result["meta"]

df[columns].to_csv("data/dakshina_hi_4cond.tsv", sep="\t", index=False)
with open("results/stage2_report.json", "w") as f:
    json.dump(report, f, indent=2)
if result is not None:
    info = {}
    if os.path.exists("versions.json"):
        with open("versions.json") as f:
            info = json.load(f)
    info["indicxlit"] = result["meta"]
    with open("versions.json", "w") as f:
        json.dump(info, f, indent=2)

print("\n=== Stage 2 report ===\n" + json.dumps(report, indent=2))
print("\n=== First 3 rows ===")
for _, row in df[columns].head(3).iterrows():
    for col in columns:
        print(f"{col:>13}: {row[col]}")
    print()
