# Stage 3: count tokens for every sentence, condition and tokenizer. No special tokens are added.
# Writes results/per_sentence_counts.csv and results/tokenizer_inventory.csv.
import base64
import json
import os
from importlib.metadata import version

import pandas as pd
import tiktoken
import transformers
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))
os.makedirs("results", exist_ok=True)
transformers.logging.set_verbosity_error()  # silences "sequence longer than model max length"

# name, label, loader, ID, year the vocabulary first shipped (not the year of the model).
# cl100k_base shipped in tiktoken 0.1.1 (December 2022); Qwen3 reuses the Qwen1.5 vocabulary
# file (January 2024); Tekken shipped with Mistral Nemo (July 2024).
TOKENIZERS = [
    ("gpt2", "GPT-2", "hf", "openai-community/gpt2", 2019),
    ("xlmr", "XLM-R", "hf", "FacebookAI/xlm-roberta-base", 2019),
    ("mt5", "mT5", "hf", "google/mt5-base", 2020),
    ("cl100k_base", "cl100k_base", "tiktoken", "cl100k_base", 2022),
    ("llama2", "Llama 2", "hf", "meta-llama/Llama-2-7b-hf", 2023),
    ("qwen3", "Qwen3", "hf", "Qwen/Qwen3-0.6B", 2024),
    ("llama3", "Llama 3", "hf", "meta-llama/Meta-Llama-3-8B", 2024),
    ("o200k_base", "o200k_base", "tiktoken", "o200k_base", 2024),
    ("mistral_tekken", "Mistral Tekken", "tekken", "mistralai/Mistral-Nemo-Instruct-2407", 2024),
    ("sarvam1", "Sarvam-1", "hf", "sarvamai/sarvam-1", 2024),
    ("gemma3", "Gemma 3", "hf", "google/gemma-3-1b-it", 2025),
    ("llama4", "Llama 4", "hf", "meta-llama/Llama-4-Scout-17B-16E-Instruct", 2025),
    ("sarvam_30b", "Sarvam-30B", "hf", "sarvamai/sarvam-30b", 2026),
]
MAIN = ["dev", "hum", "xlit", "itr"]


def generation(year):
    return "G1" if year < 2023 else "G2" if year < 2025 else "G3"


def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def load(loader, ident, token):
    """Returns (function mapping a list of texts to token counts, details for the inventory)."""
    if loader == "tiktoken":
        enc = tiktoken.get_encoding(ident)
        # disallowed_special=(): text that looks like a special token is counted as plain text
        return (lambda texts: [len(x) for x in enc.encode_batch(texts, disallowed_special=())],
                {"commit": "tiktoken " + version("tiktoken"), "vocab_size": enc.n_vocab})
    commit = HfApi().model_info(ident, token=token).sha
    if loader == "tekken":
        # Mistral's own vocabulary file, read the way mistral-common reads it. The converted
        # tokenizer.json in the same repo is known to split text with a wrong regex.
        with open(hf_hub_download(ident, "tekken.json", revision=commit, token=token), encoding="utf-8") as f:
            tekken = json.load(f)
        config = tekken["config"]
        inner = config["default_vocab_size"] - config["default_num_special_tokens"]
        ranks = {base64.b64decode(v["token_bytes"]): v["rank"] for v in tekken["vocab"][:inner]}
        enc = tiktoken.Encoding(name="tekken", pat_str=config["pattern"], mergeable_ranks=ranks, special_tokens={})
        return (lambda texts: [len(x) for x in enc.encode_batch(texts, disallowed_special=())],
                {"commit": commit, "vocab_size": config["default_vocab_size"]})
    # trust_remote_code=False: never run code shipped in a model repo, and never stop to ask
    tok = AutoTokenizer.from_pretrained(ident, revision=commit, token=token, trust_remote_code=False)
    return (lambda texts: [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]],
            {"commit": commit, "vocab_size": len(tok)})


df = pd.read_csv("data/dakshina_hi_4cond.tsv", sep="\t", dtype=str, keep_default_na=False)
conditions = [c for c in df.columns if c != "id"]
counts = {"id": df["id"].astype(int), "dev_words": df["dev"].str.split().str.len()}
token = hf_token()
print(f"{len(df)} sentences | conditions: {conditions} | HF token found: {bool(token)}\n")

inventory, seen = [], {}
for name, label, loader, ident, year in TOKENIZERS:
    row = {"name": name, "label": label, "gen": generation(year), "year": year, "loader": loader,
           "id": ident, "commit": "", "vocab_size": "", "status": "ok", "duplicate_of": "", "error": ""}
    try:
        count, details = load(loader, ident, token)
        new = {f"{name}__{c}": count(df[c].tolist()) for c in conditions}
        row.update(details)
        counts.update(new)
        # Declared before any data was seen: tokenizers with identical per-sentence counts
        # on every main condition count as one tokenizer.
        key = tuple(n for c in MAIN if c in conditions for n in new[f"{name}__{c}"])
        row["duplicate_of"] = seen.get(key, "")
        seen.setdefault(key, name)
        print(f"OK    {label:15} vocab={row['vocab_size']}  commit={str(row['commit'])[:15]}"
              + (f"  DUPLICATE of {row['duplicate_of']}" if row["duplicate_of"] else ""))
    except Exception as e:
        row.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}".replace("\n", " "))
        print(f"FAIL  {label:15} {row['error'][:160]}")
    inventory.append(row)

pd.DataFrame(counts).to_csv("results/per_sentence_counts.csv", index=False)
inventory = pd.DataFrame(inventory)
inventory.to_csv("results/tokenizer_inventory.csv", index=False)

info = {}
if os.path.exists("versions.json"):
    with open("versions.json") as f:
        info = json.load(f)
info["tokenizers"] = inventory.to_dict("records")
with open("versions.json", "w") as f:
    json.dump(info, f, indent=2)

print("\n=== Tokenizer inventory (Appendix A) ===")
print(inventory[["label", "gen", "year", "id", "vocab_size", "commit", "status", "duplicate_of"]].to_string(index=False))
print(f"\nSaved results/per_sentence_counts.csv: {len(df)} rows, {len(counts) - 2} count columns")
