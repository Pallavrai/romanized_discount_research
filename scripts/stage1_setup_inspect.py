# Stage 1: install dependencies, log versions, download Dakshina, print the Hindi file layout.
import collections
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from importlib.metadata import PackageNotFoundError, version

URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
TAR = "dakshina_dataset_v1.0.tar"
LANG = "hi"
PKGS = ["numpy", "pandas", "scipy", "matplotlib", "transformers", "tokenizers", "huggingface_hub",
        "tiktoken", "sentencepiece", "protobuf", "indic-transliteration"]

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))

# No version pins: pins would fight the packages Kaggle/Colab preinstall.
# The versions that actually ran are logged below and written to requirements.txt.
pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], capture_output=True, text=True)
if pip.returncode:
    sys.exit("pip install failed:\n" + (pip.stdout + pip.stderr)[-3000:])


def installed(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


info = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {p: installed(p) for p in PKGS},
}

with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD")) as r:
    size = int(r.headers["Content-Length"])
print(f"Dakshina archive: {size / 1e9:.2f} GB | free disk here: {shutil.disk_usage('.').free / 1e9:.1f} GB")
if not os.path.exists(TAR) or os.path.getsize(TAR) != size:
    urllib.request.urlretrieve(URL, TAR)

sha = hashlib.sha256()
with open(TAR, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        sha.update(chunk)
info["dakshina"] = {"url": URL, "bytes": size, "sha256": sha.hexdigest()}

with tarfile.open(TAR) as tar:
    members = tar.getmembers()
    print("\nTop of archive:", sorted({"/".join(m.name.split("/")[:2]) for m in members}))
    hindi = [m for m in members if LANG in m.name.split("/")]
    print(f"\n=== Everything under '{LANG}' (bytes, path) ===")
    for m in hindi:
        print(f"{m.size:>12,}  {m.name}{'/' if m.isdir() else ''}")
    romanized = [m for m in hindi if m.isfile() and "romanized" in m.name.split("/")]
    tar.extractall("data", members=romanized, filter="data")

print("\n=== Romanized files: line count, tab-separated fields per line, first 5 lines (cut at 200 chars) ===")
for m in romanized:
    # newline="" keeps the original line endings visible in repr()
    with open(os.path.join("data", m.name), encoding="utf-8", errors="replace", newline="") as f:
        lines = f.readlines()
    fields = collections.Counter(line.count("\t") + 1 for line in lines)
    print(f"\n--- {m.name}\nlines={len(lines)} fields_per_line={dict(fields)}")
    for line in lines[:5]:
        print(repr(line[:200]))

with open("versions.json", "w") as f:
    json.dump(info, f, indent=2)
with open("requirements.txt", "w") as f:
    f.write("".join(f"{p}=={v}\n" for p, v in info["packages"].items() if v))
print("\n=== versions.json ===\n" + json.dumps(info, indent=2))
