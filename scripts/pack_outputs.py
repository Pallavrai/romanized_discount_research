# Last step: put everything the paper and the repository need into one file, outputs.zip.
import glob
import os
import zipfile

os.chdir(next(d for d in ("/kaggle/working", "/content", ".") if os.path.isdir(d)))

FILES = ["versions.json", "requirements.txt", "data/dakshina_hi_4cond.tsv", "data/xlit_cache.json",
         "data/error_analysis_top50.csv", "data/error_analysis_random50.csv"] + sorted(glob.glob("results/*"))

with zipfile.ZipFile("outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for path in FILES:
        if os.path.exists(path):
            z.write(path)
            print(f"{os.path.getsize(path):>12,}  {path}")
        else:
            print(f"     MISSING  {path}")
print(f"\noutputs.zip: {os.path.getsize('outputs.zip') / 1e6:.1f} MB")

try:
    from google.colab import files
    files.download("outputs.zip")
except ImportError:
    print("On Kaggle: download outputs.zip from the Output panel on the right.")
