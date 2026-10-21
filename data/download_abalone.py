"""Download the UCI Abalone dataset (the paper's Table 3 data) and log what was fetched."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
# column names from abalone.names (the .data file has no header)
COLUMNS = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight",
           "Viscera_weight", "Shell_weight", "Rings"]

here = Path(__file__).resolve().parent / "abalone"
here.mkdir(exist_ok=True)
raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).read("abalone.data")
(here / "abalone.data").write_bytes(raw)

# keep values as the original strings so that equality is exact (no float re-formatting)
df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS, dtype=str, keep_default_na=False)
df.to_csv(here / "abalone.csv", index=False)

manifest = {"source": URL, "file": "abalone.data", "rows": len(df), "cols": df.shape[1],
            "sha256": hashlib.sha256(raw).hexdigest()}
(here / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"abalone.csv {len(df)} x {df.shape[1]} -> {here}")
