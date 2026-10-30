"""Loading the SMS Spam Collection dataset.

The dataset (Almeida & Hidalgo, UCI Machine Learning Repository) has 5,574
real text messages, each labelled "ham" (normal) or "spam". The first time
you run the project it is downloaded and saved to data/SMSSpamCollection.
"""

import io
import os
import urllib.request
import zipfile

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Same data as a plain tab-separated file, used if the UCI site is unavailable.
MIRROR_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")


def extract_from_zip(zip_bytes, out_path):
    """Pull the SMSSpamCollection file out of the downloaded zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if os.path.basename(name) == "SMSSpamCollection":
                with open(out_path, "wb") as f:
                    f.write(zf.read(name))
                return out_path
    raise ValueError("SMSSpamCollection not found inside the zip file")


def download_dataset(path=DATA_FILE):
    """Download the dataset once and keep a local copy."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading the SMS Spam Collection dataset (one time only)...")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=30) as response:
            zip_bytes = response.read()
        return extract_from_zip(zip_bytes, path)
    except Exception as err:  # UCI site down? Use a plain-text copy instead.
        print(f"  UCI download failed ({err}); trying a mirror...")
        with urllib.request.urlopen(MIRROR_URL, timeout=30) as response:
            text_bytes = response.read()
        with open(path, "wb") as f:
            f.write(text_bytes)
        return path


def load_messages(path=None):
    """Return two lists: messages and labels (1 = spam, 0 = ham).

    Each line of the file looks like:  ham<TAB>Ok lar... Joking wif u oni...
    """
    if path is None:
        path = download_dataset()
    texts, labels = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue  # skip blank or broken lines
            label, text = line.split("\t", 1)
            if label not in ("ham", "spam"):
                continue
            texts.append(text)
            labels.append(1 if label == "spam" else 0)
    return texts, labels
