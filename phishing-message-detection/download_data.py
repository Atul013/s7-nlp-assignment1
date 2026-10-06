"""Download the pinned, MIT-licensed educational phishing/benign corpus."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/Teddyha/phishing_benign_email_dataset/resolve/ce65d43/phishing%20and%20benign%20email%20dataset.jsonl"
SHA256 = "fbc5158d5308c376012a427c7a11d9a2616c6f9b569364d0e5bb085c62b0ff16"
OUTPUT = Path("data/raw/phishing_benign.jsonl")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Dataset checksum mismatch: expected {SHA256}, got {digest}")
    OUTPUT.write_bytes(payload)
    print(f"Saved {OUTPUT} ({len(payload)} bytes, sha256={digest})")


if __name__ == "__main__":
    main()
