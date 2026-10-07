# Dataset provenance

- **Name:** Phishing and Benign Email Dataset
- **Publisher used:** Teddyha on Hugging Face (duplicate of darkknight25)
- **Source:** https://huggingface.co/datasets/Teddyha/phishing_benign_email_dataset
- **Pinned revision:** `ce65d43`
- **File:** `phishing and benign email dataset.jsonl`
- **SHA-256:** `fbc5158d5308c376012a427c7a11d9a2616c6f9b569364d0e5bb085c62b0ff16`
- **License stated on dataset card:** MIT
- **Size used:** 200 synthetic/curated English emails: 100 phishing and 100 benign

Fields used by this project are `id`, `subject`, `body`, and `label`. Extra
annotation fields are not used as predictive features. The code keeps the
publisher's class meanings and does not reinterpret spam/ham as phishing.

The raw file is fetched by `download_data.py`, checksum-verified, and ignored by
Git. Review the upstream dataset card and license before redistribution.
