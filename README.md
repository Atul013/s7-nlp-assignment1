# S7 NLP — Assignment 1: Text Preprocessing

A complete, end-to-end text preprocessing pipeline applied to a self-introduction paragraph.

**Atul** · B.Tech CSE (Artificial Intelligence), Semester 7
Adi Shankara Institute of Engineering & Technology

---

## What this is

The assignment asks for a self-introduction paragraph covering *Name, Place, Schooling, Family,
Current status, Achievements, Hobbies, Interests, Strengths and Weaknesses*, and for a full text
preprocessing pipeline applied to it.

The raw paragraph here is written deliberately as **noisy, real-world text** — HTML tags, HTML
entities, a URL, an email address, emoji, contractions, digits, irregular casing and inconsistent
whitespace. A clean paragraph would make most preprocessing steps invisible; noisy text makes the
effect of every stage measurable.

## Pipeline

| # | Stage | Purpose |
|---|-------|---------|
| 1 | Noise removal (HTML, URL, email, emoji) | Strip non-linguistic markup |
| 2 | Contraction expansion | `I'm` → `I am` |
| 3 | Case folding | Collapse `Python`/`python` to one token |
| 4 | Number handling | Remove digits / normalise |
| 5 | Punctuation removal | Drop non-semantic symbols |
| 6 | Whitespace normalisation | Single-space, trimmed |
| 7 | Sentence tokenisation | Split into sentences |
| 8 | Word tokenisation | Split into tokens |
| 9 | Stopword removal (negation preserved) | Drop low-information words |
| 10 | Stemming — Porter / Snowball / Lancaster | Crude suffix stripping, compared |
| 11 | POS tagging | Grammatical category per token |
| 12 | POS-aware lemmatisation | Dictionary-correct base forms |
| 13 | Named Entity Recognition | Extract people, places, organisations |
| 14 | Frequency distribution & N-grams | Corpus statistics |
| 15 | Bag of Words & TF-IDF | Numeric feature representation |

## Results

| Stage | Tokens | Unique tokens |
|-------|--------|---------------|
| Raw text | 291 | 202 |
| Tokenised | 280 | 182 |
| Stopwords removed | 158 | 143 |
| Lemmatised (POS-aware) | 158 | 140 |

**45.7% token reduction**, vocabulary reduced from 202 to 140 unique terms.

## Notes worth reading

1. **Ordering is not arbitrary.** Contractions must be expanded before punctuation is stripped, or
   `didn't` becomes the junk token `didnt`. URLs must be removed before punctuation, or one link
   explodes into a dozen fragments. NER must run before lowercasing, because capitalisation is one
   of the strongest entity features — so the pipeline branches a case-preserved copy of the text.
2. **Naive lemmatisation is a silent failure.** `WordNetLemmatizer` defaults to `pos='n'`, so verbs
   pass through unchanged (`studying` stays `studying`) and nothing raises an error. Mapping Penn
   Treebank tags to WordNet tags first is what makes it work — the notebook shows both side by side.
3. **Default stopword lists need editing.** NLTK's list removes `not`, `no` and `nor`, which inverts
   meaning. They are explicitly retained here.
4. **Stemmers are compared, not assumed.** Lancaster over-stems (`atul` → `at`, `science` → `sci`);
   Porter is conservative but still produces non-words. Lemmatisation wins when output must stay
   human-readable.

## Run it

### Google Colab
Open `NLP_Assignment1_Text_Preprocessing.ipynb` in Colab, uncomment the `!pip install` line in the
setup cell, and run all cells.

### Locally
```bash
pip install -r requirements.txt
jupyter notebook NLP_Assignment1_Text_Preprocessing.ipynb
```

NLTK corpora (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`, `maxent_ne_chunker`,
`words`) download automatically in the setup cell.

## Files

| File | Description |
|------|-------------|
| `NLP_Assignment1_Text_Preprocessing.ipynb` | The executed notebook, all outputs saved |
| `NLP_Assignment1_Report.docx` | Submission report — code, outputs, repo link |
| `NLP_Assignment1_Report.pdf` | PDF version of the same report |
| `requirements.txt` | Pinned dependencies |
