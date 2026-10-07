"""NLTK preprocessing used by the comparison pipeline."""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

TOKENIZER = TreebankWordTokenizer()
LEMMATIZER = WordNetLemmatizer()
NEGATIONS = {"no", "nor", "not", "never", "none", "neither", "n't"}
FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "has", "have", "he", "her", "his", "i", "if",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "you", "your",
}


@lru_cache(maxsize=1)
def selective_stopwords() -> set[str]:
    """Return English stopwords while retaining semantically useful negation."""
    try:
        words = set(stopwords.words("english"))
    except LookupError:
        words = FALLBACK_STOPWORDS
    return words - NEGATIONS


def _wordnet_ready() -> bool:
    try:
        wordnet.ensure_loaded()
        return True
    except LookupError:
        return False


def nltk_preprocess(text: str) -> str:
    """Tokenize, normalize URLs/emails, filter stopwords, and lemmatize.

    Treebank tokenization is model-backed NLTK logic without a runtime download.
    WordNet lemmatization is used when the corpus is installed; otherwise tokens
    are left unchanged so inference remains available offline.
    """
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " emailtoken ", text)
    tokens = [t for t in TOKENIZER.tokenize(text) if re.search(r"[a-z0-9]", t)]
    stops = selective_stopwords()
    tokens = [t for t in tokens if t not in stops]
    if _wordnet_ready():
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)
