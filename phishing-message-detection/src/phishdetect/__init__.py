"""Leakage-aware phishing-message text classification."""

from .inference import predict_message
from .preprocessing import nltk_preprocess

__all__ = ["nltk_preprocess", "predict_message"]
