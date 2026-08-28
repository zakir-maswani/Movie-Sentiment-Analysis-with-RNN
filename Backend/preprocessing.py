import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(
            f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
        )
    except LookupError:
        nltk.download(pkg, quiet=True)

_english_stopwords = set(stopwords.words("english"))
_porter_stemmer = PorterStemmer()


def _remove_urls(text: str) -> str:
    return re.sub(r"http\S+", "", text)


def _remove_punctuation(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


def _remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def _remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in _english_stopwords)


def _stem_text(text: str) -> str:
    tokens = word_tokenize(text)
    return " ".join(_porter_stemmer.stem(t) for t in tokens)


def preprocess_review(text: str) -> str:
    """Runs the exact same cleaning pipeline used in the training notebook,
    in the exact same order: lowercase -> strip URLs -> strip punctuation
    -> strip HTML -> remove stopwords -> stem."""
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_punctuation(text)
    text = _remove_html_tags(text)
    text = _remove_stopwords(text)
    text = _stem_text(text)
    return text
