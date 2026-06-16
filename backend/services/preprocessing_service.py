import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_stemming(text):
    text = normalize_text(text)
    words = text.split()

    tokens = []

    for word in words:
        if word not in stop_words and len(word) > 2:
            tokens.append(stemmer.stem(word))

    return " ".join(tokens)


def process_query(query):
    processed_query = preprocess_stemming(query)
    query_tokens = processed_query.split()

    return {
        "original_query": query,
        "processed_query": processed_query,
        "tokens": query_tokens
    }