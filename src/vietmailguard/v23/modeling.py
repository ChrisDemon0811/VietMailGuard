"""Out-of-core Word/Character TF-IDF for the isolated V2.3 Development study.

Hashed term buckets avoid fitting a huge vocabulary/matrix in RAM. IDF really
learns from each admitted Train document. This is representation learning, not
supervised labeling of the unlabeled documents.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from vietmailguard.modeling import EmailTextComposer


def bounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["subject", "body"]].fillna("").astype(str).copy()
    return result.apply(lambda values: values.str.slice(0, 4096))


class StreamingHashTfidf(BaseEstimator, TransformerMixin):
    def __init__(self, dimensions: int = 65536):
        self.dimensions = dimensions

    def _counts(self, texts):
        if not hasattr(self, "vectorizer_"):
            self.vectorizer_ = FeatureUnion([
                ("word", HashingVectorizer(n_features=self.dimensions, ngram_range=(1, 2),
                    alternate_sign=False, norm=None, dtype=np.float32)),
                ("character", HashingVectorizer(n_features=self.dimensions, analyzer="char_wb",
                    ngram_range=(3, 5), alternate_sign=False, norm=None, dtype=np.float32))])
        return self.vectorizer_.transform(texts).tocsr()

    def partial_fit(self, texts, y=None):
        matrix = self._counts(texts)
        if not hasattr(self, "document_frequency_"):
            self.document_frequency_ = np.zeros(2*self.dimensions, dtype=np.int64)
            self.documents_seen_ = 0
        self.document_frequency_ += np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.int64)
        self.documents_seen_ += matrix.shape[0]
        self.idf_ = (np.log((1+self.documents_seen_) / (1+self.document_frequency_)) + 1).astype(np.float32)
        return self

    def fit(self, texts, y=None):
        for attr in ("document_frequency_", "documents_seen_", "idf_"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self.partial_fit(texts)

    def transform(self, texts):
        if not hasattr(self, "idf_"):
            raise ValueError("TF-IDF has not been fitted on Train")
        matrix = self._counts(texts)
        np.log(matrix.data, out=matrix.data)
        matrix.data += 1
        matrix = matrix.multiply(self.idf_).tocsr()
        return normalize(matrix, norm="l2", copy=False)


@dataclass
class HashedTextClassifier:
    representation: StreamingHashTfidf
    classifier: object
    projection: object = None

    def predict(self, frame):
        texts = EmailTextComposer().transform(bounded(frame))
        matrix = self.representation.transform(texts)
        if self.projection is not None:
            matrix = self.projection.transform(matrix)
        return self.classifier.predict(matrix)
