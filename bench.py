from random import choices, shuffle
from string import ascii_lowercase

import numpy as np
import pandas as pd
from genutility.time import PrintStatementTime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from encoders.encoder import BytesLabelEncoder, StringLabelEncoder

ascii_lowercase_bytes = list(map(lambda s: s.encode("ascii"), ascii_lowercase))
REPEAT = 1000


class PandasLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y) -> None:
        self.classes_ = pd.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        return pd.Categorical(y, categories=self.classes_).codes

    def inverse_transform(self, codes) -> np.ndarray:
        return pd.Categorical.from_codes(codes, self.classes_)


def random_string(length: int = 3):
    return "".join(choices(ascii_lowercase, k=length))


def random_bytes(length: int = 3):
    return b"".join(choices(ascii_lowercase_bytes, k=length))


def get_datasets(n_labels: int = 100, n_multiply: int = 100, string: bool = True):

    if string:
        random_label = random_string
    else:
        random_label = random_bytes
    s = set()

    while len(s) < n_labels:
        s.add(random_label(3))

    labels = list(s) * n_multiply

    shuffle(labels)

    return labels


def bench_fit(classes, labels, n_repeat):
    les = []

    for cls in classes:
        le = cls()
        with PrintStatementTime(f"{le.__class__.__name__}.fit(set): {{delta:.03f}}s"):
            for i in range(n_repeat):
                le.fit(list(set(labels)))

        le = cls()
        with PrintStatementTime(f"{le.__class__.__name__}.fit: {{delta:.03f}}s"):
            for i in range(n_repeat):
                le.fit(labels)

        les.append(le)

    return les


def bench_fit_bytes(n_repeat: int = REPEAT):

    bytes_classes = [BytesLabelEncoder, LabelEncoder, PandasLabelEncoder]
    labels = get_datasets(string=False)
    les = bench_fit(bytes_classes, labels, n_repeat)
    return les, labels


def bench_fit_string(n_repeat: int = REPEAT):

    string_classes = [StringLabelEncoder, LabelEncoder, PandasLabelEncoder]
    labels = get_datasets(string=True)
    les = bench_fit(string_classes, labels, n_repeat)
    return les, labels


def bench_transform(les, labels, n_repeat: int = REPEAT):

    encodeds = []

    for le in les:
        with PrintStatementTime(f"{le.__class__.__name__}.transform: {{delta:.03f}}s"):
            for i in range(n_repeat):
                encoded = le.transform(labels)

        encodeds.append(encoded)

    return encodeds


def bench_inverse_transform(les, encodeds, n_repeat: int = REPEAT):

    for le, encoded in zip(les, encodeds):
        with PrintStatementTime(f"{le.__class__.__name__}.inverse_transform: {{delta:.03f}}s"):
            for i in range(n_repeat):
                le.inverse_transform(encoded)


if __name__ == "__main__":

    for func in [bench_fit_bytes, bench_fit_string]:

        print(func.__name__)
        les, labels = func()
        encodeds = bench_transform(les, labels)
        bench_inverse_transform(les, encodeds)
        print()
