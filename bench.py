from functools import reduce
from random import choices, shuffle
from string import ascii_lowercase
from typing import List, Union

import numpy as np
import pandas as pd
from genutility.time import MeasureTime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from encoders.encoder import BytesLabelEncoder, StringLabelEncoder

ascii_lowercase_bytes = list(map(lambda s: s.encode("ascii"), ascii_lowercase))
REPEAT = 1000


class SklearnLabelEncoder(LabelEncoder):
    pass  # just to change the name for the output


class PandasLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y) -> None:
        self.classes_ = pd.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        return pd.Categorical(y, categories=self.classes_).codes

    def inverse_transform(self, codes) -> np.ndarray:
        return pd.Categorical.from_codes(codes, self.classes_)


def random_string(length: int = 3):
    return "".join(choices(ascii_lowercase, k=length))  # nosec B311


def random_bytes(length: int = 3):
    return b"".join(choices(ascii_lowercase_bytes, k=length))  # nosec B311


def get_datasets(n_labels: int = 100, n_multiply: int = 100, string: bool = True) -> List[Union[str, bytes]]:
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
    out = []

    for cls in classes:
        le = cls()
        with MeasureTime() as delta:
            for _i in range(n_repeat):
                le.fit(list(set(labels)))
        out.append((le.__class__.__name__, "fit(set)", delta.get()))

        le = cls()
        with MeasureTime() as delta:
            for _i in range(n_repeat):
                le.fit(labels)
        out.append((le.__class__.__name__, "fit", delta.get()))

        les.append(le)

    return les, out


def bench_fit_bytes(n_repeat: int = REPEAT):
    bytes_classes = [BytesLabelEncoder, SklearnLabelEncoder, PandasLabelEncoder]
    labels = get_datasets(string=False)
    les, out = bench_fit(bytes_classes, labels, n_repeat)
    return les, labels, out


def bench_fit_string(n_repeat: int = REPEAT):
    string_classes = [StringLabelEncoder, SklearnLabelEncoder, PandasLabelEncoder]
    labels = get_datasets(string=True)
    les, out = bench_fit(string_classes, labels, n_repeat)
    return les, labels, out


def bench_transform(les, labels, n_repeat: int = REPEAT):
    encodeds = []
    out = []

    for le in les:
        with MeasureTime() as delta:
            for _i in range(n_repeat):
                encoded = le.transform(labels)
        out.append((le.__class__.__name__, "transform", delta.get()))
        encodeds.append(encoded)

    return encodeds, out


def bench_inverse_transform(les, encodeds, n_repeat: int = REPEAT):
    out = []

    for le, encoded in zip(les, encodeds):
        with MeasureTime() as delta:
            for _i in range(n_repeat):
                le.inverse_transform(encoded)
        out.append((le.__class__.__name__, "inverse_transform", delta.get()))

    return out


def pd_min(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return reduce(np.minimum, dfs)


if __name__ == "__main__":
    alltables = {}

    for func in [bench_fit_bytes, bench_fit_string]:
        alltables[func.__name__] = []

    for _i in tqdm(range(5)):
        for func in [bench_fit_bytes, bench_fit_string]:
            table = []
            les, labels, results = func()
            table.extend(results)
            encodeds, results = bench_transform(les, labels)
            table.extend(results)
            results = bench_inverse_transform(les, encodeds)
            table.extend(results)

            df = pd.DataFrame.from_records(table, columns=["class", "method", "seconds"]).pivot(
                columns="method", index="class", values="seconds"
            )
            alltables[func.__name__].append(df)

    for name, dfs in alltables.items():
        df_min = pd_min(dfs)

        print("#", name)
        print()
        print(df_min.to_markdown(index=True))
        print()
