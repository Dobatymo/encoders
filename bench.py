from encoder.encoder import BytesLabelEncoder, StringLabelEncoder
from genutility.time import PrintStatementTime
from itertools import repeat, chain
import numpy as np
from random import choices, shuffle
from string import ascii_lowercase

from sklearn.preprocessing import LabelEncoder

ascii_lowercase_bytes = list(map(lambda s: s.encode("ascii"), ascii_lowercase))
REPEAT = 1000

def random_string(length: int=3):
    return "".join(choices(ascii_lowercase, k=length))

def random_bytes(length: int=3):
    return b"".join(choices(ascii_lowercase_bytes, k=length))

def get_datasets(n_labels: int=100, n_multiply: int=100, string: bool=True):

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

def bench_fit_bytes(n_repeat: int=REPEAT):

    labels = get_datasets(string=False)

    le = BytesLabelEncoder()
    with PrintStatementTime(f"{le.__class__.__name__}.fit(set): {{delta:.03f}}s"):
        for i in range(n_repeat):
            le.partial_fit(set(labels))

    le = BytesLabelEncoder()
    with PrintStatementTime(f"{le.__class__.__name__}.fit: {{delta:.03f}}s"):
        for i in range(n_repeat):
            le.partial_fit(labels)

    le2 = LabelEncoder()
    with PrintStatementTime(f"{le2.__class__.__name__}.fit(set): {{delta:.03f}}s"):
        le2.fit(np.array(list(set(chain.from_iterable(repeat(labels, n_repeat))))))
    
    le2 = LabelEncoder()
    with PrintStatementTime(f"{le2.__class__.__name__}.fit: {{delta:.03f}}s"):
        le2.fit(np.array(list(chain.from_iterable(repeat(labels, n_repeat)))))

    return labels, le, le2

def bench_fit_string(n_repeat: int=REPEAT):

    labels = get_datasets(string=True)

    le = StringLabelEncoder()
    with PrintStatementTime(f"{le.__class__.__name__}.fit(set): {{delta:.03f}}s"):
        for i in range(n_repeat):
            le.partial_fit(set(labels))

    le = StringLabelEncoder()
    with PrintStatementTime(f"{le.__class__.__name__}.fit: {{delta:.03f}}s"):
        for i in range(n_repeat):
            le.partial_fit(labels)

    le2 = LabelEncoder()
    with PrintStatementTime(f"{le2.__class__.__name__}.fit(set): {{delta:.03f}}s"):
        le2.fit(np.array(list(set(chain.from_iterable(repeat(labels, n_repeat))))))

    le2 = LabelEncoder()
    with PrintStatementTime(f"{le2.__class__.__name__}.fit: {{delta:.03f}}s"):
        le2.fit(np.array(list(chain.from_iterable(repeat(labels, n_repeat)))))

    return labels, le, le2

def bench_transform(labels, le, le2, n_repeat: int=REPEAT):

    with PrintStatementTime(f"{le.__class__.__name__}.transform: {{delta:.03f}}s"):
        for i in range(n_repeat):
            out = le.transform(labels)

    with PrintStatementTime(f"{le2.__class__.__name__}.transform: {{delta:.03f}}s"):
        for i in range(n_repeat):
            out = le2.transform(labels)

if __name__ == "__main__":  
    labels, le, le2 = bench_fit_bytes()
    bench_transform(labels, le, le2)

    labels, le, le2 = bench_fit_string()
    bench_transform(labels, le, le2)
