# encoders

- fast LabelEncoder for str and bytes to int

## Install

`pip install encoders`

## Dev

- `py -m pip install -e .[test,bench]`
- use `py -m pre_commit run --all-files` to run pre-commit checks manually
- `cd tests && py -m unittest discover` to run tests, `py bench.py` to run benchmarks

## benchmarks

### bytes

| class               |     fit |   fit(set) |   inverse_transform |   transform |
|:--------------------|--------:|-----------:|--------------------:|------------:|
| BytesLabelEncoder   | 4.40007 |   0.404088 |            0.656    |     2.7153  |
| PandasLabelEncoder  | 5.95293 |   0.558918 |            0.337004 |     3.31889 |
| SklearnLabelEncoder | 4.44269 |   0.448736 |            0.874752 |     7.73648 |

### string

| class               |     fit |   fit(set) |   inverse_transform |   transform |
|:--------------------|--------:|-----------:|--------------------:|------------:|
| StringLabelEncoder  | 1.91807 |   0.286374 |            0.152906 |    0.597774 |
| PandasLabelEncoder  | 2.36687 |   0.416667 |            0.318656 |    3.34337  |
| SklearnLabelEncoder | 5.9505  |   0.380508 |            0.939498 |   19.2761   |

## Plans

- fast Counter, MultiCounter for str and bytes
