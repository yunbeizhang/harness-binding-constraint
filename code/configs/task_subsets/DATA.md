# Task Subsets

EvalHarness uses tasks sampled from **SWE-bench Verified**, test split:

```text
princeton-nlp/SWE-bench_Verified
```

The full SWE-bench Verified test split contains 500 instances. `subset_100`
was sampled with stratification by the SWE-bench difficulty label and fixed
random seed `42`, so it preserves the approximate difficulty mix of the full
split.

## Sampling Design

The cleaned release keeps `subset_100.json`, the final subset used for the main
experiment. It contains 100 ordered instance IDs sampled from the SWE-bench
Verified test split.

## Difficulty Distribution

| Difficulty | Full SWE-bench Verified | subset_100 |
|---|---:|---:|
| `<15 min fix` | 194 | 39 |
| `15 min - 1 hour` | 261 | 52 |
| `1-4 hours` | 42 | 8 |
| `>4 hours` | 3 | 1 |
| Total | 500 | 100 |

## Active File

| File | Instances | Purpose |
|---|---:|---|
| `subset_100.json` | 100 | Main experiment subset for the final 3 model x 3 harness grid |

`subset_100.json` records the source dataset, split, seed, sampling method,
difficulty distribution, and ordered instance IDs.

## Repository Coverage

`subset_100` covers the following source repositories:

| Repository | Count |
|---|---:|
| `django/django` | 48 |
| `sympy/sympy` | 15 |
| `sphinx-doc/sphinx` | 12 |
| `scikit-learn/scikit-learn` | 6 |
| `astropy/astropy` | 5 |
| `matplotlib/matplotlib` | 4 |
| `pydata/xarray` | 4 |
| `pylint-dev/pylint` | 3 |
| `pytest-dev/pytest` | 2 |
| `psf/requests` | 1 |

## Reproducibility

Do not regenerate `subset_100.json` in place. If the sampling strategy changes,
create a new subset file with a new name and document the new seed, sampling
rule, and source dataset.
