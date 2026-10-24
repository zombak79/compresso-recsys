# Models from other libraries

compresso-recsys evaluates any object with
`predict_on_batch(source, *, k[, exclude_seen])` that returns the top-k items of
each row as an `SRPTensor`. A model trained elsewhere therefore needs only a
small adapter. It then runs through the same `evaluate_recommender` and
`compare_models` as the built-in models, on the same users, with paired
statistics. These scripts are examples: neither library is a dependency of
compresso-recsys.

| Script | Library | Model | Split | Adapter |
|---|---|---|---|---|
| `implicit_als.py` | [implicit](https://github.com/benfred/implicit) | ALS | ML-1M `user_split` (held-out users) | `ImplicitALS`, about 10 lines |
| `warprec_sasrec.py` | [WarpRec](https://pypi.org/project/warprec/) (Python ≥ 3.12) | SASRec | ML-1M `leave_last_out` | `WarpRecSequential`, about 15 lines |

## Matrix and sequence models

- **A matrix model** (implicit) receives a `csr_matrix` of histories.
- **A sequential model** (WarpRec) receives an `ItemSequences` batch, in the order the items were consumed.
- **Both are scored against the same targets.** The evaluator's pairing fingerprint compares histories as item sets, so the two can be compared directly.

## The one trap

Some libraries index users by their *training* row, and score the stored factors of those users. Under strong generalisation, the evaluator's rows are held-out users that are not in training.

- implicit's `recommend()` needs `recalculate_user=True` to fold each history in. Without it, ALS silently scores the wrong users and falls below popularity.
- Neither library can detect this, because targets and histories are both correctly paired.
- Check the library's documentation for how it treats unseen users before trusting the numbers.

## Reference runs

- **`implicit_als.py`** (implicit 0.7.3, CPU):
  - EASE nDCG@100 = 0.4303.
  - ALS (128 factors) 0.3830, Δ −0.0473 [−0.0529, −0.0417].
  - Pop 0.2300.
- **`warprec_sasrec.py --epochs 1`** (WarpRec 2.2.8, CPU, about 5 minutes) is a check that the adapter works, not a tuned result:
  - EASE nDCG@20 = 0.0712.
  - SASRec 0.0577.
  - Pop 0.0258.
