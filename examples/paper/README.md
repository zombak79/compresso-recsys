# Paper benchmark runner

`run.py` runs the benchmark of the compresso-recsys ECIR 2027 resource paper
from checkpoints built with `build_recsys_checkpoint`. `ecir2027.toml` holds
the full configuration. `smoke.toml` is a minutes-long end-to-end pass on
ML-1M that exercises every command and writes every output file; its numbers
are not results.

```bash
export CR_CKPT=/path/to/checkpoints RESULTS=/path/to/results
python examples/paper/run.py encode  --config examples/paper/ecir2027.toml --block Cp --dataset steam-temporal --out $RESULTS   # once per C′ dataset
python examples/paper/run.py list    --config examples/paper/ecir2027.toml --stage grid [--device gpu|cpu] > grid.txt
# run every line of grid.txt (any order, any host; each line is one fit)
python examples/paper/run.py select  --config examples/paper/ecir2027.toml --block B --dataset ml1m-llo --model EASE --out $RESULTS
python examples/paper/run.py list    --config examples/paper/ecir2027.toml --stage seeds --out $RESULTS > seeds.txt
# run every line of seeds.txt
python examples/paper/run.py compare --config examples/paper/ecir2027.toml --out $RESULTS
```

## Protocol

- **One `train` call is one fit:** one block, dataset, model, seed and grid point.
  - It fits on the training data and scores validation and test once.
  - It writes `runs/<block>/<dataset>/<model>/s<seed>[-t<trial>]/` with `result.json`, `per_user.npz`, `train.log` and `status.json`.
- **Selection uses validation only,** on the block's metric. Ties go to the first grid point in config order, so the simplest point wins.
  - Seeds 1 and 2 rerun the selected point.
  - Deterministic models (Pop, ItemKNN, EASE) run once.
- **Trainers with `patience`** stop early on the validation metric and restore their best epoch.
  - SASRec and BERT4Rec on ML-1M keep their published budgets (201 epochs; 400k steps).
  - They log validation without stopping on it.
- **Every run fixes `exclude_seen=True`,** so all results of a dataset pair up in `compare_models`.
- **`compare` runs `compare_models`** on the seed-0 per-user vectors of each model's selected point.
  - Reference: the block's `reference`. Metrics: `calibrated_recall@20` and `ndcg@100`. Holm correction applies over the dataset's models and metrics.
  - It writes `table/<key>.json` (`ComparisonReport.to_frame()` records).
  - It writes `summary/<block>/<dataset>.json` with the mean and sd over seeds.
  - It also writes `meta.json` and `MANIFEST.json`, where every file is tagged tier `git` or `archive`.

## Catalogue blocks (C, C′)

- **Catalogues per fit:** every model is fitted once and scored in each of the block's catalogues.
  - Block C: `cold`, the phase's cold items; `full`, every item.
  - Block C′: `warm`, the training items; `mixed`, the phase's whole stage catalogue.
- **Shared histories:** columns follow the checkpoint's item IDs, and histories keep only training items, so every model sees the same histories and the results pair.
- **Feature models** (Content, TEASER, TEASER-GD, through `MMConcatWrapper`) fit on the whole catalogue with the training items marked. They are then pointed at each catalogue with `build_candidates`.
  - In-training validation therefore scores the full catalogue.
- **Warm-only models** (EASE, Pop) rank training items only. They are scored in the full and mixed catalogues, never in the cold one.
- **Cold share:** `result.json` records `cold_share` under the three definitions: catalogue items, target items and target interactions.
- **Encoded features:** `encode` writes C′ text features once per dataset under `features/`. It uses the block's sentence encoder on the checkpoint's `entity_text`.

## Hosts and publication

- **Host names:** results name machines by `CR_HOST_LABEL` (for example `ostrava-h100`), or by a digest of the hostname, never by the hostname itself. `result.json` records the checkpoint's file name, not its path.
- **Scrub check:** `compare` checks every text file for this machine's hostname, the user name and the home path. Hits are listed under `scrub.flagged` in `MANIFEST.json`.
- **Protocol in meta.json:** `compare` writes each block's selection rule, grids and epoch rule (`max N epochs, … patience P`, or a fixed budget) to `meta.json`, for the captions.

## Guarantees

- **Idempotent:** `train` skips a run whose `result.json` has the same spec. It exits with code 2 when the spec differs.
- **Checksums:** it also exits with 2 when the checkpoint's sha256 differs from `SHA256SUMS` next to it.
- **Writes:** it writes only under `--out`.
- **Smoke tests:** `--max-epochs N` caps every epoch budget. The cap is part of the spec, so give smoke runs their own `--out`.
