# Past-Only Evidence Layer and Neural Hybrid for Fraud and Money-Laundering Detection

**Author:** Hasin Md. Daiyan · **Date:** 14 October 2026
**Live results page:** `<your-vercel-link>`

> The evidence layer, not the model architecture, carries the result. Given the same past-only evidence, a fully neural hybrid is slightly below LightGBM on IEEE-CIS card fraud (0.630 vs 0.635 PR-AUC) and ahead of it on IBM AMLworld money laundering (0.650 vs 0.625), with the smallest seed-to-seed spread of any model on both.

## Repository contents

| File | Description |
|---|---|
| `hybrid_ieee_cis.ipynb` | Full pipeline on IEEE-CIS (card fraud, 30-day label delay) |
| `hybrid_amlworld.ipynb` | Same pipeline on IBM AMLworld HI-Small (money laundering, 1-day label delay) |
| `Fraud_Detection_Method_and_Results.pdf` | Complete write-up: method, every table, ablations, literature comparison |
| `index.html` | Interactive results page (static, no build step) |
| `README.md` | This file |

Every number in the document and the web page is printed by one of the two notebooks or read from their stored per-seed result files.

## Method in brief

1. **Evidence layer (87 features).** Built only from earlier transactions. A fraud label is used only once it is older than the label delay (30 days on IEEE-CIS, 1 day on AMLworld).
   - Entity memory (48): count, age, mean amount, smoothed fraud rate, visible labels, first-sighting flag for 8 entities
   - Deviation features (11): how the transaction differs from the same user's past
   - Neighbour (graph) features (15): other users on the same card / address / device (IEEE-CIS) or receiver / owner (AMLworld)
   - Evidence signals (13): how much is known about the transaction
2. **Two neural component models.** TabM+G (TabM-mini with learned feature gates and piecewise-linear encoding) reads what the transaction looks like. The deviation model (+G) attends over the same user's last 16 transactions.
3. **Combination.** Fixed weighted average of the two logits, `score = w1*s1 + w2*s2 + b`, learned once on out-of-fold scores (4 time-blocked folds). An evidence gate is reported as an ablation only.

Baselines: LightGBM on raw features (trained once and retrained periodically) and LightGBM+ (raw features plus the same evidence layer).

## Protocol

- Strict chronological split: 70% train / 7.5% stop / 7.5% calibration / 15% test
- Seeds 42, 7, 123; results are test-set means
- Paired bootstrap (200 resamples per seed); a difference is "better" or "worse" only if the 95% interval excludes zero on every seed
- Leak checks: label-flip test, past-only index checks, shuffled-label null test (ROC-AUC 0.501 on IEEE-CIS, 0.488 on AMLworld)

## Headline results (test set, mean of 3 seeds)

| Model | IEEE-CIS PR-AUC | AMLworld PR-AUC |
|---|---|---|
| LightGBM, raw features | 0.5459 | 0.1571 |
| LightGBM, raw, retrained periodically | 0.5203 | 0.1513 |
| LightGBM+ (raw + evidence layer) | 0.6354 | 0.6253 |
| TabM+G alone | 0.6047 | 0.6375 |
| Deviation model (+G) alone | 0.5948 | 0.6327 |
| **Hybrid (fixed weighted average)** | **0.6296** | **0.6502** |

Test sets: IEEE-CIS 88,581 rows (3.48% fraud); AMLworld 761,586 rows (0.119% laundering).

## Key findings

- The evidence layer lifts LightGBM by +0.09 PR-AUC (IEEE-CIS) and +0.47 (AMLworld). Retraining LightGBM never replaces it.
- On IEEE-CIS the lift comes from delayed fraud labels (label-free layer: −0.003). On AMLworld account structure alone lifts PR-AUC from 0.157 to 0.562.
- The hybrid beats LightGBM+ on AMLworld on every seed (+0.025) and for regular users on IEEE-CIS (+0.023); trees are better for new and few-history users on IEEE-CIS.
- TabM beats TabNet by about +0.09 PR-AUC on both datasets.
- A fixed weighted average is at least as good as a per-transaction evidence gate on both datasets.

## Datasets (not included)

Download them yourself and update the paths in the first cells of each notebook:

- **IEEE-CIS Fraud Detection:** Kaggle competition data (`train_transaction.csv`, `train_identity.csv`). Only the training files are used because the Kaggle test file has no labels.
- **IBM AMLworld (HI-Small):** transactions and accounts files from the release accompanying Altman et al., NeurIPS 2023 Datasets and Benchmarks.

## Reproducing

1. Open the notebook for your dataset and install the packages imported in its first cells (the notebooks are the source of truth for versions).
2. Run top to bottom. Seeds 42, 7 and 123 are run in that order.
3. The AMLworld notebook was run with LightGBM on GPU and mixed-precision neural training (5M rows); the IEEE-CIS notebook runs LightGBM on CPU and the neural models in 32-bit. A GPU is strongly recommended for both.

## How the two notebooks differ

The AMLworld notebook is the IEEE-CIS code base plus a dataset adapter. The protocol, leak checks, evidence-layer functions, component models, cross-fitting, combination, LightGBM settings and seeds are identical. Differences: customer identity (uid proxy vs real accounts), entity and neighbour definitions, deviation sequence columns (19 vs 8), label delay (30 days vs 1 day), retraining cadence, removal of the 11-18 September simulation-artefact tail, and L0 pruning strength (0.5 vs 0.02, scaled by base-rate entropy).

## Known limitations

- Bootstrap intervals cover test-set sampling only; the three seeds cover training randomness.
- The hybrid averages 8 networks per score, while each LightGBM baseline is one model, so part of its stability comes from averaging.
- AMLworld is synthetic and spans 10 active days. Comparison with published F1 scores is indicative, not head-to-head.
- Some AMLworld ablation rows were run on seeds 42 and 123 only (marked in the document). Minority-class F1 is printed for seed 123 only.

## Hosting the web page

`index.html` is a single static file. On Vercel, import the GitHub repository with the framework preset "Other" and no build command. Keep the PDF and notebooks in the repository root so the page's links work.

## References

Altman et al. (2023) AMLworld · Gorishniy et al. (2025) TabM · Arik and Pfister (2021) TabNet · Louizos et al. (2018) L0 regularisation · Grover et al. (2022) Fraud Dataset Benchmark · Egressy et al. (2024) Multi-PNA · Lin et al. (2024) FraudGT · Blanuša et al. (2024) Graph Feature Preprocessor. Full list in the PDF.
