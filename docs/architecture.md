# Architecture

```
 PhysioNet Sleep-EDF ──┐                                    ┌── gold.sleep_features ──► 05 train_sleep ──► gold.sleep_predictions ─┐
  (EDF + hypnograms)   │   bronze            silver         │   (+ ±2 epoch context)     GroupKFold + HMM                          │
                       ├─► Volume files ─► silver.epochs ─► gold.epoch_features ─┤                                                     ├─► 07 export ─► app/data ─► Netlify
 OpenNeuro MET ────────┘   + manifests     silver.epoch_labels  (map<string,double>)│                                                     │
  (BIDS, 128 ch)           + raw ratings                    └── gold.emotion_features ─► 06 train_emotion ─► gold.emotion_predictions ┘
                                                                (baseline deltas)        cross/within subject + permutation null
```

## Layers

| Layer | Table | Grain | Key decisions |
|---|---|---|---|
| bronze | Volume `raw/` | file | EDF/BIDS files land untouched — the file is the bronze record |
| bronze | `sleep_edf_files`, `met_files` | recording | manifest: paths, sizes, landing time; downloads are idempotent |
| bronze | `sleep_edf_hypnogram` | annotation run | R&K labels as scored (`Sleep stage 4`, `Movement time`) |
| bronze | `met_ratings` | song | normalised subject/session keys, raw 1–9 ratings |
| silver | `epochs` | window | filtered on the **continuous** signal, then windowed; signal flat `array<float>` (channel-major) + `n_samples`; artifacts flagged not dropped; Delta `CHECK` constraints |
| silver | `epoch_labels` | window × label | long format: `sleep_stage` / `valence` / `arousal` / `quadrant` |
| gold | `epoch_features` | window | one `map<string,double>` per window — schema-flexible across sources |
| gold | `sleep_features` | epoch | widened map + lag/lead context + time-of-night |
| gold | `emotion_features` | music window | widened map + `d__*` deltas vs the session's pre-music baseline |
| gold | `*_predictions` | window | **out-of-fold** predictions only — what the app shows |

## Why these choices

**One schema, two sources.** Sleep (30 s, 100 Hz, 2 channels, expert labels per epoch) and MET
(4 s, 250 Hz, 8 channels, one self-rating per song) differ in almost every parameter, so silver
stores *windows* generically (`source`, `fs`, `channels`, `n_samples`, `signal`) and gold stores
features as a map. Source-specific shape only appears in the two model-ready gold tables.

**Filter before windowing.** A 0.5 Hz high-pass on a 30 s window rings at both edges. Filtering the
whole recording once, then cutting, avoids 2 × N edge artefacts per night.

**Flat signal arrays.** Arrow (used by `applyInPandas` / `mapInPandas`) handles flat
`array<float>` reliably and cheaply; nested arrays are where pandas UDFs get fragile.

**Code ships by value.** `00_config` registers the `braindance` package with cloudpickle by value,
so UDFs work on shared and serverless compute without `addPyFile`. The Asset Bundle
(`databricks.yml`) installs the wheel for scheduled runs.

**Subject-grouped evaluation, everywhere.** Neighbouring windows of one person are near-copies.
Any random split leaks identity; notebook 05 measures exactly how much κ that inflates.

**Sleep grammar.** Per-epoch classifier probabilities are decoded with Viterbi over a transition
matrix learned from training folds only (scaled-likelihood emissions: P(stage|x) / P(stage)).

**Honest emotion numbers.** Three views of the same model: cross-subject, within-subject, and a
permutation null with session labels shuffled. A score only counts if it clears the null's 95th
percentile.

## Sizing (defaults)

* Sleep-EDF, 20 subjects ≈ 40 nights ≈ 2 GB of EDF; after wake trimming ≈ 40k epochs × 6,000 floats ≈ 1 GB in silver.
* MET, 5 subjects: download size depends on the BIDS layout (full 128-ch nets at 1000 Hz are large — start small).
  Silver keeps only 8 channels at 250 Hz: 100 songs × 60 windows × 8,000 floats ≈ 200 MB.
* Tapes: ~100–400 KB each; the whole `app/data` stays a few MB.
