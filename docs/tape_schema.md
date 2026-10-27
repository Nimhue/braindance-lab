# BD tape schema

Everything the web app reads lives in `app/data/`. Notebook `07_export_tapes` (real data) and
`scripts/make_demo_data.py` (synthetic) both write this exact format via `braindance.export`.

Arrays are **columnar** (one array per field, same length) to keep files small. Floats are
rounded (signals to 0.1 µV, probabilities to 0.001).

## `manifest.json`

```json
{
  "demo": false,
  "generated_at": "2026-10-27T12:00:00+00:00",
  "source": "Sleep-EDF Expanded (PhysioNet) + MET (OpenNeuro)",
  "metrics": "metrics.json",
  "tapes": [
    {"id": "sleep-SC4001_n1", "kind": "sleep", "subject": "S00", "session": "SC4001_n1",
     "title": "S00 · night 1", "file": "sleep-SC4001_n1.json"}
  ]
}
```

`demo: true` makes the app show the synthetic-data banner.

## Sleep tape (`kind: "sleep"`)

| field | type | meaning |
|---|---|---|
| `stages` | `["W","N1","N2","N3","REM"]` | index → stage |
| `channels` | `["Fpz_Cz","Pz_Oz"]` | |
| `epoch_s` | `30` | |
| `epochs.t` | `float[]` | minutes from the first kept epoch |
| `epochs.true` | `int[]` | expert stage index |
| `epochs.pred` | `int[]` | HMM-smoothed out-of-fold prediction, `-1` = none (artifact) |
| `epochs.pred_raw` | `int[]` | prediction before smoothing |
| `epochs.conf` | `float[]` | max class probability |
| `epochs.artifact` | `0/1[]` | flagged in silver |
| `epochs.rel` | `{band: float[]}` | relative band power at Fpz-Cz (δ θ α σ β) |
| `snippets[]` | | a few waveforms: `epoch` (index into `epochs`), `fs` (50), `signals {ch: µV[]}` (12 s), `bands {band: µV[]}` (first channel, band-passed) |
| `radar` | `{axes, power, sync}` | mean relative power (normalised to cohort) and mean inter-channel coherence per band, both 0–1 |

## Emotion tape (`kind: "emotion"`)

| field | type | meaning |
|---|---|---|
| `rating` | `{valence, arousal, quadrant}` | listener's 1–9 self-ratings; quadrant `null` if either is exactly 5 |
| `windows.t` | `float[]` | seconds from session start (music starts at 5 s) |
| `windows.p_valence`, `p_arousal` | `float[]` | cross-subject out-of-fold P(high) |
| `windows.faa` | `float[]` | baseline-normalised frontal alpha asymmetry F3/F4 |
| `windows.artifact` | `0/1[]` | |
| `windows.rel` | `{band: float[]}` | relative band power, mean of F3 and F4 (δ θ α β γ) |
| `prediction` | `{p_valence, p_arousal, quadrant}` | song-level: mean of window probabilities |
| `snippets[]` | | `window` (index), `fs` (62.5), `signals {F3, F4}` (4 s), `bands` |
| `radar` | `{axes, power, sync}` | as above; sync = mean coherence over Fp1/Fp2, F3/F4, F7/F8, T7/T8 |

## `metrics.json`

```json
{
  "demo": false,
  "sleep":   {"classes": [...], "n_subjects": 20, "n_epochs": 41000,
              "raw": {"accuracy", "balanced_accuracy", "kappa", "macro_f1", "per_class_f1", "confusion"},
              "smoothed": {...}, "transition": [[...]], "mlflow_run_id": "..."},
  "emotion": {"n_subjects": 5, "n_songs": 100,
              "cross_subject": {"valence": {"window": {...}, "song": {...}}, "arousal": {...}},
              "within_subject": {...},
              "null": {"valence": {"mean", "p95", "draws"}, "arousal": {...}}}
}
```
