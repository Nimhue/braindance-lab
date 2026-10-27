# Braindance Lab 🧠📼

Real EEG, recorded and played back like a braindance. A Databricks lakehouse pipeline that takes
raw brain recordings — **a night of sleep** and **a person listening to a song** — cuts them into
windows, extracts the rhythms (δ θ α σ β γ), and trains models to read the brain state.
A small web app, the **BD editor**, plays the results back window by window.

The point isn't a record-breaking classifier. It's showing, with real data, why **reading sleep is
easy and reading emotion is hard** — and measuring both honestly.

| | Sleep | Emotion |
|---|---|---|
| Dataset | [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet), sleep-cassette | MET music-induced EEG (OpenNeuro, BIDS) |
| Signal | 2 EEG channels, 100 Hz, whole nights | 8 of 128 channels, 1000 → 250 Hz, 130 s per song |
| Labels | expert hypnogram, every 30 s | one self-rating per song (valence, arousal 1–9) |
| Window | 30 s | 4 s, 2 s step, + 4 s pre-music baseline |
| Model | gradient boosting + HMM (Viterbi) smoothing | gradient boosting on baseline-normalised features |
| Honest test | GroupKFold by subject; leaky split shown for contrast | cross-subject vs within-subject vs shuffled-label null |

## Repo

```
braindance-lab/
├── src/braindance/          signal + feature library (numpy/scipy, unit-tested)
│   ├── signal.py            band-pass, notch, resampling, windowing, artifact flags
│   ├── features.py          Welch band power, entropy, spectral edge, Hjorth, coherence, asymmetry
│   ├── labels.py            R&K → AASM stages, wake trimming, rating → valence/arousal/quadrant
│   ├── smoothing.py         transition matrix + Viterbi
│   ├── modeling.py          subject-grouped CV for sleep and emotion, permutation null
│   ├── export.py            BD tape builders (shared by notebook 07 and the demo script)
│   ├── synth.py             synthetic EEG for tests and demo tapes
│   └── sources/             sleep_edf.py, met.py adapters
├── notebooks/               Databricks notebooks, run in order
│   ├── 00_config.py         names, settings, makes src/ importable on workers
│   ├── 01_bronze_sleep_edf.py
│   ├── 02_bronze_met.py     ← includes an inspection step: confirm layout before trusting it
│   ├── 03_silver_epochs.py
│   ├── 04_gold_features.py
│   ├── 05_train_sleep.py
│   ├── 06_train_emotion.py
│   └── 07_export_tapes.py
├── app/                     the BD editor (static HTML/CSS/JS, no build) + data/
├── scripts/make_demo_data.py synthetic end-to-end run → app/data (no downloads)
├── tests/                   pytest, incl. an EDF write/read round trip
├── docs/                    architecture.md, tape_schema.md
├── databricks.yml           Asset Bundle: wheel + 7-task job
└── netlify.toml
```

## Run it

### 1. Locally (no data needed)

```bash
pip install -e ".[dev]"
pytest                                   # 26 tests
python scripts/make_demo_data.py         # synthetic tapes → app/data
python -m http.server -d app 8000        # open http://localhost:8000
```

The demo tapes are **synthetic** (caricatured spectra per stage / quadrant) and the app says so.

### 2. On Databricks

1. Add the repo as a Git folder.
2. In `notebooks/00_config.py` set `CATALOG` (default `braindance`) and **`MET_OPENNEURO_ID`**
   (from the MET dataset's OpenNeuro page). Adjust `SLEEP_EDF_SUBJECTS` / `MET_SUBJECTS` for size.
3. Run `01` → `07` in order on an **ML runtime** cluster (15.4 LTS ML tested settings), or deploy the job:
   ```bash
   databricks bundle deploy -t dev
   databricks bundle run braindance_pipeline -t dev
   ```
4. **MET needs one manual look.** `02_bronze_met` prints the BIDS tree, the events/ratings tables and the
   channel names. From that, set `MET_LAYOUT` (one file per song, or a long file with onsets in
   `events.tsv`) and `MET_RATINGS` (where valence/arousal live, column names) in `00_config`. The adapter
   refuses to guess silently: missing channels or rating columns raise errors with the names it found,
   and `03` prints how many sessions have no matching rating (should be 0).
5. Copy the exported tapes into the app and deploy:
   ```bash
   databricks fs cp -r dbfs:/Volumes/braindance/gold/tapes ./app/data --overwrite
   ```
   Netlify: connect the repo; `netlify.toml` publishes `app/`.

## What to look for in the results

* **04's sanity table** before any ML: N3 delta-heavy, wake alpha-heavy at Pz-Oz, N2 sigma (spindles).
* **05:** κ raw vs smoothed — the HMM usually helps most on N1/REM flicker. Then the leaky epoch-split κ, which is higher and meaningless.
* **06:** whether song-level balanced accuracy clears the shuffled-label 95th percentile — cross-subject often won't, within-subject more often does. Either outcome is a finding.

## Notes and limits

* Not a medical device or a diagnostic tool. Sleep-EDF subjects are healthy volunteers.
* Sleep-EDF is scored in Rechtschaffen & Kales; stages 3 and 4 are merged into N3.
* MET channel mapping uses approximate 10-20 equivalents on the EGI HydroCel-128 net
  (`met.HYDROCEL128_TO_1020`) — verify against the dataset's `channels.tsv`/`electrodes.tsv`.
* Check each dataset's licence and citation requirements before publishing derived files.
  DEAP was considered and not used: access is limited to permanent academic staff.

## Credits

Sleep-EDF: Kemp et al., PhysioNet. MET: Cheang et al. (Music-Induced EEG Dataset), OpenNeuro.
Built by Nim ([@Nimhue](https://github.com/Nimhue)).
