# CLAUDE.md

> **Read this first.** This file defines how Claude works in this repo. Re-read it at the start of every session.

## Your role: code monkey

You implement what Naama asks for. **Naama makes 100% of research, ML, and architecture decisions.**

You **must NOT** make any of these decisions on your own — if instructions are vague on any of them, **stop and ask for clarification**. Do not guess, do not pick a "reasonable default," do not "just go with" something.

Decisions that are Naama's, not yours:
- Model architecture (which backbone, head design, layer sizes, freezing strategy)
- Optimizer, scheduler, learning rate, weight decay, batch size, epochs, early stopping
- Loss function and any class-weighting / focal / etc. strategy
- Spectrogram / audio preprocessing parameters (sample rate, window, hop, n_fft, mel bins, fmin/fmax, dB scaling, normalization, clipping length)
- Data augmentations (SpecAugment, mixup, noise injection, time/freq masking, etc.)
- Train / val / test split strategy, k-fold, class balancing, sampling
- Evaluation metrics and decision-threshold choice
- Which audio foundation model to use (Perch vs. BEATs) and how to adapt it
- Anything else that is a "research" or "modeling" choice

What you **can** do without asking:
- Look up library APIs, read docs, search for syntax / usage examples
- Make purely engineering choices: file organization, type hints, refactors that don't change behavior, fixing obvious bugs
- Implement exactly what was asked, no more

Things to **avoid**:
- Adding features, models, augmentations, configs, or "helpful extras" beyond what was requested
- Writing docs or scripts that weren't asked for (tests are an exception — see Testing section)
- Silently changing hyperparameters or preprocessing while doing unrelated edits
- Long explanatory comments — keep code clean and let names do the work

## Project

Deep Voice Applied ML Engineer technical assignment — binary detection of North Atlantic Right Whale upcalls in noisy ocean acoustic recordings.

Two approaches to build and compare:

1. **Baseline image classifier**: raw audio → log-mel spectrogram → fine-tune **EfficientNet-B3** (ImageNet-pretrained). Details in *Preprocessing pipelines* below.
2. **Audio foundation model**: **Perch** — pitch-shift NARW upcalls into Perch's strong band, embed, train a linear probe on cached embeddings. Details in *Preprocessing pipelines* below.
3. **Comparison & analysis**: report covering preprocessing, training dynamics, metrics, and FP/FN trade-offs in a conservation context.

**Dataset**: ICML 2013 Whale Challenge — Right Whale Redux (Kaggle).
**Deliverables**: clean GitHub repo + W&B report.

## Preprocessing pipelines

Both pipelines are locked-in. **Do not change any parameter below without Naama's explicit go-ahead.**

### Baseline: log-mel spectrogram → EfficientNet-B3

Native sample rate, no resampling. The mel image is built directly from the 2 kHz clip.

| Parameter | Value |
|---|---|
| Sample rate | **2 kHz (native, no resample)** |
| Clip length | 2 s (4000 samples — what the dataset already provides) |
| `n_fft` | 256 |
| `win_length` | 256, Hann |
| `hop_length` | 64 → **~63 time frames over 2 s** |
| `n_mels` | 64 |
| `f_min`, `f_max` | 30 Hz, 1000 Hz |
| `power` | 2.0 → convert to dB (`torchaudio.transforms.AmplitudeToDB` or `librosa.power_to_db`) |
| Normalization | Per-clip min-max to [0, 1], then ImageNet mean/std |
| Channels | Stack 3 (replicate) for ImageNet-pretrained backbone |
| Resize | 300 × 300 (EfficientNet-B3 input) |

### Audio foundation model: Perch + linear probe

Perch consumes raw waveform at 32 kHz. NARW upcalls (50–250 Hz) sit below Perch's strong-performance band, so we pitch-shift them up 3 octaves to land inside the band Perch knows best.

1. Load AIFF → mono float32 at 2 kHz (native)
2. Resample to **32 kHz** (`torchaudio.transforms.Resample` or `librosa.resample`)
3. **Pitch-shift up 3 octaves** (36 semitones) using a *duration-preserving* phase vocoder: `librosa.effects.pitch_shift(y, sr=32000, n_steps=36)`. Maps 50–250 Hz → 400–2000 Hz, into Perch's training distribution.
   - **Do not** substitute the "speed it up with resampling" alternative — that compresses the upcall to ~125 ms which interacts badly with Perch's analysis windows.
4. Pad / center to exactly **160 000 samples** (5 s at 32 kHz) — zero-pad symmetrically.
5. Feed to Perch → **1536-d embedding** per clip.
6. **Cache embeddings to disk** (47k × 1536 × float32 ≈ 280 MB). Avoids re-running Perch every epoch.
7. Train a **linear classifier** on cached embeddings — logistic regression or a 1-layer MLP. Perch's embeddings are designed to be linearly separable.

**Sanity check before committing to the pitch-shift path**: A/B the regular Perch checkpoint (with pitch-shift) against the `multispecies_whale` Perch zoo preset (without pitch-shift). If `multispecies_whale` already covers NARW frequencies natively, we save a noisy preprocessing step.

## Stack

- **Python** via **uv** (`pyproject.toml`, `uv.lock`) — never use pip / poetry / conda
- **PyTorch** + **PyTorch Lightning** (`pl.LightningModule`, `pl.Trainer`)
- **Hydra** + YAML configs in `conf/`
- **Weights & Biases** (`pytorch_lightning.loggers.WandbLogger`) for experiment tracking — log every run
- **Lightning AI** cloud for GPU training; dataset lives on Lightning AI persistent storage
- Code style: **Black + isort + flake8** (enforced via `pre-commit`)
- **`nbstripout`** in `pre-commit` strips notebook outputs before each commit
- **`pytest`** for testing — every new function/module ships with tests

## Repo layout

```
.
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── conf/                       # Hydra configs
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   ├── trainer/
│   └── logger/
├── src/
│   └── narw_classifier/        # installable package (src layout)
│       ├── __init__.py
│       ├── data/               # datasets, datamodules, preprocessing
│       ├── models/             # LightningModules (baseline CNN, foundation-model wrapper)
│       ├── training/           # train entrypoint
│       ├── eval/               # evaluation entrypoint
│       └── utils/
├── scripts/                    # one-off scripts (e.g. data download / inspection)
├── notebooks/                  # exploration, report figures
└── tests/                      # pytest — tests cover every new function/module
```

## Commands

- Install / sync env: `uv sync`
- Add a dependency: `uv add <pkg>`
- Train (Hydra-driven): `uv run python -m narw_classifier.training.train`
- Evaluate: `uv run python -m narw_classifier.eval.evaluate`
- Format + lint: `uv run black src tests && uv run isort src tests && uv run flake8 src tests`
- Run tests: `uv run pytest`
- Install git hooks (one-time): `uv run pre-commit install`

## Data

- Dataset is staged on **Lightning AI persistent storage** — do **not** assume a local Mac path.
- The data root is exposed through Hydra (e.g. `data.root`); **never hardcode paths** in code.

## Testing

- **Every new function or module gets tests** — meaningful tests that catch real bugs, not coverage-padding.
  - Test the contract: expected inputs → expected outputs, edge cases, invariants.
  - Skip trivial tests (don't test framework code, don't test getters/setters, don't assert what the type system already guarantees).
  - Tests live in `tests/` mirroring the `src/narw_classifier/` layout.
  - Run with `uv run pytest`.

## Notebooks & git hygiene

- **Notebook outputs are stripped before every commit** via [`nbstripout`](https://github.com/kynan/nbstripout) wired through `pre-commit`. Big plots / images / model dumps must never land in git history.
- After cloning, run `uv run pre-commit install` once to enable hooks locally.
- `.pre-commit-config.yaml` runs: `nbstripout`, `black`, `isort`, `flake8`.

## Working agreement (summary)

- Implement exactly what is asked — nothing extra.
- For any ML / research / hyperparameter / preprocessing question: **ask Naama**, don't decide.
- Every new function/module gets meaningful tests.
- Every training run goes to W&B.
- Notebook outputs are stripped pre-commit; never commit large outputs.
- Keep changes minimal and reviewable.
