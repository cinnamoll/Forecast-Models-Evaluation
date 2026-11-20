# TimeSHAP per-covariate run on Kaggle

Same explainer as `scripts/colab/` — these three scripts are thin wrappers that set
Kaggle paths and then delegate, so the driver logic lives in one place. Read
`scripts/colab/README.md` for what each output file means and for the caveats about
`hour`, the covariate set and model comparability.

## Kaggle differences that actually bite

| | Colab | Kaggle |
|---|---|---|
| code | `git clone` into `/content` | not on the image: enable Internet and clone, or attach the repo as a **Kaggle Dataset** and point `REPO_DIR` at `/kaggle/input/<slug>/Forecast-Models-Evaluation` |
| persistence | mount Drive | `/kaggle/working` is saved on **Save version**; `/kaggle/temp` is discarded |
| results default | `/content/drive/MyDrive/timeshap_results` | `/kaggle/working/results` |
| cells | `!source /content/timeshap_env.sh && ...` | `!cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/...` (the wrapper sources the env file itself) |
| GPU | T4 / L4 / A100 | usually T4 ×2 (16 GB) or P100 (16 GB, `sm_60`, **no bf16**) — pick T4 when the notebook lets you |
| limit | ~12 h, disconnects | 12 h per session, weekly GPU quota |

Two consequences:

- On a **P100** Chronos2's bf16 path has no kernel: expect either a fallback warning
  or an error. Switch the GPU or run Chronos2 on CPU there (small series are fine:
  0.08 s per coalition measured on CPU at `--sampling 25`).
- A Kaggle session that dies mid-model loses that model only. `03_sweep.sh` skips
  every model whose `coalitions.json` exists, so re-running resumes.

## Order of cells

1. Turn the GPU **and** the Internet on (Settings → Accelerator → GPU T4 ×2;
   Settings → Internet → on), then check:

```bash
!nvidia-smi
```

2. Put the code in place — with Internet on:

```bash
!git clone https://github.com/cinnamoll/Forecast-Models-Evaluation.git /kaggle/working/Forecast-Models-Evaluation
```

   or, without Internet, attach the repository as a Kaggle Dataset and skip to step 3
   with `REPO_DIR=/kaggle/input/<dataset-slug>/Forecast-Models-Evaluation`.

3. Setup (never installs torch; Kaggle's image already ships a CUDA build). Add
   `INSTALL_TIMESFM=1` in front if the sweep includes TimesFM3:

```bash
!cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/01_setup.sh
```

4. Dry run — prints the player set of the Belgium PV game, no forward:

```bash
!cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/02_run.sh --dry-run
```

5. One series:

```bash
!cd /kaggle/working/Forecast-Models-Evaluation && \
  DATASET=belgium SERIES=pv_house_2 MODELS=Chronos2 \
  COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc" \
  bash scripts/kaggle/02_run.sh
```

6. The roster (13 series × 3 models), or a slice of it:

```bash
!cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/03_sweep.sh --dry-run
!cd /kaggle/working/Forecast-Models-Evaluation && MODELS="Chronos2 Timestral" bash scripts/kaggle/03_sweep.sh
```

7. Save the notebook version (commits `/kaggle/working`), then download
   `results/xai_timeshap/` from the output tab.

## Time budget at `--sampling 25`

Measured on an RTX 3050 for Belgium PV (2184 points, 48-step horizon):

```
7 cột + history (11 player,  2048 coalition) : Chronos2 ~3 min  | Timestral ~9 min
8 cột + history (12 player,  4096 coalition) : Chronos2 ~6 min  | Timestral ~18 min
7 cột only      ( 7 player,   128 coalition) : Chronos2 ~12 s   | Timestral ~35 s
TimesFM3                                     : chưa đo; 2 forward/coalition
```

13 series × (Chronos2 + Timestral) ≈ 3 h at 7 columns. TimesFM3 has to be measured
before it is scheduled — on a T4 it is the slowest of the three by a wide margin.
