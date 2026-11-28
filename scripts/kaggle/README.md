# Chay tren Kaggle - LightGBM vs 3 zero-shot checkpoint

| model | script | env |
| --- | --- | --- |
| LightGBM | `models/models_statsml.py --models LightGBM` | `statsml` |
| Chronos2 | `models/fmodel_chronos2.py` | `chronos` |
| Timestral | `models/fmodel_timestral.py` | `chronos` |
| TimesFM3 | `models/fmodel_timesfm3.py` | `timesfm3` |

Baseline hoc sau cua bai cu (DeepNPTS, TCN, TFT, TimeXer) da bi bo khoi luong nay.

## 0. Bo covariate (da rut gon)

    calendar   hour, dayofweek
    weather    temperature, humidity, cloud_cover, ghi, wind_speed
    price/soc  day_ahead_price, soc                  (Belgium)

Da bo han `quarter_hour`, `minute`, `is_weekend` (calendar) va `dni`, `precipitation`
(weather) -- ca trong `utils/covariates.py`, `utils/dataset_config.py`,
`utils/exogenous/sources/openmeteo.py` lan `data/exogenous/weather/*.csv`.
COV frame vi vay con 7 cot (germany/london/zonnedael), 8 cot (Belgium PV) va 9 cot
(Belgium BESS). Muon chay lai dung bo covariate cu thi phai checkout commit truoc do,
khong con duong nao bat lai 5 bien do bang flag.

## 1. Setup (mot lan moi session)

    !git clone https://github.com/<user>/Forecast-Models-Evaluation /kaggle/working/Forecast-Models-Evaluation
    !cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/01_setup.sh

- Tao env `statsml`, `chronos`, `timesfm3` tu `requirements/env_*.yml`, cai them `pandas`
  vao conda base (vi `02_run.py` import `utils/protocol.py` trong chinh interpreter goi no,
  truoc khi kip shell ra model env), roi chay dry-run.
- Miniforge nam o `/tmp/miniforge3` (mac dinh) nen **session moi phai chay lai setup**;
  dat `CONDA_HOME=/kaggle/working/miniforge3` neu chap nhan 5-8 GB trong output 20 GB.
- Khong bat buoc `source /kaggle/working/env.sh` nua: ca ba script tu tim conda
  (`scripts/kaggle/_bootstrap.sh`, `ensure_conda()` trong `02_run.py`). Chi source khi muon go
  tay `conda`/`python` trong cell.

## 2. Goc ket qua (hardcode san, khong can export)

Ca ba script ghi vao cung mot folder, da hardcode trong `scripts/kaggle/_bootstrap.sh`
(shell) va `default_results_root()` (`02_run.py`):

    /kaggle/working/results_cov_new          <- tren Kaggle
    <REPO_DIR>/results_cov_new               <- khi /kaggle/working khong ton tai

Khong dat `export RESULTS=...` trong cell: do la lenh shell, notebook python bao
`SyntaxError: invalid syntax`. Muon doi goc cho mot lan chay thi truyen inline:

    !cd /kaggle/working/Forecast-Models-Evaluation && RESULTS=/kaggle/working/results_khac \
        python scripts/kaggle/02_run.py --models "Chronos2 Timestral TimesFM3" --level COV

(`--results-dir <path>` cho rieng `02_run.py`.) Runner chi nap tuning JSON khi goc cua cell
trung voi goc cua lan tune, nen dung mot gia tri cho ca ba buoc.

Ket qua khi do:

    <root>/results_<dataset>/<Model>/Sampling_<tag>/COV      <- benchmark
    <root>/tuning/                                           <- tuning JSON + logs
    <root>/xai_timeshap/<dataset>/<series>_COV_<tag>/        <- TimeSHAP

`results_covariate` cu trong repo vi vay khong bi tron; COV va L0 cung khong de len nhau
vi `LEVEL` la mot thanh phan duong dan.

## 3. Tune LightGBM (chay truoc khi benchmark COV)

    !cd /kaggle/working/Forecast-Models-Evaluation && MODELS=LightGBM TRIALS=30 HOURS=1.5 bash scripts/kaggle/03_tune.sh

Ba tuning set: `germany`, `zonnedael9`, `belgium_pv1`; params duoc transfer sang cell bao cao
theo map trong `utils/tuning.py` (london <- zonnedael9, belgium <- belgium_pv1). Tuning JSON
nam trong `$RESULTS/tuning/`. Study resume tu sqlite cung ten, nen session sau chay them trials
chu khong lam lai tu dau; study da du `TRIALS` thi bi skip. `DRY=1` de chi in lenh.

## 4. Chay benchmark

    # LightGBM truoc, de dung tuned params cua buoc 3
    !cd /kaggle/working/Forecast-Models-Evaluation && python scripts/kaggle/02_run.py \
        --models LightGBM --datasets germany london zonnedael belgium --level COV

    # 3 zero-shot checkpoint
    !cd /kaggle/working/Forecast-Models-Evaluation && python scripts/kaggle/02_run.py \
        --models "Chronos2 Timestral TimesFM3" --level COV

- `--models` / `--datasets` nhan ca hai dang: `--models "Chronos2 Timestral TimesFM3"`
  (mot chuoi, tu tach theo khoang trang) va `--models Chronos2 Timestral TimesFM3`.
  Ten la sai se bao `FAIL: unknown model(s): ...` thay vi KeyError.
- Bo `--samplings` de chay du 4 grid (25, 33, 50, 100 %); them `--force` de chay lai cell da xong.
- `--results-dir <path>` ghi de `$RESULTS` cho rieng lan chay nay (`--gpu` mac dinh 0, do
  wrapper them vi runner khong bao gio them flag nay).
- `--dry-run` in ra dung cac lenh `conda run -n <env> python models/... --gpu 0` ma khong chay.
- Neu khong co conda tren PATH (cell moi tinh), script tu do `$CONDA_HOME` -> `/opt/conda`
  -> `~/miniforge3` -> `~/anaconda3` -> `~/miniconda3` -> `/kaggle/working/miniforge3`.

## 5. TimeSHAP

    !cd /kaggle/working/Forecast-Models-Evaluation && MODELS="LightGBM Chronos2 Timestral TimesFM3" \
        bash scripts/kaggle/04_xai.sh

- `BLOCKS=1` (mac dinh) -> `--all-covariates`: 1 player moi block, 5-7 block (32-128 coalition).
- `BLOCKS=0` -> `--per-variable`: 1 player moi cot, 7-9 cot (128-512 coalition).
- `COVARIATES` mac dinh `hour dayofweek humidity temperature cloud_cover` (dung khi `BLOCKS=0`).
- Script truyen `--params-dir "$RESULTS/tuning"`, nen chay buoc 3 truoc neu muon LightGBM
  dung tuned params thay vi defaults cua bai. `DRY=1` in lenh, `XAI_DRY_CHECK=1` in so
  player/coalition ma khong chay.

## Luu y

- LightGBM mac dinh chay 10 run (`--runs 1` neu muon khop cay ket qua cua bai).
- `models_statsml.py` can env `statsml` (`pmdarima`), khong chay duoc bang env khac.
- Chi cell co JSON trong `$RESULTS/tuning/` moi nap tuned params.
- Ket qua cu trong `results_covariate/` duoc sinh voi bo covariate 10 cot (co `dni`,
  `precipitation`, `quarter_hour`, `minute`, `is_weekend`): so sanh truc tiep voi lan
  chay moi la khong hop le, phai chay lai ca 4 model -- va nen chay vao `$RESULTS` rieng.
