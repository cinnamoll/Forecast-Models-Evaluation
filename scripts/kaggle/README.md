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

## 1. Setup

    !git clone https://github.com/<user>/Forecast-Models-Evaluation /kaggle/working/repo
    !cd /kaggle/working/repo && bash scripts/kaggle/01_setup.sh

`01_setup.sh` ghi `/kaggle/working/env.sh` de source lai moi cell sau.

## 2. Tune LightGBM (chay truoc khi benchmark COV)

    !cd /kaggle/working/repo && MODELS=LightGBM TRIALS=30 HOURS=1.5 bash scripts/kaggle/03_tune.sh

Ba tuning set: `germany`, `zonnedael9`, `belgium_pv1`; params duoc transfer sang cell bao cao
theo map trong `utils/tuning.py` (london <- zonnedael9, belgium <- belgium_pv1).
Tuning JSON nam trong `<RESULTS>/tuning/`, va runner chi nap chung khi `<RESULTS>` cua
cell trung voi `<RESULTS>` cua lan tune -- do la ly do ca ba script dung chung mot quy tac
goc duoi day.

## 3. Chay benchmark

    !cd /kaggle/working/repo && python scripts/kaggle/02_run.py \
        --models LightGBM --datasets germany london zonnedael belgium --level COV

    !cd /kaggle/working/repo && python scripts/kaggle/02_run.py \
        --models "Chronos2 Timestral TimesFM3" --level COV

- Bo `--samplings` de chay du 4 grid (25, 33, 50, 100 %); them `--force` de chay lai cell da xong.
- Goc ket qua theo `--level`: COV -> `<parent>/results_covariate`, L0 -> `<parent>/results`
  (tren Kaggle: `/kaggle/working/results_covariate`). `--results-dir` hoac `export RESULTS=...`
  se ghi de ca hai level cung mot goc.
- `--gpu` mac dinh 0; runner khong them flag do, wrapper nay them.

## 4. TimeSHAP

    !cd /kaggle/working/repo && MODELS="LightGBM Chronos2 Timestral TimesFM3" \
        bash scripts/kaggle/04_xai.sh

- `BLOCKS=1` (mac dinh) -> `--all-covariates`: 1 player moi block, 5-7 block (32-128 coalition).
- `BLOCKS=0` -> `--per-variable`: 1 player moi cot, 7-9 cot (128-512 coalition).
- `COVARIATES` mac dinh `hour dayofweek humidity temperature cloud_cover` (dung khi `BLOCKS=0`).
- Script truyen `--params-dir "<RESULTS>/tuning"`, nen chay buoc 2 truoc neu muon LightGBM
  dung tuned params thay vi defaults cua bai.

## Luu y

- LightGBM mac dinh chay 10 run (`--runs 1` neu muon khop cay ket qua cua bai).
- `models_statsml.py` can env `statsml` (`pmdarima`), khong chay duoc bang env khac.
- Chi cell co JSON trong `<RESULTS>/tuning/` moi nap tuned params.
- Ket qua cu trong `results_covariate/` duoc sinh voi bo covariate 10 cot (co `dni`,
  `precipitation`, `quarter_hour`, `minute`, `is_weekend`): so sanh truc tiep voi lan
  chay moi la khong hop le, phai chay lai ca 4 model.
