# Chay tren Kaggle - LightGBM vs 3 zero-shot checkpoint

| model | script | env |
| --- | --- | --- |
| LightGBM | `models/models_statsml.py --models LightGBM` | `statsml` |
| Chronos2 | `models/fmodel_chronos2.py` | `chronos` |
| Timestral | `models/fmodel_timestral.py` | `chronos` |
| TimesFM3 | `models/fmodel_timesfm3.py` | `timesfm3` |

Baseline hoc sau cua bai cu (DeepNPTS, TCN, TFT, TimeXer) da bi bo khoi luong nay.

## 1. Setup

    !git clone https://github.com/<user>/Forecast-Models-Evaluation /kaggle/working/repo
    !cd /kaggle/working/repo && bash scripts/kaggle/01_setup.sh

## 2. Tune LightGBM

    !cd /kaggle/working/repo && MODELS=LightGBM TRIALS=30 HOURS=1.5 bash scripts/kaggle/03_tune.sh

Ba tuning set: `germany`, `zonnedael9`, `belgium_pv1`; params duoc transfer sang cell bao cao
theo map trong `utils/tuning.py` (london <- zonnedael9, belgium <- belgium_pv1).

## 3. Chay benchmark

    !cd /kaggle/working/repo && python scripts/kaggle/02_run.py \
        --models LightGBM --datasets germany london zonnedael belgium --samplings 25 --level COV

    !cd /kaggle/working/repo && python scripts/kaggle/02_run.py \
        --models "Chronos2 Timestral TimesFM3" --samplings 25 --level COV

`--gpu` mac dinh 0; runner khong them flag do, wrapper nay them.

## 4. TimeSHAP

    !cd /kaggle/working/repo && MODELS="LightGBM Chronos2 Timestral TimesFM3" \
        bash scripts/kaggle/04_xai.sh

- `--players covariates` (mac dinh): 5 bien -> 32 coalition, dung cho so sanh cheo.
- `--players temporal`: them `hist_d1..hist_d4` -> 9 player, 512 coalition.
- LightGBM gio co lag/rolling cua y nen temporal player khong con bang 0 tuyet doi.

## Luu y

- LightGBM mac dinh chay 10 run (`--runs 1` neu muon khop cay ket qua cua bai).
- `models_statsml.py` can env `statsml` (`pmdarima`), khong chay duoc bang env khac.
- Chi cell co JSON trong `<RESULTS>/tuning/` moi nap tuned params.
