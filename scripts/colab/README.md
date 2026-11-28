# Chạy trên Colab: setup → tune → run COV (transfer learning) → TimeSHAP

| file | việc | env |
|---|---|---|
| `01_setup.sh <env>` | Miniforge vào `/content`, tạo **một** env, probe torch CUDA, ghi `/content/env.sh` | — |
| `02_tune.sh` | Optuna search — bước 1 của transfer learning | `neuralforecast` / `gluonts` theo model |
| `03_run.sh` | Chạy cell COV trên params đã tune, tự thêm `--gpu` | như trên |
| `04_xai_neuralforecast.sh` | TimeSHAP cho TimeXer, TCN, DeepNPTS | `neuralforecast` |
| `05_xai_gluonts.sh` | TimeSHAP cho TFT | `gluonts` (python 3.7 + mxnet cu112) |

Roster thật nằm ở `scripts/kaggle/02_run.py` (run), `scripts/kaggle/03_tune.sh` (tune),
`scripts/kaggle/04_xai.sh` (xai). Script Colab chỉ ghim path `/content`, results trên Drive,
danh sách model và **kiểm tra env trước khi chạy** (thiếu env thì in lệnh setup rồi thoát 2).

## Cell 0 — runtime

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!df -h /content | tail -1      # env neuralforecast ~5 GB
```

Runtime → Change runtime type → **T4 GPU**.

## Cell 1 — code + Drive

```python
!git clone https://github.com/<user>/Forecast-Models-Evaluation /content/Forecast-Models-Evaluation
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/forecast-results
```

## Cell 2 — setup env `neuralforecast`

```python
!cd /content/Forecast-Models-Evaluation && bash scripts/colab/01_setup.sh neuralforecast
```

- Cài **Miniforge** (conda-forge), không phải Miniconda: `repo.anaconda.com` chặn `conda env create`
  khi chưa accept Terms of Service (`CondaToSNonInteractiveError`). Script cũng `unset PYTHONPATH`
  (Colab export biến này, conda cảnh báo và có thể lẫn package).
- Đã có `/content/miniconda3` từ lần chạy trước mà vẫn lỗi ToS: `!rm -rf /content/miniconda3` rồi chạy lại cell.
- In `FAIL: CPU-only torch` thì chạy đúng dòng `pip install` nó đưa ra.

## Cell 3 — tune: mỗi model một cell (mỗi cell = 3 study)

Một study ≈ `HOURS` (1.2 h ở đây), nên **một model = 3 bộ tune = ~3.6 h**; gộp 3 model vào một cell là
9 study ≈ 11 h, sát giới hạn session. Vì vậy mỗi model một cell, và chạy TFT ở session `gluonts`.

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TimeXer TRIALS=30 HOURS=1.2 bash scripts/colab/02_tune.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TCN TRIALS=30 HOURS=1.2 bash scripts/colab/02_tune.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=DeepNPTS TRIALS=30 HOURS=1.2 bash scripts/colab/02_tune.sh
```

```python
# session gluonts
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT TRIALS=30 HOURS=1.2 bash scripts/colab/02_tune.sh
```

Mỗi cell in 3 dòng `TUNE  <Model>/<germany|zonnedael9|belgium_pv1>`; tổng cộng **4 model × 3 bộ = 12 study**.
Luôn để `SAMPLING=100`: `utils/tuning.py` dùng file của đúng sampling nếu có, không thì fallback về 100 %,
nên tune ở 100 là dùng được cho cả 4 mức. `DRY=1` để in lệnh trước khi tiêu GPU.

`scripts/tune_optuna.py` tune trên **3 series** (`germany`, `zonnedael9`, `belgium_pv1`); các cell khác
mượn params theo map trong `utils/tuning.py` (london ← zonnedael9, belgium ← belgium_pv1, …) — đó chính là
transfer learning của repo. JSON ghi vào `$RESULTS/tuning/` (trên Drive nên session sau vẫn còn), study nằm
trong `optuna.db` nên chạy lại là **cộng thêm trial**, đủ `TRIALS` thì `SKIP`. `DRY=1` để in lệnh trước.

## Cell 4 — run COV: theo từng sampling rate

Lưới thật: 3 model × 4 dataset × 4 sampling = **48 cell**, mỗi cell 10 run. Ước lượng (đo 1 lần fit
= 34-53 s ở london/100 %, và 1 run ở 25 % ngắn hơn ~4 lần): ~3 phút/cell ở 25 % (12 cell ≈ 40 phút),
~10 phút/cell ở 100 % (12 cell ≈ 2 h) — nên 25/33/50 chạy gộp cả 4 dataset, còn 100 % chia theo dataset.

```python
# 25 %
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="germany london zonnedael belgium" \
    SAMPLINGS="25" bash scripts/colab/03_run.sh
```

```python
# 33.33 %
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="germany london zonnedael belgium" \
    SAMPLINGS="33.333333" bash scripts/colab/03_run.sh
```

```python
# 50 %
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="germany london zonnedael belgium" \
    SAMPLINGS="50" bash scripts/colab/03_run.sh
```

```python
# 100 % — mỗi dataset một cell
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="germany" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="london" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="zonnedael" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS="TimeXer TCN DeepNPTS" DATASETS="belgium" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

Muốn tách hẳn từng model: thay `MODELS="TimeXer TCN DeepNPTS"` bằng `MODELS=TimeXer` (rồi `TCN`,
`DeepNPTS`) — lưới thành 16 cell/model. Cell đã đủ 10 run thì `SKIP`, nên chạy lại sau khi Colab ngắt
chỉ tốn cell đang dở; `FORCE=1` chỉ dùng khi muốn ghi đè.

TFT (session `gluonts`) dùng đúng 4 mức đó, mỗi mức một cell:

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="germany london zonnedael belgium" SAMPLINGS="25" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="germany london zonnedael belgium" SAMPLINGS="33.333333" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="germany london zonnedael belgium" SAMPLINGS="50" bash scripts/colab/03_run.sh
```

```python
# 100 %: TFT nặng nhất, chạy từng dataset
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="germany" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="london" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="zonnedael" SAMPLINGS="100" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="belgium" SAMPLINGS="100" bash scripts/colab/03_run.sh
```


## Cell 5 — TimeSHAP (từng model)

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODEL=TimeXer DATASETS="germany london" bash scripts/colab/04_xai_neuralforecast.sh
```

`MODEL=TCN`, `MODEL=DeepNPTS` tương tự; bỏ `MODEL=` để chạy cả ba. 5 player biến → 32 coalition,
residual = 0; thời gian nằm ở một lần fit mỗi cell.

## Cell 6 — session riêng cho TFT (env `gluonts`)

```python
!cd /content/Forecast-Models-Evaluation && bash scripts/colab/01_setup.sh gluonts
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT TRIALS=30 bash scripts/colab/02_tune.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODELS=TFT DATASETS="germany london" SAMPLINGS="25" bash scripts/colab/03_run.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    DATASETS="germany london" bash scripts/colab/05_xai_gluonts.sh
```

## Cell 7 — theo dõi

```python
!tail -n 20 /content/drive/MyDrive/forecast-results/xai_timeshap/logs/*.log | tail -40
!find /content/drive/MyDrive/forecast-results -name model_metrics_summary.csv | wc -l
!cat /content/drive/MyDrive/forecast-results/xai_timeshap/london/london_load_COV_100/group_shares_all_models.csv
```

## Cảnh báo

- Env trong `/content` **mất khi session kết thúc** → chạy lại Cell 2 mỗi session (8-20 phút). Muốn giữ:
  `CONDA_HOME=/content/drive/MyDrive/miniconda3 bash scripts/colab/01_setup.sh neuralforecast`.
- `RESULTS` phải nằm trên Drive; nếu không, `tuning/*.json` mất cùng session và bước transfer learning
  ở lần sau lại quay về paper defaults.
- Mỗi cell tune là 3 study × `HOURS` ≈ 3.6 h với `HOURS=1.2`; 4 model là 12 study ≈ 14 h → chạy mỗi model
  một cell (Cell 3) chứ đừng gộp, và đừng tăng `HOURS` quá 3 trong một cell.
- `--epochs` mặc định 100 (khớp benchmark); `EPOCHS=10` chỉ để xem nhanh.
- `group_shares_all_models.csv` chỉ gộp các model chạy chung một lệnh; chạy riêng từng model thì đọc
  `group_shares.csv` / `player_shares.csv` trong thư mục của model đó.
- TFT cần `mxnet-cu112`: nếu lỗi CUDA khi train, chạy `GPU=-1` (CPU) rồi ghi rõ trong phụ lục.
