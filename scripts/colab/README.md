# Chạy TimeSHAP (5 model) trên Colab — tách theo từng model và từng env

| file | việc |
|---|---|
| `01_setup.sh <env>` | Miniconda vào `/content`, tạo **một** conda env, probe torch CUDA, ghi `/content/env.sh`, dry run |
| `02_xai_neuralforecast.sh` | TimeXer, TCN, DeepNPTS (env `neuralforecast`) — `MODEL=` để chạy một model |
| `03_xai_gluonts.sh` | TFT (env `gluonts`, python 3.7 + mxnet cu112) — session riêng |

Roster thật nằm ở `scripts/kaggle/04_xai.sh` (dataset → series, resume theo `(dataset, series, model)`,
`--per-variable --covariate-only`, `--gpu 0`); hai script Colab chỉ ghim path + env + danh sách model,
nên chỉ có một chỗ cài đặt vòng lặp.

## Cell 0 — kiểm runtime

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!df -h /content | tail -1        # env neuralforecast ~5 GB, gluonts ~2 GB
```

Runtime → Change runtime type → **T4 GPU**. Một session free ~không quá 12 h và tự ngắt khi idle.

## Cell 1 — code + Drive

```python
!git clone https://github.com/<user>/Forecast-Models-Evaluation /content/Forecast-Models-Evaluation
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/forecast-results
```

Code phải đã push: `scripts/colab/`, `scripts/kaggle/04_xai.sh`, `scripts/xai_timeshap.py`,
`utils/xai_predictors.py`, `utils/covariates.py`.

## Cell 2 — setup env `neuralforecast` (~10-20 phút)

```python
!cd /content/Forecast-Models-Evaluation && bash scripts/colab/01_setup.sh neuralforecast
```

Kết thúc bằng 6 dòng `XAI ... --gpu 0` (dry run của 3 model × 2 dataset). Nếu in `FAIL: CPU-only torch`
thì chạy đúng dòng `pip install` nó đưa ra.

## Cell 3 — dry run một cell (không tốn GPU)

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    DRY=0 XAI_DRY_CHECK=1 DATASETS="london" MODEL=TimeXer bash scripts/colab/02_xai_neuralforecast.sh
```

In ra player set + số coalition: 5 player biến → **32 coalition**, horizon và context như cell thật.

## Cell 4 — chạy từng model (mỗi model một cell, resume độc lập)

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODEL=TimeXer DATASETS="germany london" bash scripts/colab/02_xai_neuralforecast.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODEL=TCN DATASETS="germany london" bash scripts/colab/02_xai_neuralforecast.sh
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    MODEL=DeepNPTS DATASETS="germany london" bash scripts/colab/02_xai_neuralforecast.sh
```

Một lệnh cho cả ba (cùng env, tuần tự): bỏ `MODEL=`, chạy `bash scripts/colab/02_xai_neuralforecast.sh`.
Model nào đã có `coalitions.json` thì `SKIP`, nên chạy lại sau khi Colab ngắt chỉ tốn cell đang dở.

## Cell 5 — TFT, session riêng (env `gluonts`)

```python
!cd /content/Forecast-Models-Evaluation && bash scripts/colab/01_setup.sh gluonts
```

```python
!cd /content/Forecast-Models-Evaluation && source /content/env.sh && \
    DATASETS="germany london" bash scripts/colab/03_xai_gluonts.sh
```

## Cell 6 — theo dõi / lấy kết quả

```python
!tail -n 20 /content/drive/MyDrive/forecast-results/xai_timeshap/logs/*.log | tail -40
!find /content/drive/MyDrive/forecast-results/xai_timeshap -name coalitions.json | wc -l
!cat /content/drive/MyDrive/forecast-results/xai_timeshap/london/london_load_COV_100/group_shares_all_models.csv
```

`group_shares_all_models.csv` chỉ gộp các model **chạy chung một lệnh**; chạy từng model riêng thì mỗi model
có `group_shares.csv` + `player_shares.csv` + `shap_values.csv` trong thư mục của nó — gộp lại bằng cách đọc
các file đó (chúng nằm trên Drive nên không mất khi runtime kết thúc).

## Cảnh báo

- Env nằm trong `/content` → **mất khi session kết thúc**, phải chạy lại Cell 2 (~10-20 phút). Muốn giữ:
  `CONDA_HOME=/content/drive/MyDrive/miniconda3 bash scripts/colab/01_setup.sh neuralforecast` (chậm hơn vì Drive).
- `--epochs` mặc định 100. Để so sánh công bằng với benchmark thì giữ 100; muốn xem nhanh thì `EPOCHS=10`.
- Mỗi game là 32 coalition nên rất rẻ; chi phí thật nằm ở **fit một lần per (model, dataset, series)** —
  TimeXer/TCN/DeepNPTS dùng `input_size = 2*horizon`, nên sampling 100 sẽ chậm hơn 25 đáng kể.
- TFT cần `mxnet-cu112`: nếu session Colab báo lỗi CUDA khi train, chạy TFT bằng `GPU=-1` (CPU) cho xong game
  rồi ghi rõ trong phụ lục là cell đó chạy CPU.
