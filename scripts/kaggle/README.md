# Chạy 4 model COV (TimeXer, TFT, TCN, DeepNPTS) trên Kaggle

Ba file, một việc mỗi file:

| file | việc |
|---|---|
| `01_setup.sh` | Miniforge + 2 conda env, probe torch CUDA, ghi env file, dry run |
| `02_run.py` | gọi `scripts/run_benchmark.py` và **thêm `--gpu`** (runner không tự truyền) |
| `03_tune.sh` | Optuna search cho 4 model (`scripts/tune_optuna.py`), resume theo study |
| `README.md` | các cell copy-paste |

`scripts/run_benchmark.py` là dispatcher: nó shell ra `conda run -n <env> python models/<script>.py`.
Vì vậy env phải là conda thật, đúng tên trong `command_for`:

| model | script | env |
|---|---|---|
| TimeXer | `models/model_timexer.py` | `neuralforecast` |
| TCN | `models/models_neuralforecast.py --models TCN` | `neuralforecast` |
| DeepNPTS | `models/models_neuralforecast.py --models DeepNPTS` | `neuralforecast` |
| TFT | `models/models_gluonts.py --models TemporalFusionTransformer` | `gluonts` (python 3.7 + mxnet cu112) |

## Chuẩn bị (làm 1 lần trên máy local)

Code phải nằm trên remote thì cell clone mới thấy: `scripts/kaggle/` và các sửa đổi TimeXer hiện **chưa commit**.

```bash
git add scripts/kaggle models/model_timexer.py models/models_neuralforecast.py \
        scripts/tune_optuna.py scripts/run_benchmark.py
git commit -m "kaggle runner + TimeXer historical covariates + TimeXer tuning"
git push
```

Cách khác không cần push: zip repo thành **Kaggle Dataset** (private) rồi trỏ `REPO_DIR=/kaggle/input/<dataset>/Forecast-Models-Evaluation`
(thư mục input là read-only, chỉ đọc code + `data/` — vừa đủ, vì `data/` đã tracked trong repo).

## Notebook settings

- Accelerator: **GPU T4 ×2** (P100 16GB không có bf16; T4 nhanh hơn cho 4 model này). Chỉ dùng 1 GPU — script nhận `--gpu 0`.
- Internet: **On** (bắt buộc để tải miniforge + conda packages).
- Persistence: `RESULTS` mặc định `/kaggle/working/results` — chỉ được giữ khi *Save Version*.

## Cell 1 — clone (hoặc bỏ qua nếu dùng Dataset)

```python
!git clone https://github.com/<user>/Forecast-Models-Evaluation /kaggle/working/Forecast-Models-Evaluation
```

## Cell 2 — setup (~15-25 phút, chỉ chạy 1 lần mỗi session)

```python
!cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/01_setup.sh
```

Kết thúc bằng 4 dòng `RUN ... --gpu 0` (dry run). Nếu thấy `FAIL: CPU-only torch` thì làm đúng dòng `pip install` nó in ra,
đừng cài lại torch từ conda.

Env conda nằm ở `/tmp/miniforge3` → **mất khi session kết thúc**. Muốn giữ qua các session (đổi lại: +5-8 GB vào output 20 GB):

```python
!cd /kaggle/working/Forecast-Models-Evaluation && CONDA_HOME=/kaggle/working/miniforge3 bash scripts/kaggle/01_setup.sh
```

## Cell 3 — kiểm tra trước khi tiêu GPU

```python
!source /kaggle/working/env.sh && cd $REPO_DIR && python scripts/kaggle/02_run.py --dry-run \
    --models TimeXer TFT TCN DeepNPTS --datasets germany --samplings 25 --level COV
```

Mỗi dòng lệnh phải kết thúc bằng `--gpu 0`; nếu không, GPU đang bị bỏ không.

## Cell 4 — tune (bỏ qua nếu chấp nhận paper defaults)

`run_benchmark.py` chỉ truyền `--params_json` khi tìm thấy `<RESULTS>/tuning/<Model>_<set>_<LEVEL>_<tag>_v2.json`
(`utils/tuning.py`). Không tune thì 4 model chạy paper defaults — vẫn hợp lệ, chỉ khác "tuned".

```python
!cd /kaggle/working/Forecast-Models-Evaluation && source /kaggle/working/env.sh && \
    MODELS="TimeXer TFT TCN DeepNPTS" TRIALS=30 HOURS=1.5 bash scripts/kaggle/03_tune.sh
```

- 4 model × 3 tuning set (`germany`, `zonnedael9`, `belgium_pv1`) × 30 trial. Study lưu trong `$RESULTS/tuning/optuna.db`,
  nên session sau chạy lại cùng lệnh là **cộng thêm trial**, không mất kết quả; study đã đủ `TRIALS` thì `SKIP`.
- TimeXer tune trên đúng 5 biến của cell báo cáo (Belgium thêm price/soc), không `static_*`, và dùng `TimeXerHist`
  → params tìm được khớp với lúc chạy thật.
- Xem trước không tốn GPU: `DRY=1 MODELS="TimeXer" TUNE_SETS="germany" bash scripts/kaggle/03_tune.sh`.
- Thứ tự đúng: tune ở `SAMPLING=100` (mặc định) → sweep ở mọi sampling sẽ dùng lại file 100% cho các mức khác.

## Cell 5 — chạy thật

```python
# một dataset, một sampling: 4 model × 10 run
!source /kaggle/working/env.sh && cd $REPO_DIR && python scripts/kaggle/02_run.py \
    --models TimeXer TFT TCN DeepNPTS --datasets germany --samplings 25 --level COV
```

Mở rộng dần (resume tự động — cell nào đủ 10 run thì bị SKIP, xem Cell 5):

```python
# thêm sampling
... --datasets germany --samplings 25 33.333333 50 100 --level COV
# thêm dataset
... --datasets germany london zonnedael belgium --level COV
```

Chạy nền để không giữ cell (session vẫn phải còn sống):

```python
!source /kaggle/working/env.sh && cd $REPO_DIR && nohup python -u scripts/kaggle/02_run.py \
    --models TimeXer TFT TCN DeepNPTS --datasets germany --level COV \
    > /kaggle/working/run_germany.log 2>&1 &
```

## Cell 6 — theo dõi

```python
!tail -n 20 /kaggle/working/run_germany.log
!find /kaggle/working/results -name model_metrics_summary.csv | wc -l   # số run xong
!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

## Chi phí

Đo trên repo này (CPU, germany 25%, 2 epoch): ~1 s cho TimeXer. Với 100 epoch, một run cỡ vài chục giây,
nên một cell `(dataset, sampling)` × 4 model × 10 run là **cỡ 30-90 phút** trên T4; toàn bộ 4 dataset × 4 sampling
là **vài ngày**, vượt xa một session 9-12 h → chia theo dataset, dựa vào SKIP để cộng dồn.

## Cảnh báo phải biết

- **Bộ covariate không đồng nhất.** `model_timexer.py` mặc định chỉ dùng 5 biến (hour, dayofweek, humidity, temperature,
  cloud_cover; Belgium thêm `day_ahead_price`, `soc`) và từ chối `static_*`. TCN/DeepNPTS/TFT **không có** flag
  `--covariates` nên chúng ăn nguyên block COV của loader (calendar + weather + price + soc + static).
  Muốn 4 model cùng một bộ biến thì phải thêm flag đó vào `models_neuralforecast.py` và `models_gluonts.py`.
- `--covariates` của runner chỉ an toàn với model có flag đó; truyền nó cho TCN/TFT sẽ làm script thoát mã 2.
- `runs_for()` cố định 10 run cho cả 4 model; cell thiếu run bị chạy lại **từ run 1** (không nối tiếp run dang dở).
- Chỉ cell nào có JSON trong `<RESULTS>/tuning/` mới chạy tuned params; TimeXer đã nằm trong `TUNED_MODELS` và
  `models/model_timexer.py` đã nhận `--params_json`. Chưa tune → dry run in `note: no tuned parameters for ...`.
- `SPACE_VERSION` vẫn là 2 (không bump) để các file tuned cũ của LightGBM/TCN/TFT/DeepNPTS còn dùng được; space của
  TimeXer là mới nên không đụng tên file cũ.
- `gluonts` env là python 3.7 + mxnet-cu112: build lâu nhất và dễ vỡ nhất; chạy TFT ở session riêng nếu setup quá lâu.
