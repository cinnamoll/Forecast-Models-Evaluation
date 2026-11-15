Quy trình, theo đúng thứ tự để không phải chạy lại. Tất cả lệnh chạy trong /home/cinnamoll/Code/UntitledFolder/forecast_cov_timeshap (nhánh feat/covariates-timeshap).

Bước 0 — kiểm tra trước khi chạy

cd /home/cinnamoll/Code/UntitledFolder/forecast_cov_timeshap
git status --porcelain            # code chưa commit, data/exogenous + plan là untracked
python3 scripts/run_benchmark.py --dry-run --datasets london --models LightGBM TCN

--gpu -1 = CPU, --gpu 0|1 = GPU. Mỗi mô hình chạy trong env riêng: statsml (LightGBM), neuralforecast (DeepNPTS/PatchTST/TCN), gluonts (TFT), chronos (Chronos2/Timestral).

Bước 1 — dựng 4 env, rồi smoke test 1 ô trước khi chạy cả bảng

conda env create -f requirements/env_statsml.yml
conda env create -f requirements/env_neuralforecast.yml
conda env create -f requirements/env_gluonts.yml        # python 3.7 + mxnet-cu112, dựng trước vì dễ hỏng nhất
conda env create -f requirements/env_chronos.yml        # có timestral cài từ git

conda run -n statsml        python models/models_statsml.py        --dataset london --sampling_rates 100 --models LightGBM --runs 1 --exog_level COV
conda run -n neuralforecast python models/models_neuralforecast.py --dataset london --sampling_rates 25 --models TCN --runs 1 --epochs 20 --exog_level COV
conda run -n gluonts        python models/models_gluonts.py         --dataset london --sampling_rates 100 --models TemporalFusionTransformer --runs 1 --epochs 5 --exog_level COV
conda run -n chronos        python models/fmodel_chronos2.py        --dataset london --sampling_rates 100 --runs 1 --exog_level COV
conda run -n chronos        python models/fmodel_timestral.py       --dataset london --sampling_rates 100 --runs 1 --exog_level COV

Xem results/results_london/<Model>/Sampling_100/COV/Run_1/training_log.txt: phải thấy dòng "covariate lists" (NF), "covariates: … known-future, … past-only, … static" (TFT), "lagged N past-only covariate(s)" (LightGBM). Nếu không thấy → covariate không vào mô hình, đừng chạy tiếp.

Bước 2 — L0 trước (mốc so với bảng gốc)

python scripts/run_benchmark.py --level L0                 # 688 run: 43 run/ô × 16 ô

So trung vị MAE của LightGBM và 4 mô hình học sâu với results/ cũ (cùng 4 bộ, 4 mức sampling). Lệch >2% thì dừng, tìm nguyên nhân trước khi làm COV.

Bước 3 — tune trên LCL + WPuQ (trước khi chạy COV, để COV dùng luôn tham số)

conda run -n statsml        python scripts/tune_optuna.py --model LightGBM --dataset LCL  --level COV --sampling 100
conda run -n neuralforecast python scripts/tune_optuna.py --model DeepNPTS --dataset WPuQ --level COV --sampling 100
conda run -n neuralforecast python scripts/tune_optuna.py --model PatchTST --dataset LCL  --level COV --sampling 100
conda run -n neuralforecast python scripts/tune_optuna.py --model TCN      --dataset WPuQ --level COV --sampling 100
conda run -n gluonts        python scripts/tune_optuna.py --model TFT      --dataset LCL  --level COV --sampling 100

10 study, mỗi study ≤30 trial / ≤3 h, lưu ở results/tuning/optuna.db nên session chết thì chạy lại đúng lệnh là nối tiếp. Kết quả: results/tuning/<Model>_<dataset>_COV_100.json + cảnh báo nếu tham số nằm ở biên.

Bước 4 — chuyển tham số sang Belgium/Zonnedael (và sang các mức sampling còn lại)

for m in LightGBM DeepNPTS PatchTST TCN TFT; do
  src=$(ls results/tuning/${m}_london_COV_100.json results/tuning/${m}_germany_COV_100.json 2>/dev/null | head -1)
  for ds in belgium zonnedael; do
    for tag in 25 33 50 100; do cp "$src" results/tuning/${m}_${ds}COV${tag}.json; done
  done
done

Đây là chuyển giao có chủ đích (plan §5): ghi rõ trong bài là "tham số chuyển giao", và nếu file JSON có boundary_hits khác rỗng thì phải nới không gian hoặc tune riêng bộ đó.

Bước 5 — chạy COV (bảng chính) cho 4 bộ

python scripts/run_benchmark.py --level COV --datasets belgium    # 172 run
python scripts/run_benchmark.py --level COV --datasets germany
python scripts/run_benchmark.py --level COV --datasets london
python scripts/run_benchmark.py --level COV --datasets zonnedael

Chạy từng bộ để dễ resume: ô nào đã có đủ run thì in SKIP. Script tự thêm --params_json khi tìm thấy file tune tương ứng. --force chỉ dùng khi muốn chạy lại ô đã xong.

Bước 6 — XAI TimeSHAP

xem bộ player trước, không train
python scripts/xai_timeshap.py --dataset london --series london_load --dry-run

conda run -n statsml python scripts/xai_timeshap.py --dataset london   --series london_load              --models LightGBM
conda run -n gluonts python scripts/xai_timeshap.py --dataset belgium  --series pv_house_1               --models TFT
conda run -n chronos python scripts/xai_timeshap.py --dataset germany  --series germany_load             --models Chronos2 Timestral
3 mô hình NeuralForecast: cần --allow-refit (refit mỗi coalition, chỉ các khối covariate)
conda run -n neuralforecast python scripts/xai_timeshap.py --dataset zonnedael --series zonnedael_customer_8 --models DeepNPTS PatchTST TCN --allow-refit

13 series (pv_house_1..4, bess_house_1..4, germany_load, london_load, zonnedael_customer_8/9/43). Mỗi ô ghi group_shares.csv, shap_values.csv, base_value.csv, coalitions.json, temporal_pruning.csv dưới results/xai_timeshap/. Script tự dừng nếu additivity > 1e-6 — lỗi ở đó là lỗi thật, không bỏ qua.

Bước 7 — DM test (giữ nguyên file gốc)
utils/dm_test.py tìm file theo Sampling_100/Run_1/*.csv, nên cây COV (có thêm tầng COV/) nó không thấy. Muốn chạy DM trên kết quả COV mà không sửa file test, dựng cây phẳng bằng symlink:

mkdir -p /tmp/dm_cov
for ds in belgium germany london zonnedael; do
  for d in results/results_$ds/*/Sampling_100/COV/Run_1; do
    [ -d "$d" ] || continue
    model=$(basename $(dirname $(dirname $(dirname "$d"))))
    mkdir -p /tmp/dm_cov/results_$ds/$model/Sampling_100
    ln -sfn "$(readlink -f "$d")" /tmp/dm_cov/results_$ds/$model/Sampling_100/Run_1
  done
done
python utils/dm_test.py --datasets belgium germany london zonnedael --base_dir /tmp/dm_cov --output_dir results/dm_test_cov


Kiểm tra sau mỗi bước
- results/results_<ds>/<Model>/Sampling_<tag>/COV/Run_n/model_metrics_summary.csv phải có đủ 8 dòng (Belgium) hoặc 1/3 dòng (bộ khác).
- exogenous_support.json / dòng log covariate: TFT phải có past_dynamic_features 7–8 cột; Chronos2/Timestral log rõ static bị bỏ.
- XAI: coalitions.json ghi n_players, coalitions, additivity_residual; group_shares.csv tổng = 1.

Một điểm cần chốt trước khi chạy dài: tune hiện làm ở --sampling 100 rồi chuyển sang 25/33/50. Nếu bạn muốn mỗi mức sampling có tham số riêng thì chạy thêm --sampling 25 33 50 (gấp 4 lần ngân sách tune), còn không thì bước 4 ở trên là lựa chọn "chuyển giao", phải ghi vào bài.
