# Gemma ASR Finetuning

## 檔案結構

- `data_prep.py`：載入音訊資料集、音訊採樣率轉換與文字正規化（V6）
- `model_setup.py`：負責初始化 Gemma 模型與設定 LoRA (Low-Rank Adaptation) 的微調參數，為訓練做好準備。
- `train.py`：微調訓練的主程式。整合前處理與模型，設定訓練的 hyper-parameters（例如 learning rate 等），並執行訓練流程。
- `benchmark_finetune.py`：負責對微調過後的模型（加載 LoRA adapter）進行評估（Benchmark）。
- `requirements.txt`：本專案所需的所有 Python 套件與版本依賴清單。

## 環境建置

```bash
conda create -n taipei-finetune python=3.10
conda activate taipei-finetune

pip install -r requirements.txt
```

## 執行步驟

### 1. 訓練模型 (Finetune)
執行以下指令開始進行模型的微調。所有的訓練超參數（Hyperparameters）皆已在 `train.py` 內設定完畢（目前學習率調整為 2e-4 等設定）：

```bash
python train.py
# or us nohup to run without interruption
nohup python -u train.py > train_run.log 2>&1 &
```
訓練完成後，LoRA 的權重（Adapter）將會預設儲存至 `./checkpoints/final_lora` 資料夾下。

### 2. 執行評估 (Benchmark)
執行 `benchmark_finetune.py`。
腳本會將我們訓練好的 LoRA 權重掛載到 Gemma 上，並在驗證集 / 測試集上進行推論與效能評估：

```bash
# set max_samples = 0 to evaluate all samples, --split for test or validation
python benchmark_finetune.py --peft_model ./checkpoints/final_lora --max_samples 0 --split test
```
評估完成後，結果會輸出至 `finetune_eval_results.jsonl`（或透過 `--output` 指定的檔案）供後續分析。
