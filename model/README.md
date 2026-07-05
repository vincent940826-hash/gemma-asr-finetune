# Gemma ASR Taiwanese

這是一個基於 `google/gemma-4-E4B-it` 進行 Fine-tune 的台語語音辨識 (ASR) 模型。

## 模型權重
模型權重 (LoRA Adapter) 已經託管在 Hugging Face，你可以直接透過 `peft` 下載並載入：
🔗 [LeeYuCheng/gemma-asr-taiwanese](https://huggingface.co/LeeYuCheng/gemma-asr-taiwanese)

## 快速開始 (Quick Start)

### 1. 安裝所需套件
```bash
pip install torch transformers peft librosa
```

### 2. 執行推論 (Inference)
參考 `inference.py`

```bash
python inference.py
```

### 3. 注意事項
- 程式碼預設預期音檔的取樣率為 `16kHz`，如果你的音檔不同，請先進行轉換或在 `librosa.load` 時指定正確的 `sr`。
- 本模型在載入時，針對 `audio_config.attention_invalid_logits_value` 有做特殊修正以避免 float16 overflow，細節請參考 `inference.py` 內的實作。

若打算自己寫推論腳本（不使用提供的 `inference.py`），請遵守以下設定，否則可能會遇到推論結果出現亂碼、幻覺，或是 CUDA 錯誤：

1. **Prompt 格式與 Chat Template**
   - 必須使用精準的指令字串：`"請將以下語音內容轉寫為繁體中文。"`
   - 必須使用 `processor.apply_chat_template()` 來將文字與音訊組合成模型預期的對話結構。
   - 呼叫 template 時，請帶入參數 `add_generation_prompt=True` 與 `enable_thinking=False`。

2. **Tokenizer Padding 設置**
   - 為支援 Batch Generation，請手動將 Tokenizer 設為向左填充：
     ```python
     processor.tokenizer.padding_side = 'left'
     processor.tokenizer.pad_token = processor.tokenizer.eos_token
     ```

3. **Processor 呼叫參數**
   - 將文字輸入給 Processor 時，請務必加上 `padding=True` 與 `add_special_tokens=False`（因為 chat template 已經加過 special tokens 了，重複加會破壞結構）。

4. **Float16 溢位 (Overflow) 修正**
   - 基底模型 `gemma-4-E4B-it` 在半精度 (float16) 運算下，`audio_tower` 的 attention 會發生溢位。載入模型後必須手動修正：
     ```python
     base_model.config.audio_config.attention_invalid_logits_value = -60000.0
     base_model.audio_tower.config.attention_invalid_logits_value = -60000.0
     ```

5. **運算精度 (Autocast)**
   - 呼叫 `model.generate()` 時，請使用 `with torch.autocast("cuda", dtype=torch.float16):` 包覆，以確保在 V100 等環境下運算穩定。

6. **音訊格式**
   - 預期音檔的取樣率為 `16kHz`，如果音檔格式不同，請在讀取時 (如 `librosa.load`) 指定 `sr=16000` 進行轉換。

---

## 💡 常見問題與疑難排解 (Troubleshooting)

### 載入模型時遇到 `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
如果你是在 Linux 伺服器 (特別是使用 Conda 環境) 上執行，並且在載入 LoRA Adapter 時遇到這個 CUDA 錯誤，這通常是因為 Conda 環境與系統底層的 NVIDIA 驅動程式函式庫 (CUDA Driver Library) 找不到彼此造成的。

**解決方法**：在執行 Python 腳本之前，手動指定 `LD_PRELOAD` 強制載入系統的 CUDA 函式庫：
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
python inference.py
```
*(請注意：此解法高度相依於你的作業系統環境。一般 Windows、Mac 或使用標準 Google Colab 的使用者**不需要**也不應該加入這個設定，否則反而會報錯找不到檔案。)*
