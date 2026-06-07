import sys
sys.modules['torchcodec'] = None
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    AutomaticSpeechRecognitionPipeline,
    AutoProcessor, 
    AutoModelForMultimodalLM
)
from jiwer import cer
import re
from opencc import OpenCC

# --- 參數設定 ---
# 測試樣本數：先用 10 筆測試腳本，若要跑全部 5,119 筆，請設為 None
MAX_SAMPLES = None
MODEL_BREEZE = "MediaTek-Research/Breeze-ASR-25"
MODEL_GEMMA = "google/gemma-4-E4B-it"
DATASET_ID = "voidful/common_voice_25_zh-TW"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 繁簡轉換器（確保標籤和預測結果都是正體中文，避免因繁簡不一影響 CER）
cc = OpenCC('s2t')

def clean_text(text):
    """基本文字清理：轉正體、移除標點符號、轉小寫、去除多餘空格"""
    if not text:
        return ""
    text = cc.convert(text)
    # 移除常見標點符號
    text = re.sub(r'[，。！？：；「」『』、（）—─""\'’.]', '', text)
    # 轉小寫並去除前後空格
    return text.strip().lower()

# --- 1. 載入資料集並統一重採樣至 16kHz ---
print("Loading dataset...")
# 載入官方 test split
dataset = load_dataset(DATASET_ID, split="test") 
# 強制將音訊欄位重採樣至 16,000 Hz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

if MAX_SAMPLES is not None:
    dataset = dataset.select(range(MAX_SAMPLES))

# --- 2. 初始化 Breeze-ASR-25 ---
print(f"\nInitializing {MODEL_BREEZE}...")
breeze_processor = WhisperProcessor.from_pretrained(MODEL_BREEZE)
breeze_model = WhisperForConditionalGeneration.from_pretrained(MODEL_BREEZE).to(device).eval()
breeze_pipeline = AutomaticSpeechRecognitionPipeline(
    model=breeze_model,
    tokenizer=breeze_processor.tokenizer,
    feature_extractor=breeze_processor.feature_extractor,
    chunk_length_s=0
)

# --- 3. 初始化 Gemma-4-E4B ---
print(f"Initializing {MODEL_GEMMA}...")
gemma_processor = AutoProcessor.from_pretrained(MODEL_GEMMA)
gemma_model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_GEMMA, 
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

# --- 4. 開始推論與評估 ---
references = []
breeze_predictions = []
gemma_predictions = []

print("\nStarting Benchmark...")
for idx, sample in enumerate(dataset):
    # 取得音訊陣列與標準文本
    audio_array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    ref_text = clean_text(sample["sentence"])
    references.append(ref_text)
    
    # --- Breeze 推論 ---
    breeze_out = breeze_pipeline(audio_array, return_timestamps=False)
    breeze_pred = clean_text(breeze_out["text"])
    breeze_predictions.append(breeze_pred)
    
    # --- Gemma 推論 ---
    # 依照 Gemma 4 官方說明，音訊應放在提示詞之後，並使用專屬 ASR Prompt 格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Transcribe the following speech segment in Chinese into Chinese text.\n\nFollow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
                },
                {
                    "type": "audio", 
                    "audio": audio_array  # 直接傳入 numpy array，processor 會處理採樣率
                },
            ]
        }
    ]
    
    # 應用 Chat Template (關閉 thinking 模式以進行純 ASR 測試)
    text_prompt = gemma_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    gemma_inputs = gemma_processor(
        text=text_prompt, 
        audio=audio_array, 
        sampling_rate=sr, 
        return_tensors="pt"
    ).to(device)
    
    input_len = gemma_inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        # 套用官方建議的 sampling 參數
        gemma_outputs = gemma_model.generate(
            **gemma_inputs, 
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64
        )
    
    gemma_response = gemma_processor.decode(gemma_outputs[0][input_len:], skip_special_tokens=True)
    gemma_pred = clean_text(gemma_response)
    gemma_predictions.append(gemma_pred)
    
    # 印出當前進度與即時對比
    print(f"[{idx+1}/{len(dataset)}]")
    print(f"  REF: {ref_text}")
    print(f"  BRZ: {breeze_pred}")
    print(f"  GMA: {gemma_pred}")
    print("-" * 50)

# --- 5. 計算整體 CER ---
# jiwer 的 cer 函數會將字串當作字元序列計算，適合中文 CER
breeze_cer = cer(references, breeze_predictions)
gemma_cer = cer(references, gemma_predictions)

print("\n" + "="*20 + " FINAL RESULTS " + "="*20)
print(f"Breeze-ASR-25 Overall CER : {breeze_cer * 100:.2f}%")
print(f"Gemma-4-E4B    Overall CER : {gemma_cer * 100:.2f}%")
print("="*55)