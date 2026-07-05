import torch
import librosa
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

# 你的 Hugging Face 模型 ID
PEFT_MODEL_ID = "LeeYuCheng/gemma-asr-taiwanese"
BASE_MODEL_ID = "google/gemma-4-E4B-it"

def main():
    print(f"Loading processor from {BASE_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    
    # [ALIGNMENT]: 設定 left padding，以利 batch generation 且確保行為與 Benchmark 一致
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Loading base model {BASE_MODEL_ID} on GPU...")
    # [ALIGNMENT]: 這裡維持使用 torch.float16 與 device_map="cuda" 
    # (benchmark_finetune.py 也是 float16，並依據 device 決定 device_map)
    base_model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda"
    )
    
    # 修正 float16 overflow 問題 (與訓練及 Benchmark 時相同)
    if hasattr(base_model.config, "audio_config") and base_model.config.audio_config is not None:
        base_model.config.audio_config.attention_invalid_logits_value = -60000.0
    if hasattr(base_model, "audio_tower") and getattr(base_model, "audio_tower") is not None:
        if hasattr(base_model.audio_tower, "config") and base_model.audio_tower.config is not None:
            base_model.audio_tower.config.attention_invalid_logits_value = -60000.0

    print(f"Loading LoRA adapter from {PEFT_MODEL_ID}...")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    model.eval()

    # --- 以下為推論 (Inference) 測試範例 ---
    audio_path = "test_audio.wav"  # 請替換成你實際的音檔路徑
    try:
        print(f"Loading audio file: {audio_path}")
        # 讀取音檔 (通常 ASR 模型預期 16kHz 取樣率)
        audio_input, sr = librosa.load(audio_path, sr=16000) 
        
        # [ALIGNMENT]: 使用與 Benchmark 完全相同的 prompt text
        prompt_text = "請將以下語音內容轉寫為繁體中文。"
        
        # [ALIGNMENT]: 使用 apply_chat_template 來構建輸入結構
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio", "audio": audio_input},
                ]
            }
        ]
        
        # [ALIGNMENT]: 套用 chat template 且不啟用 thinking
        text_prompt = processor.apply_chat_template(
            [messages],  # batch_messages 格式
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )[0]
        
        # [ALIGNMENT]: 確保 padding=True 與 add_special_tokens=False
        inputs = processor(
            text=[text_prompt],
            audio=[audio_input],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to("cuda") # Note: 不轉換 audio tensor 也是 benchmark 預設行為
        
        # [ALIGNMENT]: 取得 prompt 的 token 長度，以利後續 slicing
        input_len = inputs["input_ids"].shape[1]
        
        print("Generating transcription...")
        with torch.no_grad():
            # [ALIGNMENT]: 使用 torch.autocast 確保 V100 等環境運算正確
            with torch.autocast("cuda", dtype=torch.float16):
                # [ALIGNMENT]: 設定 max_new_tokens=256 與 do_sample=False
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    do_sample=False
                )
        
        # [ALIGNMENT]: 只擷取新生成的 token 進行解碼 (切掉 input prompt)
        pred_tokens = outputs[0][input_len:]
        transcription = processor.decode(pred_tokens, skip_special_tokens=True)
        
        print("\n=== 辨識結果 ===")
        print(transcription)
        
    except FileNotFoundError:
        print(f"\n[提示] 請將 '{audio_path}' 替換成真實的音檔路徑來測試 inference。")

if __name__ == "__main__":
    main()
