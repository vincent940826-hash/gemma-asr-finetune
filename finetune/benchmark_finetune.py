import sys
import os

# 1. 將根目錄（taipei/）與 benchmark/ 加入 Python Path，這樣才能使用 benchmark/src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../benchmark")))
sys.modules['torchcodec'] = None  # 繞過不相容套件

import argparse
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import PeftModel

# 導入 benchmark 的核心庫
from benchmark.src.datasets.base import DATASET_REGISTRY
# 必須手動導入各個 dataset 模組以觸發註冊裝飾器
import benchmark.src.datasets.common_voice
import benchmark.src.datasets.ascend
import benchmark.src.datasets.formosa_speech

from benchmark.src.models.gemma import GemmaASRModel
from benchmark.src.evaluator import ASREvaluator

# 2. 繼承並定義一個能載入 LoRA 的 Gemma Model 類別
class GemmaFinetunedASRModel(GemmaASRModel):
    def __init__(self, base_model_id: str = "google/gemma-4-E4B-it", peft_model_id: str = "./checkpoints/final_lora", device: str = "cuda"):
        self.model_id = base_model_id
        self.peft_model_id = peft_model_id
        self.device = device
        
        print(f"\nInitializing base model: {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
        # 使用與您訓練時相同的 4-bit 量化配置載入 base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base_model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, 
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # 載入您微調好的 LoRA 權重
        print(f"Loading finetuned LoRA weights from: {self.peft_model_id}...")
        self.model = PeftModel.from_pretrained(base_model, self.peft_model_id)
        self.model.eval()

    def transcribe_batch(self, audio_arrays: list, sampling_rates: list) -> list[str]:
        prompt_text = (
            "Transcribe the following speech segment in Traditional Chinese into Traditional Chinese text. "
            "Follow these specific instructions for formatting the answer:\n"
            "* Only output the transcription, with no newlines.\n"
            "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
        )
        
        batch_messages = []
        for audio_array in audio_arrays:
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "audio", "audio": audio_array},
                    ]
                }
            ])
            
        text_prompts = self.processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        sr = sampling_rates[0] if sampling_rates else 16000
        
        gemma_inputs = self.processor(
            text=text_prompts,
            audio=audio_arrays,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        input_len = gemma_inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            gemma_outputs = self.model.generate(
                **gemma_inputs,
                max_new_tokens=256,
            )
            
        predictions = []
        for i in range(len(audio_arrays)):
            pred_tokens = gemma_outputs[i][input_len:]
            pred_text = self.processor.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            
        return predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate Finetuned Gemma using Benchmark framework")
    parser.add_argument("--dataset_name", type=str, choices=list(DATASET_REGISTRY.keys()), default="common_voice")
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation") # 預設使用驗證集來評估微調效果
    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (batch_size=1 is recommended for multimodal LoRA)")
    parser.add_argument("--output", type=str, default="finetune_eval_results.jsonl", help="Output file path")
    parser.add_argument("--peft_model", type=str, default="./checkpoints/final_lora", help="Path to your LoRA adapter")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 載入資料集
    loader_cls = DATASET_REGISTRY[args.dataset_name]
    dataset_id = args.dataset_id or loader_cls.DEFAULT_DATASET_ID
    split = args.split or loader_cls.DEFAULT_SPLIT
    
    loader = loader_cls(dataset_id=dataset_id, split=split)
    max_samples = args.max_samples if args.max_samples > 0 else None
    dataset = loader.load(max_samples=max_samples)
    print(f"Loaded {len(dataset)} samples from {args.dataset_name} ({split}) for evaluation.")

    # 2. 建立微調模型 (這裡只測試 Finetuned Gemma，我們給 breeze 傳入一個 Mock 或是空物件以節省顯存)
    finetuned_gemma = GemmaFinetunedASRModel(
        base_model_id="google/gemma-4-E4B-it",
        peft_model_id=args.peft_model,
        device=device
    )

    # 為了套用原本的 ASREvaluator，我們需要建立一個 Dummy Breeze Model 避免程式噴錯
    class DummyBreezeModel:
        def transcribe_batch(self, audio_arrays, sampling_rates):
            return ["(Skipped Breeze)"] * len(audio_arrays)

    # 3. 執行評估
    evaluator = ASREvaluator(output_path=args.output)
    evaluator.evaluate(
        breeze_model=DummyBreezeModel(),
        gemma_model=finetuned_gemma,
        dataset=dataset,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()