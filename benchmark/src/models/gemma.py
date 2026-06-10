import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from .base import BaseASRModel

class GemmaASRModel(BaseASRModel):
    def __init__(self, model_id: str = "google/gemma-4-E4B-it", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        
        print(f"\nInitializing {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # 設定 left padding，以利 batch generation
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

    def transcribe_batch(self, audio_arrays: list, sampling_rates: list) -> list[str]:
        # 建立 ASR Prompt
        batch_messages = []
        for audio_array in audio_arrays:
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Transcribe the following speech segment in Chinese into Chinese text.\n\nFollow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
                        },
                        {
                            "type": "audio", 
                            "audio": audio_array
                        },
                    ]
                }
            ])
            
        # 應用 Chat Template (關閉 thinking 模式以進行純 ASR 測試)
        text_prompts = self.processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 取得統一的 sampling rate (此專案已被 dataset loader cast 為 16000)
        sr = sampling_rates[0] if sampling_rates else 16000
        
        gemma_inputs = self.processor(
            text=text_prompts, 
            audio=audio_arrays, 
            sampling_rate=sr, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        prompt_len = gemma_inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            gemma_outputs = self.model.generate(
                **gemma_inputs, 
                max_new_tokens=256,
                temperature=1.0,
                top_p=0.95,
                top_k=64
            )
            
        predictions = []
        for i in range(len(audio_arrays)):
            # 擷取新生成的 token 並進行解碼
            pred_tokens = gemma_outputs[i][prompt_len:]
            pred_text = self.processor.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            
        return predictions
