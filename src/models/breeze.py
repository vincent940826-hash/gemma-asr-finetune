import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutomaticSpeechRecognitionPipeline
from .base import BaseASRModel

class BreezeASRModel(BaseASRModel):
    def __init__(self, model_id: str = "MediaTek-Research/Breeze-ASR-25", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        
        print(f"\nInitializing {self.model_id}...")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        
        self.pipeline = AutomaticSpeechRecognitionPipeline(
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device=0 if self.device == "cuda" else -1
        )

    def transcribe_batch(self, audio_arrays: list, sampling_rates: list) -> list[str]:
        inputs = [{"raw": arr, "sampling_rate": sr} for arr, sr in zip(audio_arrays, sampling_rates)]
        # For batching, pipeline takes generator or list.
        # We specify batch_size=len(audio_arrays)
        outputs = self.pipeline(inputs, batch_size=len(audio_arrays), return_timestamps=False)
        if isinstance(outputs, dict):
            return [outputs["text"]]
        return [out["text"] for out in outputs]
