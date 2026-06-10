from datasets import load_dataset, Audio
from .base import BaseDatasetLoader, register_dataset

@register_dataset("common_voice")
class CommonVoiceLoader(BaseDatasetLoader):
    DEFAULT_DATASET_ID = "voidful/common_voice_25_zh-TW"
    DEFAULT_SPLIT = "test"

    def __init__(self, dataset_id: str = None, split: str = None):
        super().__init__(dataset_id, split)

    def load(self, max_samples: int = None) -> list[dict]:
        print(f"Loading {self.dataset_id} (split: {self.split})...")
        dataset = load_dataset(self.dataset_id, split=self.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        if max_samples is not None:
            dataset = dataset.select(range(max_samples))
            
        formatted_data = []
        for sample in dataset:
            formatted_data.append({
                "audio": sample["audio"]["array"],
                "sampling_rate": sample["audio"]["sampling_rate"],
                "reference": sample["sentence"]
            })
        return formatted_data
