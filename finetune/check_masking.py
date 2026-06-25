import torch
from transformers import AutoProcessor
from data_prep import ASRDataCollator
import numpy as np

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
collator = ASRDataCollator(processor)

features = [
    {
        "audio_array": np.zeros(16000), # 1 sec audio
        "prompt_text": "Transcribe the following speech segment in Traditional Chinese into Traditional Chinese text.",
        "target_text": "這是一個測試句子。"
    }
]

batch = collator(features)
input_ids = batch["input_ids"][0]
labels = batch["labels"][0]

print("Input IDs length:", len(input_ids))
print("Labels length:", len(labels))

# Find where labels are NOT -100
valid_labels_idx = torch.where(labels != -100)[0]
if len(valid_labels_idx) > 0:
    first_valid = valid_labels_idx[0].item()
    print("First valid label index:", first_valid)
    print("Masked text:", processor.decode(input_ids[:first_valid]))
    print("Target text to predict:", processor.decode(input_ids[first_valid:]))
else:
    print("ALL LABELS ARE -100! (MASKING FAILED)")

