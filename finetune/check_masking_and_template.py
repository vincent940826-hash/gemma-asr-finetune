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

print("=== MASKING TEST ===")
print("Input IDs length:", len(input_ids))
print("Labels length:", len(labels))

# Find where labels are NOT -100
valid_labels_idx = torch.where(labels != -100)[0]
if len(valid_labels_idx) > 0:
    first_valid = valid_labels_idx[0].item()
    print("First valid label index:", first_valid)
    print("Masked text:", processor.decode(input_ids[:first_valid]))
    print("Target text to predict:", processor.decode(input_ids[valid_labels_idx]))
else:
    print("ALL LABELS ARE -100! (MASKING FAILED)")

print("\n=== TEMPLATE INFERENCE VS TRAINING TEST ===")
train_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Prompt."},
            {"type": "audio", "audio": np.zeros(16000)},
        ]
    },
    {
        "role": "model",
        "content": [
            {"type": "text", "text": "Response."}
        ]
    }
]
train_template = processor.apply_chat_template(train_messages, tokenize=False)
print("TRAIN TEMPLATE:\n", repr(train_template))

inference_messages = [[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Prompt."},
            {"type": "audio", "audio": np.zeros(16000)},
        ]
    }
]]
infer_template_default = processor.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)[0]
infer_template_no_think = processor.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)[0]

print("\nINFERENCE TEMPLATE (DEFAULT):\n", repr(infer_template_default))
print("\nINFERENCE TEMPLATE (NO THINK):\n", repr(infer_template_no_think))
