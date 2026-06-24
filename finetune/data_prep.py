import os
import json
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset

class ASRDatasetLoader:
    def __init__(self, processor):
        self.processor = processor
        # Dynamically fetch sampling rate
        if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
            self.target_sr = processor.feature_extractor.sampling_rate
        else:
            self.target_sr = 16000 # Fallback
            
        self.gemma_prompt = "請將以下語音內容轉寫為繁體中文。"

    def process_audio(self, audio_path=None, audio_array=None, orig_sr=None):
        if audio_path is not None:
            audio_array, _ = librosa.load(audio_path, sr=self.target_sr)
        elif audio_array is not None and orig_sr is not None:
            if orig_sr != self.target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio_array

    def load_common_voice(self, split="train", limit=None):
        print(f"Loading Common Voice split: {split}")
        ds = load_dataset("voidful/common_voice_25_zh-tw", split=split)
        if limit is not None:
            ds = ds.select(range(limit))
            
        def standardize(batch):
            audio_data = batch["audio"]
            audio_array = self.process_audio(
                audio_array=audio_data["array"], 
                orig_sr=audio_data["sampling_rate"]
            )
            return {
                "audio_array": audio_array,
                "target_text": batch["sentence"]
            }
            
        ds = ds.map(standardize, remove_columns=ds.column_names, num_proc=1) # Reduced num_proc to avoid memory issues
        return ds

    def load_local_jsonl(self, jsonl_path):
        print(f"Loading local JSONL dataset: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        def gen():
            for line in lines:
                data = json.loads(line)
                yield {
                    "audio_filepath": data["audio_filepath"],
                    "text": data["text"]
                }
                
        ds = HFDataset.from_generator(gen)
        
        def standardize(batch):
            audio_array = self.process_audio(audio_path=batch["audio_filepath"])
            return {
                "audio_array": audio_array,
                "target_text": batch["text"]
            }
            
        ds = ds.map(standardize, remove_columns=ds.column_names, num_proc=1)
        return ds

    def get_combined_dataset(self, include_cv=True, local_jsonl_paths=None, cv_split="train"):
        datasets_to_concat = []
        if include_cv:
            datasets_to_concat.append(self.load_common_voice(split=cv_split))
        
        if local_jsonl_paths:
            for path in local_jsonl_paths:
                datasets_to_concat.append(self.load_local_jsonl(path))
                
        if not datasets_to_concat:
            raise ValueError("No datasets selected for loading.")
            
        combined_ds = concatenate_datasets(datasets_to_concat)
        return combined_ds


class GemmaASRDataset(Dataset):
    def __init__(self, hf_dataset, prompt):
        self.dataset = hf_dataset
        self.prompt = prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "audio_array": item["audio_array"],
            "prompt_text": self.prompt,
            "target_text": item["target_text"]
        }


class ASRDataCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, features):
        audio_arrays = [f["audio_array"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]
        target_texts = [f["target_text"] for f in features]
        
        batch_messages = []
        for i in range(len(features)):
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_texts[i]},
                        {"type": "audio", "audio": audio_arrays[i]},
                    ]
                },
                {
                    "role": "model",
                    "content": [
                        {"type": "text", "text": target_texts[i]}
                    ]
                }
            ])
            
        full_texts = self.processor.apply_chat_template(
            batch_messages,
            tokenize=False
        )
        
        # Processor pads audio lists dynamically to the max length in batch
        batch = self.processor(
            text=full_texts,
            audio=audio_arrays,
            return_tensors="pt",
            padding=True
        )
        
        labels = batch["input_ids"].clone()
        
        for i in range(len(labels)):
            # To compute prompt length accurately, we format a message with only the user turn
            user_msg = [[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_texts[i]},
                        {"type": "audio", "audio": audio_arrays[i]},
                    ]
                }
            ]]
            prompt_only_text = self.processor.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )[0]
            
            # Accurately compute prompt length WITH audio tokens
            prompt_inputs = self.processor(
                text=prompt_only_text,
                audio=[audio_arrays[i]],
                return_tensors="pt"
            )
            prompt_length = prompt_inputs["input_ids"].shape[1]
            
            if self.processor.tokenizer.padding_side == "right":
                labels[i, :prompt_length] = -100
            else:
                pad_len = (batch["input_ids"][i] == self.processor.tokenizer.pad_token_id).sum()
                labels[i, pad_len : pad_len + prompt_length] = -100
                
            labels[i, batch["input_ids"][i] == self.processor.tokenizer.pad_token_id] = -100
            
        batch["labels"] = labels
        return batch
