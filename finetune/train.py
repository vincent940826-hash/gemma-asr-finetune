import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model
from model_setup import get_processor_and_model, apply_lora
from data_prep import ASRDatasetLoader, GemmaASRDataset, ASRDataCollator

def main():
    processor, model = get_processor_and_model()
    
    # Use the regex target_modules defined in model_setup.py
    model = apply_lora(model)
    
    # Load dataset
    loader = ASRDatasetLoader(processor)
    print("Loading datasets...")
    raw_train_dataset = loader.load_common_voice(split="train")
    raw_eval_dataset = loader.load_common_voice(split="validation")
    
    train_dataset = GemmaASRDataset(raw_train_dataset, prompt=loader.gemma_prompt)
    eval_dataset = GemmaASRDataset(raw_eval_dataset, prompt=loader.gemma_prompt)
    collator = ASRDataCollator(processor)
    
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, # 提高學習率 (V6 已解決格式衝突，可安全提速)
        warmup_ratio=0.1,   # 增加預熱步數，穩定初期訓練
        num_train_epochs=3,
        fp16=True, # Critical for V100
        bf16=False,
        optim="paged_adamw_8bit", # Critical for VRAM limit
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=20, # 保留最近 20 個 Checkpoint 避免硬碟塞滿
        remove_unused_columns=False, # Required because inputs are dynamic dicts
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("./checkpoints/final_lora")

if __name__ == "__main__":
    main()
