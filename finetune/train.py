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
    # For demonstration, we load a small subset of Common Voice
    # In a real run, you'd load the full dataset or local jsonl
    print("Loading datasets...")
    raw_dataset = loader.load_common_voice(split="train", limit=100) 
    
    train_dataset = GemmaASRDataset(raw_dataset, prompt=loader.gemma_prompt)
    collator = ASRDataCollator(processor)
    
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True, # Critical for V100
        bf16=False,
        optim="paged_adamw_8bit", # Critical for VRAM limit
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False, # Required because inputs are dynamic dicts
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("./checkpoints/final_lora")

if __name__ == "__main__":
    main()
