import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def print_model_modules(model):
    """Utility to print all named modules in the model for LoRA targeting."""
    for name, module in model.named_modules():
        print(name)

def get_processor_and_model(model_id="google/gemma-4-E4B-it"):
    """
    Loads the processor and 4-bit quantized model ready for PEFT.
    """
    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Configure 4-bit Quantization (Must use float16 for V100)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Fix the attention_invalid_logits_value to avoid float16 overflow
    if hasattr(model.config, "audio_config") and model.config.audio_config is not None:
        model.config.audio_config.attention_invalid_logits_value = -60000.0
    if hasattr(model, "audio_tower") and getattr(model, "audio_tower") is not None:
        if hasattr(model.audio_tower, "config") and model.audio_tower.config is not None:
            model.audio_tower.config.attention_invalid_logits_value = -60000.0

    return processor, model

def apply_lora(model, target_modules=None):
    """
    Applies LoRA to the targeted modules.
    If target_modules is None, dynamically targets all leaf linear layers
    in the audio tower and embed audio.
    """
    model = prepare_model_for_kbit_training(model)
    
    if target_modules is None:
        target_modules = []
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            has_children = len(list(module.children())) > 0
            if not has_children and ("Linear" in class_name or "linear" in name):
                # 1. Target LLM attention (q_proj, v_proj) to learn the format/punctuation without hallucination
                if "language_model" in name and ("q_proj" in name or "v_proj" in name):
                    target_modules.append(name)
                # 2. Target audio projectors to adapt acoustic features to the LLM
                elif "embed_audio" in name or "subsample_conv_projection" in name:
                    target_modules.append(name)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

if __name__ == "__main__":
    print("Loading processor and model...")
    processor, model = get_processor_and_model()
    print("Model loaded successfully.")
    
    print("\n--- Model Modules ---")
    # This utility is explicitly requested so the user can verify layer names
    print_model_modules(model)
    
    print("\nTo apply LoRA, use apply_lora(model) and pass the exact target_modules if needed.")
