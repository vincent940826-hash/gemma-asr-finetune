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

    # 2. Load model on GPU
    print("Loading base model on GPU...")
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda"
    )

    # No V13 merging; we are starting fresh with Deep Voice Mapping (V16)

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
    
    if target_modules is None:
        # V16 Deep Voice Mapping: Target EXACTLY the same layers as V13
        target_modules = ['audio_tower.layers.0.self_attn.k_proj.linear', 'audio_tower.layers.5.self_attn.v_proj.linear', 'audio_tower.layers.2.self_attn.v_proj.linear', 'audio_tower.layers.0.self_attn.q_proj.linear', 'audio_tower.layers.10.self_attn.q_proj.linear', 'audio_tower.layers.1.self_attn.q_proj.linear', 'post.linear', 'audio_tower.layers.2.self_attn.q_proj.linear', 'audio_tower.layers.6.self_attn.v_proj.linear', 'ffw_layer_1.linear', 'audio_tower.layers.11.self_attn.q_proj.linear', 'audio_tower.layers.8.self_attn.k_proj.linear', 'audio_tower.layers.6.self_attn.k_proj.linear', 'audio_tower.layers.5.self_attn.q_proj.linear', 'audio_tower.layers.0.self_attn.v_proj.linear', 'audio_tower.layers.11.self_attn.v_proj.linear', 'audio_tower.layers.10.self_attn.k_proj.linear', 'audio_tower.layers.5.self_attn.k_proj.linear', 'audio_tower.layers.7.self_attn.k_proj.linear', 'embed_audio.embedding_projection', 'linear_start.linear', 'audio_tower.layers.3.self_attn.q_proj.linear', 'audio_tower.layers.1.self_attn.v_proj.linear', 'audio_tower.layers.4.self_attn.v_proj.linear', 'output_proj', 'audio_tower.layers.3.self_attn.v_proj.linear', 'audio_tower.layers.9.self_attn.k_proj.linear', 'audio_tower.layers.9.self_attn.v_proj.linear', 'audio_tower.layers.10.self_attn.v_proj.linear', 'audio_tower.layers.7.self_attn.q_proj.linear', 'audio_tower.layers.11.self_attn.k_proj.linear', 'audio_tower.layers.1.self_attn.k_proj.linear', 'relative_k_proj', 'audio_tower.layers.4.self_attn.k_proj.linear', 'audio_tower.layers.2.self_attn.k_proj.linear', 'audio_tower.layers.7.self_attn.v_proj.linear', 'input_proj_linear', 'audio_tower.layers.8.self_attn.v_proj.linear', 'audio_tower.layers.6.self_attn.q_proj.linear', 'audio_tower.layers.3.self_attn.k_proj.linear', 'ffw_layer_2.linear', 'audio_tower.layers.4.self_attn.q_proj.linear', 'audio_tower.layers.8.self_attn.q_proj.linear', 'audio_tower.layers.9.self_attn.q_proj.linear', 'linear_end.linear']

    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=target_modules,
        lora_dropout=0.1,
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
