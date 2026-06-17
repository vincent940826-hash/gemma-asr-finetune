import torch
import librosa
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import PeftModel

def load_inference_model(base_model_id="google/gemma-4-E4B-it", peft_model_id="./checkpoints/final_lora"):
    processor = AutoProcessor.from_pretrained(base_model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForMultimodalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()
    return processor, model

def transcribe(audio_path, processor, model):
    # Fetch sampling rate dynamically
    if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
        sr = processor.feature_extractor.sampling_rate
    else:
        sr = 16000
        
    audio_array, _ = librosa.load(audio_path, sr=sr)
    
    prompt = (
        "Transcribe the following speech in Traditional Chinese into Traditional Chinese text. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
        "<|audio|>"
    )
    
    inputs = processor(
        text=prompt,
        audio=[audio_array],
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        # Autoregressive generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=128
        )
        
    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_length:]
    transcription = processor.decode(generated_ids, skip_special_tokens=True)
    return transcription

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio.wav>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    print("Loading model for inference...")
    processor, model = load_inference_model()
    
    print(f"Transcribing {audio_file}...")
    result = transcribe(audio_file, processor, model)
    print("\nTranscription Result:")
    print(result)
