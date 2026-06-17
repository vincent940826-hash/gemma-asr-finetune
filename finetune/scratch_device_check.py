import torch
from model_setup import get_processor_and_model

print("Loading model...")
processor, model = get_processor_and_model()
print("Model loaded.")

# Print the model device and the device of its parameters
print(f"Model device attribute: {model.device}")
print(f"First parameter device: {next(model.parameters()).device}")
print(f"First parameter dtype: {next(model.parameters()).dtype}")
print(f"Is model on CUDA: {next(model.parameters()).is_cuda}")
