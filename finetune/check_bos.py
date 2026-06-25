from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
messages = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]
text = processor.apply_chat_template(messages, tokenize=False)
print("Template text:", repr(text[0]))
encoded = processor(text=text, return_tensors="pt")
print("Tokens:", encoded["input_ids"][0][:10])
print("Token strings:", processor.batch_decode(encoded["input_ids"][0][:10]))
