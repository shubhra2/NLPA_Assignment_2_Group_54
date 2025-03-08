import torch
from transformers import MarianMTModel, MarianTokenizer

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained model
model_name = "./final_model"  # Path to saved model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(
    model_name).to(device)  # Move model to GPU


def translate_text(text):
    tokens = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    tokens = {key: value.to(device)
              for key, value in tokens.items()}  # Move tokens to GPU

    translated = model.generate(**tokens)
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return output[0]


# Example input text
input_text = "Hello, how are you?"
translated_text = translate_text(input_text)

print("Input:", input_text)
print("Translated:", translated_text)
