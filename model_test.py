import torch
from huggingface_hub import HfFolder
from transformers import MarianMTModel, MarianTokenizer

token = HfFolder.get_token()
# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "en-hi": "./final_model_en_hi",
    "hi-en": "./final_model_hi_en",
    "en-mr": "./final_model_en_mr",
    "mr-en": "./final_model_mr_en"
}

def load_model(direction):
    if direction not in MODEL_PATHS:
        raise ValueError(f"Invalid direction. Choose from: {list(MODEL_PATHS.keys())}")

    model_path = MODEL_PATHS[direction]
    tokenizer = MarianTokenizer.from_pretrained(model_path, token=token)
    model = MarianMTModel.from_pretrained(model_path, token=token).to(device)
    return tokenizer, model

def translate_text(text, direction):
    tokenizer, model = load_model(direction)

    # Prepare input tokens
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = {key: value.to(device) for key, value in tokens.items()}

    translated = model.generate(**tokens)
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return output[0]

# Test examples for all directions
test_sentences = {
    "en-hi": "Hello, how are you?",
    "hi-en": "नमस्ते, आप कैसे हैं?",
    "en-mr": "Hello, how are you?",
    "mr-en": "नमस्कार, तुम्ही कसे आहात?"
}

# Run tests for all directions
for direction, text in test_sentences.items():
    print(f"\nTesting {direction} translation:")
    print(f"Input: {text}")
    translated = translate_text(text, direction)
    print(f"Output: {translated}")
