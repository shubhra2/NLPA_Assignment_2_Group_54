from huggingface_hub import HfApi, HfFolder
import torch
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model names for different translation directions
MODEL_CONFIGS = {
    "en-hi": {
        "model_name": "Helsinki-NLP/opus-mt-en-hi",
        "save_dir": "./final_model_en_hi",
        "repo_id": "rooftopcoder/opus-mt-en-hi-samanantar-finetuned"
    },
    "hi-en": {
        "model_name": "Helsinki-NLP/opus-mt-hi-en",
        "save_dir": "./final_model_hi_en",
        "repo_id": "rooftopcoder/opus-mt-hi-en-samanantar-finetuned"
    },
    "en-mr": {
        "model_name": "Helsinki-NLP/opus-mt-en-mr",
        "save_dir": "./final_model_en_mr",
        "repo_id": "rooftopcoder/opus-mt-en-mr-samanantar-finetuned"
    },
    "mr-en": {
        "model_name": "Helsinki-NLP/opus-mt-mr-en",
        "save_dir": "./final_model_mr_en",
        "repo_id": "rooftopcoder/opus-mt-mr-en-samanantar-finetuned"
    }
}

# Load and split datasets
dataset_size = 1000
dataset_hi = load_dataset("ai4bharat/samanantar", "hi", split={"train": f"train[:{dataset_size}]"})['train']
split_dataset_hi = dataset_hi.train_test_split(test_size=0.1, seed=42)

dataset_mr = load_dataset("ai4bharat/samanantar", "mr", split={"train": f"train[:{dataset_size}]"})['train']
split_dataset_mr = dataset_mr.train_test_split(test_size=0.1, seed=42)

def get_tokenizer_and_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def create_tokenize_function(tokenizer, source_key="src", target_key="tgt"):
    def tokenize_fn(examples):
        source_texts = examples[source_key]
        target_texts = examples[target_key]
        model_inputs = tokenizer(source_texts, truncation=True, padding="max_length", max_length=128)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target_texts, truncation=True, padding="max_length", max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize_fn

def inspect_tokenized_dataset(tokenized_dataset, tokenizer, num_examples=2):
    """
    Print the first few examples from the tokenized dataset for inspection.

    Args:
        tokenized_dataset: The dataset after tokenization
        tokenizer: The tokenizer used for encoding/decoding
        num_examples: Number of examples to print (default: 2)
    """
    print("\nInspecting tokenized dataset:")
    print("-" * 50)

    for i in range(min(num_examples, len(tokenized_dataset))):
        example = tokenized_dataset[i]

        # Decode the input_ids back to text
        input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        # Decode the labels back to text
        label_text = tokenizer.decode(example['labels'], skip_special_tokens=True)

        print(f"\nExample {i+1}:")
        print(f"Input text: {input_text}")
        print(f"Target text: {label_text}")
        print(f"Input shape: {len(example['input_ids'])}")
        print(f"Label shape: {len(example['labels'])}")
        print("-" * 50)

def train_model(config, train_dataset, val_dataset, direction):
    print(f"\nTraining {direction} model...")
    tokenizer, model = get_tokenizer_and_model(config["model_name"])

    # Determine source and target keys based on direction
    source_key = "src" if direction.startswith("en") else "tgt"
    target_key = "tgt" if direction.startswith("en") else "src"

    tokenize_fn = create_tokenize_function(tokenizer, source_key, target_key)

    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)

    inspect_tokenized_dataset(tokenized_train, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{config['save_dir']}_results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16 if torch.cuda.is_available() else 4,
        per_device_eval_batch_size=16 if torch.cuda.is_available() else 4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=f"{config['save_dir']}_logs",
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        hub_model_id=config["repo_id"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config["save_dir"])
    trainer.processing_class.save_pretrained(config["save_dir"])
    # Upload to Huggingface Hub
    token = HfFolder.get_token()
    trainer.push_to_hub(token=token)

# Train models for each direction
for direction, config in MODEL_CONFIGS.items():
    if 'hi' in direction:
        train_model(config, split_dataset_hi["train"], split_dataset_hi["test"], direction)
    elif 'mr' in direction:
        train_model(config, split_dataset_mr["train"], split_dataset_mr["test"], direction)
