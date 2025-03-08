import torch
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Samanantar dataset (English-Hindi, "hi" subset)
dataset = load_dataset("ai4bharat/samanantar", "hi",
                       # Only 'train' split exists
                       split={"train": "train[:100000]"})['train']

# ✅ Manually split the dataset into train (90%) and validation (10%)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

# ✅ Model selection for English to Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"

# ✅ Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(
    model_name).to(device)  # Move model to GPU

# ✅ Tokenization function for Samanantar dataset


def tokenize_function(examples):
    source_texts = examples["src"]  # English text
    target_texts = examples["tgt"]  # Hindi text

    # Tokenize source text (input to the model)
    model_inputs = tokenizer(
        source_texts, truncation=True, padding="max_length", max_length=128)

    # Tokenize target text (output/labels for the model)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, truncation=True,
                           padding="max_length", max_length=128)

    # Assign tokenized target text as labels
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ✅ Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_validation = validation_dataset.map(tokenize_function, batched=True)

# ✅ Training arguments (optimized for GPU)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    # Larger batch size for GPU
    per_device_train_batch_size=16 if torch.cuda.is_available() else 4,
    per_device_eval_batch_size=16 if torch.cuda.is_available() else 4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),  # Enable mixed precision for GPU
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
)

# ✅ Train the model
trainer.train()

# ✅ Save final model
trainer.save_model("./final_model")
