import gradio as gr
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Global variables to store model and tokenizer
model = None
tokenizer = None

# Load the base mT5 model from Hugging Face


def load_mt5_model():
    global model, tokenizer
    try:
        print("Loading mT5 base model from Hugging Face...")
        # Using the base variant of mT5
        model_name = "rooftopcoder/mT5_base_English_Gujrati"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True)

        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading mT5 model: {e}")
        return None, None


# Dictionary for language codes
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
}

# Reverse dictionary for display purposes
language_names = {v: k for k, v in language_codes.items()}

# Dictionary for transliteration example (simplified)
transliteration_examples = {
    "en_to_hi": {
        "namaste": "नमस्ते",
        "kaise ho": "कैसे हो"
    }
}

# Function to perform translation with mT5


def translate(input_text, source_lang, target_lang):
    """
    Translates text from source language to target language using mT5

    Args:
        input_text: Text to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Translated text
    """
    global model, tokenizer

    # Handle edge cases
    if not input_text.strip():
        return "Error: Please enter some text to translate."

    if source_lang == target_lang:
        return "Source and target languages are the same. No translation needed."

    # Ensure model is loaded
    if model is None or tokenizer is None:
        model, tokenizer = load_mt5_model()
        if model is None or tokenizer is None:
            return "Error: Could not load translation model."

    try:
        # Format input for mT5 translation task
        task_prefix = f"translate {source_lang} to {target_lang}: "
        input_text_formatted = task_prefix + input_text

        # Tokenize the input
        inputs = tokenizer(input_text_formatted,
                           return_tensors="pt", max_length=512, truncation=True)

        # Generate translation with beam search
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,  # Adjust based on your needs
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True
        )

        # Decode the translation
        translated_text = tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        # Note: mT5 might not be fine-tuned for all these language pairs,
        # so results will vary in quality
        return translated_text

    except Exception as e:
        print(f"Translation error: {e}")
        # Fallback to placeholder for demo purposes
        return f"[Demo mode: This would be the {language_names[source_lang]} to {language_names[target_lang]} translation of: '{input_text}']"

# Helper function for handling the UI translation process


def perform_translation(input_text, source_lang, target_lang):
    """Wrapper function for the Gradio interface"""
    source_code = language_codes[source_lang]
    target_code = language_codes[target_lang]

    # Check for transliteration case (English text but Hindi language selected)
    if source_code == "en" and input_text.lower() in transliteration_examples.get(f"en_to_{target_code}", {}):
        transliterated = transliteration_examples[f"en_to_{target_code}"][input_text.lower(
        )]
        return f"Transliterated: {transliterated}"

    # Add a small delay to simulate processing
    # time.sleep(0.5)

    # Perform translation
    result = translate(input_text, source_code, target_code)
    return result

# Create Gradio interface


def create_interface():
    with gr.Blocks(title="Neural Machine Translation - Indian Languages") as demo:
        gr.Markdown("# Neural Machine Translation for Indian Languages")
        gr.Markdown(
            "Translate between multiple Indian languages using mT5 model")

        with gr.Row():
            with gr.Column():
                source_lang = gr.Dropdown(
                    choices=list(language_codes.keys()),
                    label="Source Language",
                    value="English"
                )

                input_text = gr.Textbox(
                    lines=5,
                    placeholder="Enter text to translate...",
                    label="Input Text"
                )

            with gr.Column():
                target_lang = gr.Dropdown(
                    choices=list(language_codes.keys()),
                    label="Target Language",
                    value="Hindi"
                )

                output_text = gr.Textbox(
                    lines=5,
                    label="Translated Text",
                    placeholder="Translation will appear here..."
                )

        # Add translation button
        with gr.Row():
            translate_btn = gr.Button("Translate", variant="primary")

        # Event handlers
        translate_btn.click(
            fn=perform_translation,
            inputs=[input_text, source_lang, target_lang],
            outputs=[output_text],
            api_name="translate"
        )

        # Add examples
        gr.Examples(
            examples=[
                ["Hello, how are you?", "English", "Hindi"],
                ["नमस्ते दुनिया", "Hindi", "English"],
                ["வணக்கம்", "Tamil", "Telugu"],
                ["నమస్కారం", "Telugu", "Bengali"]
            ],
            inputs=[input_text, source_lang, target_lang],
            fn=perform_translation,
            outputs=output_text,
            cache_examples=True  # Cache examples for faster loading
        )

        # Add information about the models
        gr.Markdown("""
        ## Model Information

        This demo uses Google's mT5-base model from Hugging Face Transformers.

        ### Notes:
        - The base mT5 model is not fine-tuned specifically for Indian languages
        - For a production system, the model should be fine-tuned on the Samanantar dataset
        - Translation quality varies by language pair

        ### Supported Language Pairs:
        - English ↔ Hindi
        - English ↔ Tamil
        - Hindi ↔ Marathi
        - And more...
        """)

    # Preload the model when the interface starts
    load_mt5_model()

    return demo


# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
