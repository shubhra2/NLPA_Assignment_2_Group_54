import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

# Global variables to store models and tokenizers
marian_model = None
marian_tokenizer = None

# Model name
MARIAN_MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

def load_models():
    global marian_model, marian_tokenizer
    try:
        print("Loading models from Hugging Face...")

        # Load MarianMT
        marian_tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL_NAME)
        marian_model = MarianMTModel.from_pretrained(MARIAN_MODEL_NAME)

        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

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
    Translates text using either mT5 or MarianMT based on language pair
    """
    global marian_model, marian_tokenizer

    # Handle edge cases
    if not input_text.strip():
        return "Error: Please enter some text to translate."

    if source_lang == target_lang:
        return "Source and target languages are the same. No translation needed."

    try:
        # Use MarianMT for English-Hindi translation
        if (source_lang == "en" and target_lang == "hi") or (source_lang == "hi" and target_lang == "en"):
            tokens = marian_tokenizer(input_text, return_tensors="pt", padding=True)
            translated_tokens = marian_model.generate(**tokens)
            translated_text = marian_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        return translated_text

    except Exception as e:
        print(f"Translation error: {e}")
        return f"Error during translation: {str(e)}"

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
    load_models()

    return demo


# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
