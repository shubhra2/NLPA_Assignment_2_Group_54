import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import torch  # Add this import at the top with other imports

# Global variables to store models and tokenizers
marian_model = None
marian_tokenizer = None

# Model name
MARIAN_MODEL_NAME = "./final_model"


def load_models():
    global marian_model, marian_tokenizer
    try:
        print("Loading models from Hugging Face...")

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load MarianMT
        marian_tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL_NAME)
        marian_model = MarianMTModel.from_pretrained(MARIAN_MODEL_NAME)

        # Move model to GPU if available
        marian_model = marian_model.to(device)

        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# Update language codes dictionary to only include English and Hindi
language_codes = {
    "English": "en",
    "Hindi": "hi"
}

# Reverse dictionary for display purposes
language_names = {v: k for k, v in language_codes.items()}

# Function to perform transliteration from English to Hindi


def transliterate_text(text, from_scheme=sanscript.ITRANS, to_scheme=sanscript.DEVANAGARI):
    """
    Transliterates text from one script to another
    Default is from ITRANS (Roman) to Devanagari (Hindi)
    """
    try:
        return transliterate(text, from_scheme, to_scheme)
    except Exception as e:
        print(f"Transliteration error: {e}")
        return text

# Function to perform translation with MarianMT


def translate(input_text, source_lang, target_lang):
    """
    Translates text using MarianMT for English-Hindi translation
    """
    global marian_model, marian_tokenizer

    # Handle edge cases
    if not input_text.strip():
        return "Error: Please enter some text to translate."

    if source_lang == target_lang:
        return "Source and target languages are the same. No translation needed."

    try:
        # Get the device the model is on
        device = next(marian_model.parameters()).device

        # Tokenize and move to same device as model
        tokens = marian_tokenizer(input_text, return_tensors="pt",
                                padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Generate translation
        translated = marian_model.generate(**tokens)

        # Move back to CPU for decoding
        translated = translated.cpu()
        output = marian_tokenizer.batch_decode(
            translated, skip_special_tokens=True)
        return output[0]

    except Exception as e:
        print(f"Translation error: {e}")
        return f"Error during translation: {str(e)}"

# Helper function for handling the UI translation process


def perform_translation(input_text, source_lang, target_lang):
    """Wrapper function for the Gradio interface"""
    source_code = language_codes[source_lang]
    target_code = language_codes[target_lang]

    # If the source is English and target is Hindi, check for transliteration
    # This will detect romanized Hindi text and convert it to Devanagari
    if source_code == "en" and target_code == "hi":
        # Check if the input might be romanized Hindi (contains common Hindi words)
        common_hindi_words = ["namaste", "dhanyavad",
                              "kaise", "hai", "aap", "tum", "main"]
        words = input_text.lower().split()

        # If any common Hindi word is found in English text, offer transliteration
        if any(word in common_hindi_words for word in words):
            # Perform transliteration
            transliterated = transliterate_text(input_text)

            # If transliteration changed the text, return both transliteration and translation
            if transliterated != input_text:
                translation = translate(input_text, source_code, target_code)
                return f"Transliterated: {transliterated}\n\nTranslated: {translation}"

    # Standard translation (no transliteration needed)
    return translate(input_text, source_code, target_code)

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

        # Add transliteration button (new)
        with gr.Row():
            transliterate_btn = gr.Button(
                "Transliterate Only", variant="secondary")

        # Event handlers
        translate_btn.click(
            fn=perform_translation,
            inputs=[input_text, source_lang, target_lang],
            outputs=[output_text],
            api_name="translate"
        )

        # Direct transliteration handler (new)
        def direct_transliterate(text):
            if not text.strip():
                return "Please enter text to transliterate"
            return transliterate_text(text)

        transliterate_btn.click(
            fn=direct_transliterate,
            inputs=[input_text],
            outputs=[output_text],
            api_name="transliterate"
        )

        # Update examples to include transliteration examples
        gr.Examples(
            examples=[
                ["Hello, how are you?", "English", "Hindi"],
                ["What's your name?", "English", "Hindi"],
                # Transliteration example
                ["namaste mere dost", "English", "Hindi"],
                # Transliteration example
                ["aap kaise ho", "English", "Hindi"],
            ],
            inputs=[input_text, source_lang, target_lang],
            fn=perform_translation,
            outputs=output_text,
            cache_examples=True
        )

        # Update model information markdown
        gr.Markdown("""
        ## Model Information

        This demo uses the MarianMT model from Hugging Face Transformers.

        ### Notes:
        - The model supports English-Hindi translation
        - Based on the Helsinki-NLP/opus-mt-en-hi model
        - Optimized for English -> Hindi translation pairs
        - Now includes transliteration support for Romanized Hindi text

        ### Supported Features:
        - English -> Hindi translation
        - Romanized Hindi -> Devanagari Hindi transliteration

        ### Examples of Transliteration:
        - "namaste" → "नमस्ते"
        - "aap kaise ho" → "आप कैसे हो"
        - "mera naam" → "मेरा नाम"
        """)

    # Preload the model when the interface starts
    load_models()

    return demo


# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
