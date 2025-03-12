import gradio as gr
from huggingface_hub import HfFolder
from transformers import MarianMTModel, MarianTokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import torch  # Add this import at the top with other imports

# Global variables to store models and tokenizers
models = {}
tokenizers = {}
token = HfFolder.get_token()

# Model configurations
MODEL_CONFIGS = {
    "en-hi": {
        "model_path": "rooftopcoder/opus-mt-en-hi-samanantar-finetuned",
        "name": "English to Hindi"
    },
    "hi-en": {
        "model_path": "rooftopcoder/opus-mt-hi-en-samanantar-finetuned",
        "name": "Hindi to English"
    },
    "en-mr": {
        "model_path": "rooftopcoder/opus-mt-en-mr-samanantar-finetuned",
        "name": "English to Marathi"
    },
    "mr-en": {
        "model_path": "rooftopcoder/opus-mt-mr-en-samanantar-finetuned",
        "name": "Marathi to English"
    }
}

# Update language codes dictionary
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

# Reverse dictionary for display purposes
language_names = {v: k for k, v in language_codes.items()}

def load_models():
    try:
        print("Loading models from local storage...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        for direction, config in MODEL_CONFIGS.items():
            print(f"Loading {config['name']} model...")
            tokenizers[direction] = MarianTokenizer.from_pretrained(config["model_path"], token=token)
            models[direction] = MarianMTModel.from_pretrained(config["model_path"], token=token).to(device)

        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

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
    Translates text using MarianMT models
    """
    direction = f"{source_lang}-{target_lang}"
    if direction not in models or direction not in tokenizers:
        return "Error: Unsupported language pair"

    if not input_text.strip():
        return "Error: Please enter some text to translate."

    try:
        device = next(models[direction].parameters()).device
        tokens = tokenizers[direction](input_text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        translated = models[direction].generate(**tokens)
        translated = translated.cpu()
        output = tokenizers[direction].batch_decode(translated, skip_special_tokens=True)
        return output[0]
    except Exception as e:
        print(f"Translation error: {e}")
        return f"Error during translation: {str(e)}"

# Helper function for handling the UI translation process
def perform_translation(input_text, source_lang, target_lang):
    """Wrapper function for the Gradio interface"""
    source_code = language_codes[source_lang]
    target_code = language_codes[target_lang]

    # Handle transliteration for Hindi and Marathi
    if source_code == "en" and target_code in ["hi", "mr"]:
        common_indic_words = {
            "hi": ["namaste", "dhanyavad", "kaise", "hai", "aap", "tum", "main"],
            "mr": ["namaskar", "dhanyawad", "kase", "ahe", "tumhi", "mi"]
        }

        words = input_text.lower().split()
        if any(word in common_indic_words.get(target_code, []) for word in words):
            transliterated = transliterate_text(input_text)
            if transliterated != input_text:
                translation = translate(input_text, source_code, target_code)
                return f"Transliterated: {transliterated}\n\nTranslated: {translation}"

    return translate(input_text, source_code, target_code)

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Neural Machine Translation - Indian Languages") as demo:
        gr.Markdown("# Neural Machine Translation for Indian Languages")
        gr.Markdown("Translate between English, Hindi, and Marathi using MarianMT models")

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

        translate_btn = gr.Button("Translate", variant="primary")
        transliterate_btn = gr.Button("Transliterate Only", variant="secondary")

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

        # Examples for all language pairs
        gr.Examples(
            examples=[
                ["Hello, how are you?", "English", "Hindi"],
                ["नमस्ते, आप कैसे हैं?", "Hindi", "English"],
                ["Hello, how are you?", "English", "Marathi"],
                ["नमस्कार, तुम्ही कसे आहात?", "Marathi", "English"],
            ],
            inputs=[input_text, source_lang, target_lang],
            fn=perform_translation,
            outputs=output_text,
            cache_examples=True
        )

        gr.Markdown("""
        ## Model Information

        This demo uses fine-tuned MarianMT models for translation between:
        - English ↔️ Hindi
        - English ↔️ Marathi

        ### Features:
        - Bidirectional translation support
        - Transliteration support for romanized Indic text
        - Optimized models for each language pair
        """)

    return demo

# Launch the interface
if __name__ == "__main__":
    # Load all models before launching the interface
    if load_models():
        demo = create_interface()
        demo.launch(share=False)
    else:
        print("Failed to load models. Please check the model paths and try again.")
