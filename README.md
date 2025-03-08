# Neural Machine Translation for English-Hindi

This project implements a Neural Machine Translation system for English-Hindi translation using the MarianMT model fine-tuned on 100k split of Samanantar, with a user-friendly Gradio interface.

![NMT UI Screenshot](assets/nmt_ui_screenshot.png)

## Features

- Unidirectional translation between English and Hindi
- User-friendly web interface built with Gradio
- Example translations included
- Built on Helsinki-NLP's MarianMT model

## Installation

### Local Setup with Virtual Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NLPA_Assignment_2_Group_54.git
cd NLPA_Assignment_2_Group_54
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your virtual environment is activated
2. Run the UI:
```bash
python nmt_ui.py
```
3. Open your browser and navigate to `http://localhost:7860`

## Supported Language Pairs

- English -> Hindi (using rooftopcoder/opus-mt-en-hi-samanantar-100k model)

## Training the Model

The `train.py` script is used to train the MarianMT model on the Samanantar dataset. The script performs the following steps:
- Loads the Samanantar dataset (English-Hindi subset).
- Splits the dataset into training and validation sets.
- Tokenizes the dataset.
- Sets up training arguments optimized for GPU.
- Trains the model using the Hugging Face `Trainer` class.
- Saves the trained model to the specified directory.
- Uploads the trained model to the Hugging Face Hub.

To train the model, run:
```bash
python train.py
```

## Testing the Model

The `model_test.py` script is used to test the trained MarianMT model. The script performs the following steps:
- Loads the trained model and tokenizer from the Hugging Face Hub.
- Translates a sample input text from English to Hindi.
- Prints the translated text.

To test the model, run:
```bash
python model_test.py
```

## User Interface

The `nmt_ui.py` script provides a Gradio-based user interface for translating text between English and Hindi. The interface includes options for transliteration of Romanized Hindi text to Devanagari script.

To launch the interface, run:
```bash
python nmt_ui.py
```

## Model Information

This project uses the MarianMT model from Hugging Face Transformers.

### Notes:
- The model supports English-Hindi translation.
- Based on the Helsinki-NLP/opus-mt-en-hi model.
- Optimized for English -> Hindi translation pairs.
- Includes transliteration support for Romanized Hindi text.

### Supported Features:
- English -> Hindi translation.
- Romanized Hindi -> Devanagari Hindi transliteration.

### Examples of Transliteration:
- "namaste" → "नमस्ते"
- "aap kaise ho" → "आप कैसे हो"
- "mera naam" → "मेरा नाम"

## Project Structure

```
NLPA_Assignment_2_Group_54/
├── nmt_ui.py        # Main application file with Gradio interface
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## License

MIT

## Group Members

- Shubhra J Gadhwala: 2023aa05750
- Sandeep Kumar Yadav: 2023ab05047
- Ravi Krishna Mayura: 2023ab05157
- Satheesh Kumar G: 2023ab05041
