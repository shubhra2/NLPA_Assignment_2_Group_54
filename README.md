# Neural Machine Translation for English-Hindi

This project implements a Neural Machine Translation system for English-Hindi translation using the MarianMT model (Helsinki-NLP/opus-mt-en-hi), with a user-friendly Gradio interface.

![NMT UI Screenshot](assets/nmt_ui_screenshot.png)

## Features

- Bidirectional translation between English and Hindi
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
2. Run the application:
```bash
python nmt_ui.py
```
3. Open your browser and navigate to `http://localhost:7860`

## Supported Language Pairs

- English -> Hindi (using Helsinki-NLP/opus-mt-en-hi MarianMT model)

## Project Structure

```
NLPA_Assignment_2_Group_54/
├── nmt_ui.py        # Main application file with Gradio interface
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT

## Group Members

- Shubhra J Gadhwala: 2023aa05750
- Sandeep Kumar Yadav: 2023ab05047
- Ravi Krishna Mayura: 2023ab05157
- Satheesh Kumar G: 2023ab05041
