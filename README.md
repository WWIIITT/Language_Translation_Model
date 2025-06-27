# 🌐 Language Translation App with LangChain

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful multilingual translation application built with Gradio, LangChain, and Hugging Face Transformers. This app provides an intuitive web interface for translating text between multiple languages, with special support for Cantonese translation.


## ✨ Features

- **Multiple Language Support**: Translate between 10+ language pairs including English, French, Spanish, German, Italian, Portuguese, Chinese, and Cantonese
- **Cantonese Specialization**: Enhanced Cantonese translation using Facebook's NLLB-200 model
- **Dual Translation Modes**: Switch between LangChain integration and direct HuggingFace pipeline
- **GPU Acceleration**: Automatic GPU detection and utilization for faster translations
- **User-Friendly Interface**: Clean, modern Gradio interface with examples and clear instructions
- **Batch Processing Ready**: Infrastructure for batch translation capabilities
- **Context-Aware Translation**: Framework for context-sensitive translations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/WWIIITT/language-translation-app.git
cd language-translation-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python LTM.py
```

The app will launch and provide you with a local URL (typically `http://localhost:7860`). If you set `share=True`, it will also provide a public URL for sharing.

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
gradio>=4.0.0
langchain>=0.1.0
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

## 🗺️ Supported Translation Pairs

### Standard Models (Helsinki-NLP OPUS-MT)
- English ↔ French
- English ↔ Spanish
- English ↔ German
- English ↔ Italian
- English ↔ Portuguese
- English ↔ Chinese (Simplified/Traditional)
- Cantonese/Chinese → English

### Enhanced Cantonese Support (Facebook NLLB-200)
- English ↔ Cantonese (Alternative) - Better quality for Cantonese

## 💡 Usage

### Basic Translation

1. Select your translation direction from the dropdown
2. Enter the text you want to translate
3. Click the "🔄 Translate" button
4. View the translated result

### Advanced Options

- **LangChain Mode**: Toggle between LangChain integration and direct pipeline for different processing approaches
- **Examples**: Click on any example to quickly test the translation

### API Usage

You can also use the translation functionality programmatically:

```python
from LTM import TranslationApp

# Initialize the app
app = TranslationApp()

# Translate text
translated = app.translate_with_langchain(
    "Hello, world!", 
    "English to French"
)
print(translated)  # "Bonjour, le monde!"
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Gradio UI     │────▶│  TranslationApp  │────▶│  HuggingFace    │
│   Interface      │     │     Class        │     │   Models        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           │
                        ┌──────────────┐                   │
                        │  LangChain   │◀──────────────────┘
                        │  Pipeline    │
                        └──────────────┘
```

## 🔧 Configuration

### Model Configuration

Models are defined in the `TRANSLATION_MODELS` and `ALTERNATIVE_MODELS` dictionaries. To add new models:

```python
TRANSLATION_MODELS["English to NewLanguage"] = "model-name-here"
```

### Performance Tuning

- **GPU Usage**: Automatically detected and used when available
- **Max Length**: Default 512 tokens, adjustable in pipeline configuration
- **Batch Size**: Can be modified for batch processing needs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black LTM.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure you have stable internet for first-time model downloads
   - Check available disk space (models can be 1-2GB each)

2. **GPU Not Detected**
   - Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
   - Install appropriate PyTorch version for your CUDA version

3. **Cantonese Translation Issues**
   - NLLB model requires additional memory (4GB+)
   - Ensure `sentencepiece` is properly installed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Helsinki-NLP](https://github.com/Helsinki-NLP) for OPUS-MT models
- [Facebook/Meta](https://github.com/facebookresearch) for NLLB-200 model
- [Gradio](https://gradio.app/) for the amazing UI framework
- [LangChain](https://langchain.com/) for the LLM orchestration framework
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library


## 🔮 Roadmap

- [ ] Add more language pairs
- [ ] Implement document translation
- [ ] Add translation quality metrics
- [ ] Create REST API endpoint
- [ ] Add translation history
- [ ] Implement custom model fine-tuning
- [ ] Add support for audio translation



⭐️ If you find this project useful, please consider giving it a star!
