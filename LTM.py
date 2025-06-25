import gradio as gr
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Define available translation models
TRANSLATION_MODELS = {
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English to German": "Helsinki-NLP/opus-mt-en-de",
    "English to Italian": "Helsinki-NLP/opus-mt-en-it",
    "English to Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "English to Chinese (Simplified)": "Helsinki-NLP/opus-mt-en-zh",
    "English to Chinese (Traditional/Cantonese)": "Helsinki-NLP/opus-mt-en-zh",
    "English to Cantonese (via ZH)": "Helsinki-NLP/opus-mt-en-zh",
    "French to English": "Helsinki-NLP/opus-mt-fr-en",
    "Spanish to English": "Helsinki-NLP/opus-mt-es-en",
    "German to English": "Helsinki-NLP/opus-mt-de-en",
    "Italian to English": "Helsinki-NLP/opus-mt-it-en",
    "Chinese to English": "Helsinki-NLP/opus-mt-zh-en",
    "Cantonese/Chinese to English": "Helsinki-NLP/opus-mt-zh-en",
}

# Alternative models for better Chinese/Cantonese support
ALTERNATIVE_MODELS = {
    "English to Cantonese (Alternative)": "facebook/nllb-200-distilled-600M",
    "Cantonese to English (Alternative)": "facebook/nllb-200-distilled-600M",
}


class TranslationApp:
    def __init__(self):
        self.current_model = None
        self.current_pipeline = None
        self.current_model_name = None
        self.tokenizer = None

    def load_model(self, model_name):
        """Load a translation model if not already loaded"""
        if self.current_model_name != model_name:
            print(f"Loading model: {model_name}")

            # Check if it's the NLLB model (for Cantonese support)
            if "nllb" in model_name.lower():
                # Load NLLB model with tokenizer for language codes
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

                self.current_pipeline = pipeline(
                    "translation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=512
                )
            else:
                # Create standard translation pipeline
                self.current_pipeline = pipeline(
                    "translation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )

            # Create LangChain HuggingFacePipeline
            self.current_model = HuggingFacePipeline(pipeline=self.current_pipeline)
            self.current_model_name = model_name

            print(f"Model {model_name} loaded successfully!")

    def translate_with_langchain(self, text, translation_direction):
        """Translate text using LangChain with the selected model"""
        if not text.strip():
            return "Please enter text to translate."

        # Check if it's an alternative model
        model_name = ALTERNATIVE_MODELS.get(translation_direction)
        if not model_name:
            model_name = TRANSLATION_MODELS.get(translation_direction)

        if not model_name:
            return "Invalid translation direction selected."

        # Load the model if needed
        self.load_model(model_name)

        # Handle NLLB model specifically for Cantonese
        if "nllb" in model_name.lower() and "Cantonese" in translation_direction:
            try:
                if "English to Cantonese" in translation_direction:
                    # Set source language to English and target to Cantonese
                    self.tokenizer.src_lang = "eng_Latn"
                    forced_bos_token_id = self.tokenizer.lang_code_to_id["yue_Hant"]

                    result = self.current_pipeline(
                        text,
                        forced_bos_token_id=forced_bos_token_id
                    )
                else:  # Cantonese to English
                    # Set source language to Cantonese and target to English
                    self.tokenizer.src_lang = "yue_Hant"
                    forced_bos_token_id = self.tokenizer.lang_code_to_id["eng_Latn"]

                    result = self.current_pipeline(
                        text,
                        forced_bos_token_id=forced_bos_token_id
                    )

                return result[0]['translation_text']

            except Exception as e:
                return f"Translation error: {str(e)}"

        # Standard translation for other models
        # Create a prompt template for translation
        prompt = PromptTemplate(
            input_variables=["text"],
            template="{text}"
        )

        # Create a chain
        chain = LLMChain(llm=self.current_model, prompt=prompt)

        try:
            # Run the translation
            result = chain.run(text=text)

            # Extract the translated text
            if ">>" in result and "<<" in result:
                translated = result.split("<<", 1)[1].strip()
            else:
                translated = result.strip()

            return translated

        except Exception as e:
            return f"Translation error: {str(e)}"

    def translate_direct(self, text, translation_direction):
        """Alternative: Direct translation without LangChain for comparison"""
        if not text.strip():
            return "Please enter text to translate."

        # Check alternative models first
        model_name = ALTERNATIVE_MODELS.get(translation_direction)
        if not model_name:
            model_name = TRANSLATION_MODELS.get(translation_direction)

        if not model_name:
            return "Invalid translation direction selected."

        # Load the model if needed
        if self.current_model_name != model_name:
            if "nllb" in model_name.lower():
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

                self.current_pipeline = pipeline(
                    "translation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=512
                )
            else:
                self.current_pipeline = pipeline(
                    "translation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            self.current_model_name = model_name

        try:
            # Handle NLLB model for Cantonese
            if "nllb" in model_name.lower() and "Cantonese" in translation_direction:
                if "English to Cantonese" in translation_direction:
                    self.tokenizer.src_lang = "eng_Latn"
                    forced_bos_token_id = self.tokenizer.lang_code_to_id["yue_Hant"]
                else:  # Cantonese to English
                    self.tokenizer.src_lang = "yue_Hant"
                    forced_bos_token_id = self.tokenizer.lang_code_to_id["eng_Latn"]

                result = self.current_pipeline(
                    text,
                    forced_bos_token_id=forced_bos_token_id
                )
            else:
                # Standard translation
                result = self.current_pipeline(text, max_length=512)

            return result[0]['translation_text']

        except Exception as e:
            return f"Translation error: {str(e)}"


# Initialize the app
app = TranslationApp()


# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Language Translation with LangChain") as interface:
        gr.Markdown("""
        # 🌐 Language Translation App

        This app uses Hugging Face models with LangChain to translate text between different languages.
        Select a translation direction and enter your text below.
        """)

        with gr.Row():
            with gr.Column():
                # Input components
                translation_direction = gr.Dropdown(
                    choices=list(TRANSLATION_MODELS.keys()) + list(ALTERNATIVE_MODELS.keys()),
                    value="English to French",
                    label="Translation Direction"
                )

                input_text = gr.Textbox(
                    label="Text to Translate",
                    placeholder="Enter text here...",
                    lines=5
                )

                with gr.Row():
                    translate_btn = gr.Button("🔄 Translate", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            with gr.Column():
                # Output components
                output_text = gr.Textbox(
                    label="Translated Text",
                    lines=5,
                    interactive=False
                )

                with gr.Accordion("Advanced Options", open=False):
                    use_langchain = gr.Checkbox(
                        label="Use LangChain (vs. direct pipeline)",
                        value=True,
                        info="Toggle between LangChain integration and direct HuggingFace pipeline"
                    )

        # Examples
        gr.Examples(
            examples=[
                ["Hello, how are you today?", "English to French"],
                ["The weather is beautiful today.", "English to Spanish"],
                ["I love programming with Python.", "English to German"],
                ["Hello, how are you?", "English to Cantonese (Alternative)"],
                ["Welcome to Hong Kong!", "English to Chinese (Traditional/Cantonese)"],
                ["Bonjour, comment allez-vous?", "French to English"],
                ["Me encanta aprender nuevos idiomas.", "Spanish to English"],
                ["你好嗎？", "Cantonese/Chinese to English"],
                ["今日天氣好好。", "Cantonese to English (Alternative)"],
            ],
            inputs=[input_text, translation_direction],
            label="Example Translations"
        )

        # Event handlers
        def translate(text, direction, use_lc):
            if use_lc:
                return app.translate_with_langchain(text, direction)
            else:
                return app.translate_direct(text, direction)

        translate_btn.click(
            fn=translate,
            inputs=[input_text, translation_direction, use_langchain],
            outputs=output_text
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=[],
            outputs=[input_text, output_text]
        )

        # Info section
        gr.Markdown("""
        ### ℹ️ About this App

        **Features:**
        - Uses Helsinki-NLP's OPUS-MT models for high-quality translation
        - Supports Cantonese translation using Facebook's NLLB model
        - Integrates with LangChain for extended functionality
        - Supports multiple language pairs including Cantonese
        - GPU acceleration when available

        **How it works:**
        1. Select your desired translation direction
        2. Enter the text you want to translate
        3. Click "Translate" to see the result

        **Cantonese Support:**
        - For basic Cantonese, use "English to Chinese (Traditional/Cantonese)"
        - For better Cantonese support, use the "Alternative" options with NLLB model
        - NLLB model supports 200+ languages including Cantonese (yue_Hant)

        **Note:** First translation might take longer as the model loads. NLLB model is larger and may take more time to load.
        """)

    return interface


# Additional utility functions for extended functionality
def create_batch_translation_chain(model):
    """Create a LangChain chain for batch translation"""
    batch_prompt = PromptTemplate(
        input_variables=["texts"],
        template="Translate the following texts:\n{texts}"
    )
    return LLMChain(llm=model, prompt=batch_prompt)


def create_context_aware_translation_chain(model):
    """Create a LangChain chain that considers context"""
    context_prompt = PromptTemplate(
        input_variables=["context", "text"],
        template="Context: {context}\nTranslate: {text}"
    )
    return LLMChain(llm=model, prompt=context_prompt)


# Main execution
if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()

    # Launch the app
    interface.launch(
        share=True,  # Set to True to create a public link
        debug=True,  # Enable debug mode for development
        server_name="localhost",
        server_port=7860  # Default Gradio port
    )