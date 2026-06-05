import torch
import streamlit as st
import io
import soundfile as sf
import transformers
from transformers import pipeline

# 1. Main Page Setup
st.set_page_config(page_title="Runyankole AI App", layout="centered")
st.title("🇺🇬 True Runyankole AI Translator")
st.write("Powered by an open-access, regional fine-tuned translation engine.")
st.markdown("---")

# --- SECURE AUTHENTICATION CONFIGURATION ---
try:
    HF_ACCESS_TOKEN = st.secrets["HF_ACCESS_TOKEN"]
except Exception:
    st.error("❌ Missing HF_ACCESS_TOKEN! Please configure your secrets.toml file locally or add it to your Streamlit Cloud Dashboard setup.")
    st.stop()
# -------------------------------------------

# 2. Public Local AI Model Pipelines
@st.cache_resource
def load_speech_models():
    """Loads OpenAI's lightweight public speech transcriber."""
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

@st.cache_resource
def load_translation_engine():
    """
    Loads Sunbird AI's specialized NLLB adaptation using an explicit 
    authentication token securely extracted from Streamlit environment parameters.
    """
    model_name = "Sunbird/translate-nllb-1.3b-salt"
    
    tokenizer = transformers.NllbTokenizer.from_pretrained(
        model_name, 
        token=HF_ACCESS_TOKEN
    )
    model = transformers.M2M100ForConditionalGeneration.from_pretrained(
        model_name, 
        token=HF_ACCESS_TOKEN
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

# Initialize AI Engines safely
try:
    asr_pipeline = load_speech_models()
    tokenizer, translation_model, device = load_translation_engine()
except Exception as init_err:
    st.error(f"Failed to load public AI model weights: {init_err}")
    st.stop()

# Sunbird SALT custom embedded language token string mappings
# NLLB Tokenizers accept string keys natively to safely handle vocab adjustments
language_tokens = {
    'eng': 'eng_Latn',  
    'nyn': 'nyn_Latn'   
}

def translate_via_neural_net(text, direction_mode):
    try:
        if direction_mode == "English to Runyankole":
            src_lang = language_tokens['eng']
            tgt_lang_token_id = 256002  # Forced target token register ID for 'nyn'
        else:
            src_lang = language_tokens['nyn']
            tgt_lang_token_id = 256047  # Forced target token register ID for 'eng'

        # Set the tokenizer source language context correctly
        tokenizer.src_lang = src_lang
        
        # Tokenize text safely without losing the input data array
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output predictions using proper model tensors
        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_token_id,
            max_length=256,
            num_beams=4,      
            do_sample=False
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return "No text output generated."
        
    except Exception as e:
        return f"Translation Processing Error: {str(e)}"

# 3. Sidebar Selection Options
direction = st.sidebar.selectbox("Flow Mode:", ["English to Runyankole", "Runyankole to English"])

# 4. Handle Persistent Values 
# We track if a new voice transcription happened to alter text input defaults dynamically
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

# Audio Recorder Layout Layer
recorded_audio = st.audio_input("Record voice input")

if recorded_audio is not None:
    # Read raw audio bytes to check if it's genuinely a new recording event
    current_audio_bytes = recorded_audio.getvalue()
    if current_audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = current_audio_bytes
        
        with st.spinner("Decoding speech audio variables..."):
            try:
                # Reset stream pointer and read file array
                recorded_audio.seek(0)
                audio_data, sample_rate = sf.read(io.BytesIO(current_audio_bytes))
                
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Standardize stereo down to mono channel
                
                result = asr_pipeline({"raw": audio_data, "sampling_rate": sample_rate})
                st.session_state.voice_text = result["text"]
                st.rerun()  # Forces dynamic text_input box value refreshment immediately
            except Exception as e:
                st.error(f"Audio Decode Error: {e}")

# Form submission layout block
with st.form("translation_form", clear_on_submit=False):
    # Field automatically inherits text derived from either manual input typing or voice triggers
    sentence = st.text_input("Input Phrase:", value=st.session_state.voice_text)
    submit_button = st.form_submit_button(label="Translate Text", type="primary")

if submit_button and sentence:
    with st.spinner("Processing Model Inference Tensors..."):
        output_translation = translate_via_neural_net(sentence, direction)
        st.subheader("Neural Network Result:")
        st.success(output_translation)
