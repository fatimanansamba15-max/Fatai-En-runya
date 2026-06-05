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
# Safely pulls your Hugging Face key from your hidden environment configurations
try:
    HF_ACCESS_TOKEN = st.secrets["HF_ACCESS_TOKEN"]
except Exception:
    st.error("❌ Missing HF_ACCESS_TOKEN! Please configure your secrets.toml file locally or add it to your Streamlit Cloud Dashboard dashboard setup.")
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
    
    # Pass the secure token parameter directly into the model initializers
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

# Sunbird SALT custom embedded language token numerical ID mappings
language_tokens = {
    'eng': 256047,  # English vocabulary register ID
    'nyn': 256002   # Runyankole vocabulary register ID
}

def translate_via_neural_net(text, direction_mode):
    try:
        if direction_mode == "English to Runyankole":
            src_lang_token = language_tokens['eng']
            tgt_lang_token = language_tokens['nyn']
        else:
            src_lang_token = language_tokens['nyn']
            tgt_lang_token = language_tokens['eng']

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs['input_ids'] = src_lang_token

        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_token,
            decoder_start_token_id=tgt_lang_token,
            max_length=256,
            num_beams=4,      
            do_sample=False
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return "No output generated."
        
    except Exception as e:
        return f"Translation Processing Error: {str(e)}"

# 3. Sidebar Selection Options
direction = st.sidebar.selectbox("Flow Mode:", ["English to Runyankole", "Runyankole to English"])

# 4. Handle Persistent Values
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# Audio Recorder Layout Layer
recorded_audio = st.audio_input("Record voice input")
if recorded_audio is not None:
    with st.spinner("Decoding speech audio variables..."):
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(recorded_audio.read()))
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            result = asr_pipeline({"raw": audio_data, "sampling_rate": sample_rate})
            st.session_state.voice_text = result["text"]
        except Exception as e:
            st.error(f"Audio Decode Error: {e}")

# Form submission layout block
with st.form("translation_form", clear_on_submit=False):
    sentence = st.text_input("Input Phrase:", value=st.session_state.voice_text)
    submit_button = st.form_submit_button(label="Translate Text", type="primary")

if submit_button and sentence:
    with st.spinner("Processing Model Inference Tensors..."):
        output_translation = translate_via_neural_net(sentence, direction)
        st.subheader("Neural Network Result:")
        st.success(output_translation)
