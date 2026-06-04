import torch
import streamlit as st
import io
import soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Main Page Setup
st.set_page_config(page_title="Runyankole AI App", layout="centered")
st.title("🇺🇬 True Runyankole AI Translator")
st.write("Powered by an open-access, cloud-stable translation engine.")
st.markdown("---")

# 2. Public Local AI Model Pipelines (No Login Tokens Required)
@st.cache_resource
def load_speech_models():
    """Loads OpenAI's lightweight public speech transcriber."""
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

@st.cache_resource
def load_translation_engine():
    """Loads Meta's public open-access translation engine."""
    # Using Facebook's public 600M model which doesn't require a gated Hugging Face account login
    model_name = "facebook/nllb-200-distilled-600M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
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

# Meta NLLB explicit 4-digit script language country codes
language_codes = {
    'eng': 'eng_Latn',  # English
    'nyn': 'nyn_Latn'   # Runyankole / Nyankole dialect script mapping
}

def translate_via_neural_net(text, direction_mode):
    try:
        if direction_mode == "English to Runyankole":
            src_lang = language_codes['eng']
            tgt_lang = language_codes['nyn']
        else:
            src_lang = language_codes['nyn']
            tgt_lang = language_codes['eng']

        # Tokenize text string into standard token sequences
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Look up correct forced target language token identification ID
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        # Generate translation sentences matching language constraints
        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=1,      # Greedy search decoding structure for rapid speed
            do_sample=False
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return result[0] if isinstance(result, list) and len(result) > 0 else "No output generated."
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

# Form submission layout block to fix typing lag issues
with st.form("translation_form", clear_on_submit=False):
    sentence = st.text_input("Input Phrase:", value=st.session_state.voice_text)
    submit_button = st.form_submit_button(label="Translate Text", type="primary")

# Execute conversion algorithms exclusively on submission triggers
if submit_button and sentence:
    with st.spinner("Processing Model Inference Tensors..."):
        output_translation = translate_via_neural_net(sentence, direction)
        st.subheader("Neural Network Result:")
        st.success(output_translation)
