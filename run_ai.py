import torch
import streamlit as st
import io
import soundfile as sf
import transformers
from transformers import AutoModelForSeq2SeqLM, pipeline

# 1. Main Page Setup
st.set_page_config(page_title="Runyankole AI App", layout="centered")
st.title("🇺🇬 True Runyankole AI Translator")
st.write("Powered by an open-access, regional fine-tuned translation engine.")
st.markdown("---")

# 2. Public Local AI Model Pipelines
@st.cache_resource
def load_speech_models():
    """Loads OpenAI's lightweight public speech transcriber."""
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

@st.cache_resource
def load_translation_engine():
    """
    Loads Sunbird AI's specialized NLLB adaptation.
    Uses dedicated NllbTokenizer and conditional generation classes
    as required by the model's architectural custom vocabulary.
    """
    model_name = "Sunbird/translate-nllb-1.3b-salt"
    
    # Use exact NllbTokenizer class to preserve index integrity
    tokenizer = transformers.NllbTokenizer.from_pretrained(model_name)
    model = transformers.M2M100ForConditionalGeneration.from_pretrained(model_name)
    
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

        # 1. Tokenize text input
        inputs = tokenizer(text, return_tensors="pt")
        
        # 2. Safely push entire dictionary tensors to active device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 3. Explicitly overwrite source sequence start identifier on the correct device
        inputs['input_ids'][0][0] = src_lang_token

        # 4. Generate translation with architecture-compatible decoder keys
        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_token,
            decoder_start_token_id=tgt_lang_token, # Prevents generation failure/loops
            max_length=256,
            num_beams=4,      # Kept for accurate contextual local phrasing
            do_sample=False
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        # Extract individual text string elements safely from list return arrays
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
