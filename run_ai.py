from transformers import AutoTokenizer, M2M100ForConditionalGeneration
import torch
import streamlit as st
import io
import soundfile as sf
import transformers
from transformers import pipeline, NllbTokenizer, M2M100ForConditionalGeneration

# 1. Main Page Setup
st.set_page_config(page_title="Runyankole Neural App", layout="centered")
st.title("🇺🇬 True Runyankole AI Translator")
st.markdown("---")


# 2. Cached Deep Learning Engine Setup
@st.cache_resource
def load_speech_models():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


@st.cache_resource
def load_translation_engine():
    # Official Sunbird AI Model Repository
    model_name = "facebook/nllb-200-distilled-600M"
def load_translation_engine():
    # Line 24 (Make sure your tokenizer line is here)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Line 25 (Make sure this has the EXACT same number of leading spaces as line 24!)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device


# Initialize Tensors
asr_pipeline = load_speech_models()
tokenizer, translation_model, device = load_translation_engine()

# Map language ids according to official SALT documentation rules
language_tokens = {'eng': 256047, 'nyn': 256002}


def translate_via_neural_net(text, direction_mode):
    try:
        if direction_mode == "English to Runyankole":
            src_lang, tgt_token = 'eng', language_tokens['nyn']
        else:
            src_lang, tgt_token = 'nyn', language_tokens['eng']

        # Convert text input directly to a tensor matrix grid
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # FIX: Modify ONLY the first column entry of the tensor matrix, keeping the rest of your text intact
        inputs['input_ids'][:, 0] = language_tokens[src_lang]

        # SPEED ACCELERATOR: Added generation constraints to bypass slow calculations
        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=tgt_token,
            max_length=256,
            num_beams=1,      # Forces Greedy Search for instant text processing
            do_sample=False   # Disables sampling overhead latency
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return result[0] if isinstance(result, list) and len(result) > 0 else "No output generated."
    except Exception as e:
        return f"Translation Error Matrix Trace: {str(e)}"


# 3. Sidebar Selection Options
direction = st.sidebar.selectbox("Flow Mode:", ["English to Runyankole", "Runyankole to English"])

# 4. Handle Persistent Values
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# Audio Recorder
recorded_audio = st.audio_input("Record voice input")
if recorded_audio is not None:
    with st.spinner("Processing speech audio layer..."):
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(recorded_audio.read()))
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            result = asr_pipeline({"raw": audio_data, "sampling_rate": sample_rate})
            st.session_state.voice_text = result["text"]
        except Exception as e:
            st.error(f"Audio Decode Error: {e}")

# BROWSER LAG FIX: Streamlit Form Layout Block
with st.form("translation_form", clear_on_submit=False):
    sentence = st.text_input("Input Phrase:", value=st.session_state.voice_text)
    submit_button = st.form_submit_button(label="Translate Text", type="primary")

# Execute translation ONLY when the user clicks the submit button
if submit_button and sentence:
    with st.spinner("Running Neural Net Inference..."):
        output_translation = translate_via_neural_net(sentence, direction)
        st.subheader("Neural Network Result:")
        st.success(output_translation)
