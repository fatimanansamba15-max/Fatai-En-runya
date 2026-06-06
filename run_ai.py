import streamlit as st
import torch
import io
import soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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
    # Official NLLB Base Model (Supports Runyankole via nyn_Latn)
    model_name = "facebook/nllb-200-distilled-600M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return tokenizer, model, device


# Initialize Neural Components
asr_pipeline = load_speech_models()
tokenizer, translation_model, device = load_translation_engine()

# Map language ids using standard NLLB BCP-47 string identifiers 
language_tokens = {
    'eng': 'eng_Latn', 
    'nyn': 'nyn_Latn'
}

def translate_via_neural_net(text, direction_mode):
    try:
        if direction_mode == "English to Runyankole":
            src_lang = language_tokens['eng']
            tgt_lang = language_tokens['nyn']
        else:
            src_lang = language_tokens['nyn']
            tgt_lang = language_tokens['eng']

        inputs = tokenizer(text, return_tensors="pt", src_lang=src_lang).to(device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=1,      
            do_sample=False   
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        # FIX: Explicitly pull index 0 to return a clean string, not a list object
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
            # Directly updates the text field because of the key binding
            st.session_state.voice_text = result["text"]
        except Exception as e:
            st.error(f"Audio Decode Error: {e}")

# Layout without form block to allow instant component interaction
sentence = st.text_input("Input Phrase:", key="voice_text")
submit_button = st.button(label="Translate Text", type="primary")

# Execute translation on the first click
if submit_button and sentence:
    with st.spinner("Running Neural Net Inference..."):
        output_translation = translate_via_neural_net(sentence, direction)
        st.subheader("Neural Network Result:")
        st.success(output_translation)
