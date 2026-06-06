import streamlit as st
import torch
import io
import soundfile as sf
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# 1. App Window Initialization
st.set_page_config(page_title="Runyankole Neural App", layout="centered")
st.title("🇺🇬 True Runyankole AI Translator")
st.markdown("---")


# 2. Optimized Pipeline / Engine Caching
@st.cache_resource
def load_speech_models():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


@st.cache_resource
def load_translation_pipeline():
    model_name = "facebook/nllb-200-distilled-600M"
    
    # Load foundational components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = 0 if torch.cuda.is_available() else -1
    
    # FIX: Initialize the native translation pipeline to handle internal NLLB language tagging seamlessly
    translation_pipe = pipeline(
        "translation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )
    return translation_pipe


# Instantiate Models
asr_pipeline = load_speech_models()
nllb_translator = load_translation_pipeline()

# Standard BCP-47 Target Codes utilized by Meta NLLB-200
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

        # FIX: Let the pipeline execute translation and handle source/target language switches internally
        pipe_output = nllb_translator(
            text, 
            src_lang=src_lang, 
            tgt_lang=tgt_lang, 
            max_length=256
        )
        
        # Safely extract translation text string from list dictionary response
        return pipe_output[0]['translation_text']
        
    except Exception as e:
        return f"Translation Subsystem Error Trace: {str(e)}"


# 3. Mode Configuration UI
direction = st.sidebar.selectbox("Flow Mode:", ["English to Runyankole", "Runyankole to English"])

# Initialize memory space for voice text inputs
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# 4. Audio Processing Workflow Block
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

# 5. Direct State Binding Layout (Form Block Removed to Prevent State Freezing)
sentence = st.text_input("Input Phrase:", key="voice_text")
submit_button = st.button(label="Translate Text", type="primary")

# Execute conversion on button confirmation state matching
if submit_button and sentence:
    with st.spinner("Running Neural Net Inference..."):
        output_translation = translate_via_neural_net(sentence, direction)
        st.subheader("Neural Network Result:")
        st.success(output_translation)
