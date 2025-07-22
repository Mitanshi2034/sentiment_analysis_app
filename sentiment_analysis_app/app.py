# Streamlit App for Sentiment, Emotion, and Sarcasm Analysis

# 📘 Streamlit App for Sentiment, Emotion & Sarcasm Detection with Best Model

import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
from transformers import pipeline

# Page Title
st.set_page_config(page_title="Sentiment Dashboard", layout="centered")
st.title("🧠 Sentiment, Emotion & Sarcasm Detector")

# Load best model info from Excel
try:
    df = pd.read_csv("model_comparison_results.csv")
    best_model_row = df.sort_values("F1-Score", ascending=False).iloc[0]
    best_model_name = best_model_row["Model"].lower().replace("-", "")  # cnn / lstm / bilstm
    st.success(f"🏆 Best Model: {best_model_row['Model']} (F1: {best_model_row['F1-Score']:.2f})")
except Exception as e:
    st.error(f"Error loading comparison file: {e}")
    st.stop()

# Load model and tokenizer
model_path = f"models/best_model_{best_model_name}.h5"
tokenizer_path = f"models/tokenizer_{best_model_name}.pkl"

try:
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
except Exception as e:
    st.error(f"❌ Failed to load model/tokenizer: {e}")
    st.stop()

# Load emotion and sarcasm detection transformers
@st.cache_resource(show_spinner=False)
def load_transformer_models():
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    sarcasm_model = pipeline("text-classification", model="lxyuan/sarcasm-detection-roberta")
    return emotion_model, sarcasm_model

emotion_model, sarcasm_model = load_transformer_models()

# Input field
user_input = st.text_area("Enter a tweet or product review:")

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):

            # ✅ Sentiment Prediction using trained model
            seq = tokenizer.texts_to_sequences([user_input])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=128, padding='post')
            prediction = model.predict(padded)[0][0]
            sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
            st.subheader("📊 Sentiment Prediction")
            st.write(f"*Result:* {sentiment}  |  *Confidence:* {prediction:.2f}")

            # 🎭 Emotion Detection
            st.subheader("🎭 Emotion Detection")
            emotion_scores = emotion_model(user_input)[0]
            sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
            for emo in sorted_emotions:
                st.write(f"{emo['label']}: {emo['score']:.2f}")
            st.bar_chart({e['label']: e['score'] for e in sorted_emotions})

            # 😏 Sarcasm Detection
            st.subheader("😏 Sarcasm Detection")
            sarcasm = sarcasm_model(user_input)[0]
            sarcastic_label = sarcasm['label']
            sarcastic = "Yes" if sarcastic_label == "LABEL_1" else "No"
            st.write(f"*Sarcastic?* {sarcastic}  |  *Confidence:* {sarcasm['score']:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with 💡 using CNN/LSTM + RoBERTa + Streamlit")