import os
import base64
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from tf_keras.models import Sequential, load_model as keras_load_model
from tf_keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from tf_keras.optimizers import Adam

st.set_page_config(
    page_title="Emotion Recognition UI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_weights.h5")
IMG_SIZE = 48
CLASS_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
EMOJI_MAP = {
    "angry": "😠",
    "disgusted": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😮",
}
BADGE_CLASS = {
    "angry": "angry",
    "disgusted": "disgusted",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
}

css = r"""
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
  --bg-1: #0d0520;
  --bg-2: #1a0a38;
  --card: rgba(14, 8, 40, 0.93);
  --accent: #9b5dff;
  --accent-soft: rgba(155, 93, 255, 0.18);
  --text: #eef1ff;
  --muted: rgba(226, 231, 255, 0.72);
  --shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
}

html, body {
  min-height: 100%;
  margin: 0;
  font-family: 'Poppins', sans-serif;
  color: var(--text);
  background:
    radial-gradient(ellipse at 15% 20%, rgba(120, 40, 220, 0.38) 0%, transparent 45%),
    radial-gradient(ellipse at 85% 80%, rgba(80, 0, 180, 0.32) 0%, transparent 40%),
    radial-gradient(ellipse at 50% 50%, rgba(60, 0, 130, 0.22) 0%, transparent 60%),
    linear-gradient(160deg, #0d0520 0%, #180840 35%, #0f0630 65%, #08021a 100%);
  overflow-x: hidden;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  background:
    radial-gradient(circle at 30% 40%, rgba(160, 60, 255, 0.18) 0%, transparent 55%),
    radial-gradient(circle at 75% 20%, rgba(100, 20, 200, 0.20) 0%, transparent 40%),
    radial-gradient(circle at 60% 85%, rgba(130, 0, 220, 0.15) 0%, transparent 45%);
  pointer-events: none;
  animation: pulse-glow 8s ease-in-out infinite alternate;
}

body::after {
  content: "";
  position: fixed;
  inset: 0;
  background: linear-gradient(120deg, rgba(100, 0, 160, 0.10), rgba(60, 0, 120, 0.08), rgba(140, 50, 240, 0.07));
  pointer-events: none;
  animation: drift 22s linear infinite;
}

@keyframes drift {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(10px, -14px) scale(1.01); }
  66% { transform: translate(-8px, 10px) scale(0.99); }
}

@keyframes pulse-glow {
  0% { opacity: 0.6; }
  100% { opacity: 1; }
}

.stApp, .streamlit-container, .main {
  min-height: 100vh;
  padding: 52px 24px 72px;
}

.app-shell {
  max-width: 1120px;
  margin: 0 auto;
  display: grid;
  gap: 30px;
  grid-template-columns: minmax(320px, 1fr);
}

.card {
  position: relative;
  border-radius: 22px;
  background: rgba(16, 20, 45, 0.92);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
  overflow: hidden;
  transition: transform 0.35s ease, border-color 0.35s ease;
}

.card:hover {
  transform: translateY(-2px);
  border-color: rgba(127, 91, 255, 0.22);
}

.hero-card, .upload-card, .result-card {
  padding: 34px;
}

.hero-card {
  width: 100%;
  margin: 0 auto 28px;
  min-height: 75vh;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 48px 36px;
}

.hero-card .hero-pill {
  width: auto;
  padding: 14px 28px;
  font-size: 1rem;
}

.hero-card h1 {
  margin: 0;
  font-size: clamp(3.4rem, 5vw, 5.6rem);
  line-height: 1.01;
  letter-spacing: -0.05em;
  color: #f9fbff;
}

.hero-card .subtitle {
  margin-top: 18px;
  color: rgba(235, 241, 255, 0.72);
  font-size: 1.05rem;
  max-width: 760px;
}

.hero-card p {
  margin: 0;
  max-width: 760px;
  color: rgba(235, 241, 255, 0.78);
  font-size: 1.05rem;
  line-height: 1.8;
}

.hero-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 18px;
  border-radius: 999px;
  background: rgba(127, 91, 255, 0.16);
  color: #e7e9ff;
  font-size: 0.95rem;
  border: 1px solid rgba(127, 91, 255, 0.18);
}

.upload-zone {
  min-height: 250px;
  border-radius: 24px;
  border: 2px dashed rgba(127, 91, 255, 0.45);
  padding: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 14px;
  text-align: center;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  transition: all 0.34s ease;
}

.upload-zone:hover {
  border-color: rgba(3, 207, 255, 0.88);
  box-shadow: 0 0 0 1px rgba(3, 207, 255, 0.16), 0 30px 80px rgba(0, 0, 0, 0.18);
  transform: translateY(-1px);
}

.upload-zone strong {
  margin: 0;
  font-size: 1.1rem;
  color: #f7f9ff;
}

.upload-zone span {
  color: rgba(234, 241, 255, 0.72);
  max-width: 420px;
  line-height: 1.7;
}

.upload-action {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  color: rgba(179, 191, 255, 0.88);
  font-size: 0.98rem;
}

.upload-action svg {
  width: 24px;
  height: 24px;
  fill: #7f5bff;
}

.upload-help {
  margin-top: 18px;
  color: rgba(235, 241, 255, 0.72);
  font-size: 0.98rem;
}

.upload-card {
  padding-bottom: 62px;
}

.upload-card + .stFileUploader {
  margin-top: 24px !important;
  padding: 0 !important;
}

.upload-card + .stFileUploader > div {
  width: 100% !important;
  background: transparent !important;
  border: none !important;
}

.upload-card + .stFileUploader > div > div {
  background: rgba(255,255,255,0.04) !important;
  border-radius: 20px !important;
  border: 1px dashed rgba(127, 91, 255, 0.32) !important;
}

.stFileUploader button {
  width: 100% !important;
  border-radius: 18px !important;
  padding: 18px 22px !important;
}

.stFileUploader {
  width: 100% !important;
  margin-top: -40px !important;
}

.stFileUploader > div {
  width: 100% !important;
}

.stFileUploader button {
  width: 100% !important;
  border-radius: 18px !important;
  padding: 18px 22px !important;
  background: linear-gradient(135deg, rgba(127, 91, 255, 0.98), rgba(48, 136, 255, 0.95)) !important;
  color: #ffffff !important;
  border: none !important;
  box-shadow: 0 18px 42px rgba(73, 82, 178, 0.24) !important;
  font-weight: 700 !important;
}

.stButton button {
  width: 100% !important;
  padding: 18px 24px !important;
  border-radius: 18px !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.02em !important;
  background: linear-gradient(135deg, #7f5bff, #33c7ff) !important;
  border: 1px solid rgba(255, 255, 255, 0.14) !important;
  color: #fff !important;
  box-shadow: 0 20px 50px rgba(15, 37, 86, 0.28) !important;
}

.stButton button:hover {
  transform: translateY(-1px);
}

.predict-card {
  display: grid;
  gap: 20px;
}

.result-card h2 {
  margin: 0;
  font-size: 1.5rem;
  color: #f7f8ff;
}

.result-list {
  display: grid;
  gap: 16px;
}

.result-item {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 18px;
  padding: 20px 22px;
  border-radius: 20px;
  background: rgba(26, 33, 68, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.07);
  transition: transform 0.25s ease, border-color 0.25s ease;
}

.result-item:hover {
  transform: translateY(-1px);
  border-color: rgba(127, 91, 255, 0.18);
}

.result-icon {
  display: grid;
  place-items: center;
  width: 52px;
  height: 52px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.04);
  font-size: 1.6rem;
}

.result-label {
  font-size: 1rem;
  font-weight: 600;
  color: #eef1ff;
}

.result-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 12px 18px;
  border-radius: 999px;
  font-size: 0.95rem;
  font-weight: 600;
  min-width: 126px;
  text-align: center;
  color: #fff;
}

.result-badge.happy {
  background: rgba(120, 225, 118, 0.18);
  color: #b8ffb0;
}

.result-badge.sad {
  background: rgba(90, 150, 255, 0.18);
  color: #c6e7ff;
}

.result-badge.angry {
  background: rgba(255, 112, 112, 0.18);
  color: #ffb4b4;
}

.result-badge.fearful {
  background: rgba(255, 190, 95, 0.18);
  color: #ffe4a3;
}

.result-badge.surprised {
  background: rgba(80, 220, 255, 0.18);
  color: #bdf0ff;
}

.result-badge.disgusted {
  background: rgba(155, 105, 255, 0.18);
  color: #dbbdfb;
}

.result-badge.neutral {
  background: rgba(190, 192, 255, 0.14);
  color: #dde2ff;
}

.loader-shell {
  position: relative;
  border-radius: 22px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
  min-height: 92px;
}

.loader-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent);
  transform: translateX(-100%);
  animation: shimmer 1.6s infinite;
}

.loader-shell p {
  position: relative;
  z-index: 1;
  margin: 0;
  padding: 30px 24px;
  color: rgba(235, 242, 255, 0.78);
  font-size: 1rem;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.glossy-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 200px;
  padding: 14px 32px;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(127, 91, 255, 0.98), rgba(48, 136, 255, 0.95));
  border: 1px solid rgba(255,255,255,0.18);
  color: #fff !important;
  font-weight: 700;
  text-decoration: none;
  box-shadow: 0 18px 42px rgba(73, 82, 178, 0.24);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.glossy-pill:link,
.glossy-pill:visited,
.glossy-pill {
  color: #fff !important;
}

@keyframes fadeInUp {
  0% { opacity: 0; transform: translateY(18px); }
  100% { opacity: 1; transform: translateY(0); }
}

.emoji-bg {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}

.emoji {
  position: absolute;
  opacity: 0;
  animation: float-up linear infinite;
  user-select: none;
  filter: drop-shadow(0 0 6px rgba(180, 120, 255, 0.4));
}

/* --- size tiers --- */
.emoji.sm  { font-size: 1.4rem; }
.emoji.md  { font-size: 2.2rem; }
.emoji.lg  { font-size: 3.2rem; }

/* --- individual positions, durations, delays --- */
.emoji:nth-child(1)  { left:  3%; animation-duration: 14s; animation-delay:  0s; }
.emoji:nth-child(2)  { left:  9%; animation-duration: 18s; animation-delay:  3s; }
.emoji:nth-child(3)  { left: 15%; animation-duration: 12s; animation-delay:  6s; }
.emoji:nth-child(4)  { left: 22%; animation-duration: 20s; animation-delay:  1s; }
.emoji:nth-child(5)  { left: 28%; animation-duration: 16s; animation-delay:  9s; }
.emoji:nth-child(6)  { left: 34%; animation-duration: 13s; animation-delay:  4s; }
.emoji:nth-child(7)  { left: 40%; animation-duration: 19s; animation-delay:  7s; }
.emoji:nth-child(8)  { left: 46%; animation-duration: 15s; animation-delay:  2s; }
.emoji:nth-child(9)  { left: 52%; animation-duration: 17s; animation-delay: 11s; }
.emoji:nth-child(10) { left: 58%; animation-duration: 11s; animation-delay:  5s; }
.emoji:nth-child(11) { left: 64%; animation-duration: 22s; animation-delay:  8s; }
.emoji:nth-child(12) { left: 70%; animation-duration: 14s; animation-delay:  0s; }
.emoji:nth-child(13) { left: 76%; animation-duration: 18s; animation-delay: 13s; }
.emoji:nth-child(14) { left: 82%; animation-duration: 16s; animation-delay:  3s; }
.emoji:nth-child(15) { left: 88%; animation-duration: 20s; animation-delay:  6s; }
.emoji:nth-child(16) { left: 93%; animation-duration: 13s; animation-delay: 10s; }
.emoji:nth-child(17) { left:  6%; animation-duration: 25s; animation-delay: 14s; }
.emoji:nth-child(18) { left: 18%; animation-duration: 21s; animation-delay:  2s; }
.emoji:nth-child(19) { left: 36%; animation-duration: 17s; animation-delay: 16s; }
.emoji:nth-child(20) { left: 48%; animation-duration: 23s; animation-delay:  0s; }
.emoji:nth-child(21) { left: 62%; animation-duration: 15s; animation-delay: 12s; }
.emoji:nth-child(22) { left: 74%; animation-duration: 19s; animation-delay:  5s; }
.emoji:nth-child(23) { left: 86%; animation-duration: 24s; animation-delay:  8s; }
.emoji:nth-child(24) { left: 25%; animation-duration: 12s; animation-delay: 18s; }
.emoji:nth-child(25) { left: 55%; animation-duration: 16s; animation-delay:  4s; }
.emoji:nth-child(26) { left: 79%; animation-duration: 20s; animation-delay: 15s; }
.emoji:nth-child(27) { left: 44%; animation-duration: 13s; animation-delay:  7s; }
.emoji:nth-child(28) { left: 97%; animation-duration: 18s; animation-delay:  1s; }

@keyframes float-up {
  0%   { transform: translateY(110vh) rotate(0deg)   scale(0.7); opacity: 0; }
  8%   { opacity: 0.18; }
  50%  { transform: translateY(50vh)  rotate(180deg) scale(1.05); opacity: 0.22; }
  92%  { opacity: 0.14; }
  100% { transform: translateY(-15vh) rotate(360deg) scale(0.8); opacity: 0; }
}

@media (min-width: 980px) {
  .app-shell { grid-template-columns: 1.03fr 0.97fr; }
}

@media (max-width: 720px) {
  .stApp, .streamlit-container, .main { padding: 26px 16px 56px; }
  .hero-card, .upload-card, .result-card { padding: 26px; border-radius: 20px; }
  .result-item { grid-template-columns: 1fr; text-align: left; }
}
"""

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model_weights():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        Conv2D(22, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(22, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(96, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(96, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(len(CLASS_LABELS), activation="softmax"),
    ])
    optimiser = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
    else:
        raise FileNotFoundError(f'Model weights not found at {MODEL_PATH}')

    return model


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image, array


def build_prediction_html(predicted_label, score, probabilities, report_b64):
    items_html = ''
    for label, prob in zip(CLASS_LABELS, probabilities):
        emoji = EMOJI_MAP.get(label, '')
        badge_class = BADGE_CLASS.get(label, 'neutral')
        items_html += (
            "<div class='result-item'>"
            f"<div class='result-icon'>{emoji}</div>"
            "<div>"
            f"<div class='result-label'>{label.title()}</div>"
            f"<div style='color: rgba(235, 241, 255, 0.68); font-size: 0.94rem;'>{prob:.1%} confidence</div>"
            "</div>"
            f"<div class='result-badge {badge_class}'>{emoji} {label.title()}</div>"
            "</div>"
        )

    html = (
        "<div class='card result-card fade-in'>"
        "<h2>Prediction preview</h2>"
        f"<p style='color: rgba(235, 241, 255, 0.76); margin-top: 12px;'>Detected emotion: <strong>{EMOJI_MAP[predicted_label]} {predicted_label.title()}</strong> with {score:.1%} confidence.</p>"
        f"<div class='result-list'>{items_html}</div>"
        f"<div style='margin-top: 24px; text-align: right;'><a class='glossy-pill' download='emotion_report.txt' href='data:text/plain;base64,{report_b64}'>Download Report</a></div>"
        "</div>"
    )
    return html


def build_placeholder_html():
    html = "<div class='card result-card fade-in'>"
    html += "<h2>Prediction preview</h2>"
    html += "<p style='color: rgba(235, 241, 255, 0.76); margin-top: 12px;'>Upload an image and click Predict Emotion to see the model in action.</p>"
    html += "<div class='loader-shell'><p>Waiting for image upload and model prediction...</p></div>"
    html += "<div style='margin-top: 24px;'><a class='glossy-pill' target='_blank'>Download Report</a></div>"
    html += "</div>"
    return html

st.markdown("""
    <div class="emoji-bg">
      <span class="emoji md">😊</span>
      <span class="emoji sm">😢</span>
      <span class="emoji lg">😠</span>
      <span class="emoji md">😨</span>
      <span class="emoji sm">🤢</span>
      <span class="emoji lg">😮</span>
      <span class="emoji md">😐</span>
      <span class="emoji sm">😊</span>
      <span class="emoji lg">😢</span>
      <span class="emoji md">😠</span>
      <span class="emoji sm">😨</span>
      <span class="emoji lg">🤢</span>
      <span class="emoji md">😮</span>
      <span class="emoji sm">😐</span>
      <span class="emoji lg">😊</span>
      <span class="emoji md">😢</span>
      <span class="emoji sm">😠</span>
      <span class="emoji lg">😨</span>
      <span class="emoji md">🤢</span>
      <span class="emoji sm">😮</span>
      <span class="emoji lg">😐</span>
      <span class="emoji md">😊</span>
      <span class="emoji sm">😢</span>
      <span class="emoji lg">😠</span>
      <span class="emoji md">😨</span>
      <span class="emoji sm">🤢</span>
      <span class="emoji lg">😮</span>
      <span class="emoji md">😐</span>
    </div>
    <div class="hero">
      <h1>✦ Emotion recognition</h1>
      <p>Upload an image of a face to detect emotions.</p>
      <p style="margin-top: 18px; color: rgba(235, 241, 255, 0.72); max-width: 760px; font-size: 1.05rem; line-height: 1.7;">Choose a photo to see how the model interprets facial expressions in real time.</p>
    </div>
    """, unsafe_allow_html=True)
# st.markdown("""
# <div class='card upload-card fade-in'>
#   <div class='hero-pill'>Emotion recognition</div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown("""
#     <div class="hero" style="margin-bottom: 32px; text-align: center;">
#       <h1>✦ Emotion recognition</h1>
#       <p>Upload a PDF or paste your text and get an AI-powered summary
#          generated by a fine-tuned BART model.</p>
#     </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class='card upload-card fade-in'>
#   <div class='upload-zone'>
#     <strong>Drag & drop your image here</strong>
#     <span>Or click to choose a photo of a face for emotion detection.</span>
#     <div class='upload-action'>
#       <svg viewBox='0 0 24 24'><path d='M12 3v10m0 0l-4-4m4 4l4-4'/></svg>
#       Drop inside the card or use the button below
#     </div>
#     <p class='upload-help'>Use the widget below to upload your file.</p>
#   </div>
# </div>
# """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Selected image", use_container_width=True)

predict_button = st.button("Predict Emotion", use_container_width=True)

if uploaded_file is None:
    st.markdown(build_placeholder_html(), unsafe_allow_html=True)
else:
    if predict_button:
        try:
            model = load_model_weights()
            image_preview, image_array = preprocess_image(uploaded_file)
            with st.spinner("Predicting emotion..."):
                prediction = model.predict(image_array, verbose=0)[0]
            predicted_idx = int(np.argmax(prediction))
            predicted_label = CLASS_LABELS[predicted_idx]
            predicted_score = float(prediction[predicted_idx])
            report_text = (
                f"Emotion Recognition Report\n"
                f"==========================\n"
                f"Predicted emotion: {predicted_label.title()}\n"
                f"Confidence: {predicted_score:.2%}\n\n"
                f"Full distribution:\n"
            )
            for label, prob in zip(CLASS_LABELS, prediction):
                report_text += f"- {label.title()}: {prob:.2%}\n"
            report_b64 = base64.b64encode(report_text.encode('utf-8')).decode('utf-8')
            html = build_prediction_html(predicted_label, predicted_score, prediction, report_b64)
            st.markdown(html, unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Model error: {exc}")
    else:
        st.markdown("""
        <div class='card result-card fade-in'>
          <h2>Prediction preview</h2>
          <p style='color: rgba(235, 241, 255, 0.76); margin-top: 12px;'>Ready to predict. Click the button below after uploading an image.</p>
        </div>
        """, unsafe_allow_html=True)
