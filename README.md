<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&pause=1000&color=9B59B6&center=true&vCenter=true&width=600&lines=Emotion+Recognition+System;Deep+Learning+%7C+CNN+%7C+Keras" alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/Contributors-3-6A0DAD?style=for-the-badge&logo=github&logoColor=white" />
<img src="https://img.shields.io/badge/Model-CNN-7D3C98?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Emotions-7%20Classes-8E44AD?style=for-the-badge&logo=keras&logoColor=white" />
<img src="https://img.shields.io/badge/Framework-Streamlit-9B59B6?style=for-the-badge&logo=streamlit&logoColor=white" />

<br/><br/>

> 🎭 **A deep learning system that detects and classifies human facial emotions in real time using a Convolutional Neural Network (CNN) built with Keras.**

</div>

---

<details open>
  <summary>🔮 Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#emotion-classes">Emotion Classes</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#run-locally">Run Locally</a></li>
    <li><a href="#deploy-to-streamlit-community-cloud">Deploy to Streamlit Community Cloud</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>

---

## 🧠 About The Project

This project is a **facial emotion recognition system** developed as part of a Machine Learning and Artificial Intelligence course. The goal is to automatically detect and classify the emotional state of a person from a facial image.

The system processes input images through a **Convolutional Neural Network (CNN)** pipeline, extracting spatial features from the face and outputting a probability distribution over **7 distinct emotion categories**. The model was iteratively trained and evaluated using well-known performance metrics including accuracy, loss curves, and confusion matrices.

The project is deployed as an interactive **Streamlit** web application, allowing users to upload images and receive instant emotion predictions — all without needing to retrain the model, thanks to the included pre-trained weights (`model_weights.h5`).

---

## 🏗️ Model Architecture

The core of this project is a **Convolutional Neural Network (CNN)** implemented using the **Keras API** (TensorFlow backend). CNNs are the state-of-the-art architecture for image classification tasks due to their ability to learn hierarchical spatial features directly from raw pixel data.

### Architecture Overview

```
Input Image (48×48 grayscale)
        ↓
┌─────────────────────┐
│  Conv2D + BatchNorm │  ← Feature extraction layers
│  Conv2D + BatchNorm │
│  MaxPooling + Dropout│
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Conv2D + BatchNorm │  ← Deeper feature maps
│  Conv2D + BatchNorm │
│  MaxPooling + Dropout│
└─────────────────────┘
        ↓
┌─────────────────────┐
│    Flatten Layer    │
│  Dense (FC) Layers  │  ← Classification head
│  Dropout + Dense    │
└─────────────────────┘
        ↓
  Softmax Output (7 classes)
```

### Key Design Choices

| Component | Detail |
|-----------|--------|
| **Architecture** | Multi-layer CNN |
| **Activation** | ReLU (hidden), Softmax (output) |
| **Regularization** | Dropout + Batch Normalization |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Cross-Entropy |
| **Input Size** | 48 × 48 pixels (grayscale) |
| **Output** | 7-class probability distribution |
| **Weights** | Saved as `model_weights.h5` |

---

## 🎭 Emotion Classes

The model classifies facial images into **7 emotion categories**:

| # | Emotion | Description |
|---|---------|-------------|
| 1 | 😡 **Angry** | Expressions of frustration or anger |
| 2 | 🤢 **Disgust** | Expressions of revulsion |
| 3 | 😨 **Fear** | Expressions of fear or anxiety |
| 4 | 😊 **Happy** | Joyful and smiling expressions |
| 5 | 😐 **Neutral** | Calm, expressionless faces |
| 6 | 😢 **Sad** | Sorrowful or downcast expressions |
| 7 | 😲 **Surprise** | Expressions of surprise or shock |

---

## 🛠️ Built With

<div align="center">

[![Keras](https://img.shields.io/badge/Keras-D10000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F89A36?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pillow](https://img.shields.io/badge/Pillow-FFD43B?style=for-the-badge&logo=python&logoColor=black)](https://python-pillow.org/)

</div>

---

## 📦 Dataset

The model was trained on a modified version of the **FER (Facial Emotion Recognition)** dataset available on [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer).

| Property | Value |
|----------|-------|
| **Total Samples** | 35,887 images |
| **Classes** | 7 emotion categories |
| **Image Format** | 48 × 48 pixels, grayscale |
| **Split** | Train / Validation / Test |

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/somiaamari/Emotion-Recognition.git
cd Emotion-Recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

### 4. (Optional) Explore the notebook

```bash
jupyter notebook emotion_recognition.ipynb
```

> ✅ The pre-trained weights (`model_weights.h5`) are included — no retraining required!

---

## ☁️ Deploy to Streamlit Community Cloud

The app entry point is `app.py`. Streamlit Community Cloud installs the pinned
dependencies from `requirements.txt`; `runtime.txt` selects Python 3.11. The
pre-trained `model_weights.h5` file must remain in the repository root because
the app loads it when the first prediction is requested.

1. Push this repository to GitHub, including `app.py`, `requirements.txt`,
   `runtime.txt`, and `model_weights.h5`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and
   choose **Create app**.
3. Select this repository and the branch to deploy, then set the main file path
   to `app.py`.
4. In **Advanced settings**, select Python 3.11 if a runtime version is
   requested, then deploy.

No Streamlit secrets or external services are required. To run the same pinned
environment locally, install `requirements.txt` using Python 3.11 before
running `streamlit run app.py`.

---

## 👩‍💻 Contributors

<div align="center">

<table>
  <tr>
    <td align="center">
      <b>🟣 Amari Soumia</b><br/>
      <sub>Deep Learning & Model Development</sub>
    </td>
    <td align="center">
      <b>🟣 Berrahou Meriem</b><br/>
      <sub>Data Preprocessing & Evaluation</sub>
    </td>
    <td align="center">
      <b>🟣 Mehda Nessrin</b><br/>
      <sub>Application Development & Deployment</sub>
    </td>
  </tr>
</table>

<br/>

*This project was developed as part of a Machine Learning & AI course.*

</div>

---

<div align="center">

<sub>Built with 💜 using Keras & Streamlit</sub>

</div>
