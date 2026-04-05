import streamlit as st
import torch
import tensorflow as tf
from PIL import Image
import numpy as np
import timm
from torchvision import transforms
import gdown
import os

def download_models():
    os.makedirs("models", exist_ok=True)

    files = {
        "best_densenet121_model.h5": "1Iom9OgT_ZO0SWozE7VtPqXttK-Y51nZy",
        "best_fusion_model.h5": "1iVElAsQleizV53ndz20i4hblwlq9-Iyp",
        "best_inceptionv3_model.h5": "1LiueIEmFmT-YR47_m82rtrDq5gZDJJ_Y",
        "final_model.pth": "13WR7Mvs-odTYy7SpNlntv8M3NKDLjTSw"
    }

    for filename, file_id in files.items():
        path = f"models/{filename}"
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

# -------------------------------
# CLEAN BACKGROUND STYLE
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #2E7D32;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CLASS LABELS
# -------------------------------
class_names = ["Adenocarcinoma", "Squamous Cell Carcinoma", "Normal"]

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_swin(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(image).unsqueeze(0)
    return img

# -------------------------------
# FEATURE EXTRACTION (Fusion)
# -------------------------------
def extract_features(image, densenet, inception):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    feat1 = densenet.predict(img, verbose=0)
    feat2 = inception.predict(img, verbose=0)

    combined = np.concatenate([feat1, feat2], axis=1)
    return combined

download_models()

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_fusion_models():
    fusion = tf.keras.models.load_model("models/best_fusion_model.h5")

    densenet_full = tf.keras.models.load_model("models/best_densenet121_model.h5")
    inception_full = tf.keras.models.load_model("models/best_inceptionv3_model.h5")

    densenet = tf.keras.Model(inputs=densenet_full.input,
                              outputs=densenet_full.layers[-2].output)

    inception = tf.keras.Model(inputs=inception_full.input,
                               outputs=inception_full.layers[-2].output)

    return fusion, densenet, inception

@st.cache_resource
def load_densenet():
    return tf.keras.models.load_model("models/best_densenet121_model.h5")

@st.cache_resource
def load_inception():
    return tf.keras.models.load_model("models/best_inceptionv3_model.h5")

@st.cache_resource
def load_swin_model():
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=3
    )
    model.load_state_dict(torch.load("models/final_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<h1 style='text-align: center;'>🧬 Lung Cancer Classification</h1>
<p style='text-align: center; font-size:18px;'>
Feature Fusion Ensemble vs Swin Transformer
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# MODEL SELECTION
# -------------------------------
st.subheader("🔧 Select Model")
model_option = st.selectbox(
    "",
    ["Feature Fusion", "DenseNet121", "InceptionV3", "Swin Transformer"]
)

st.markdown("---")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.subheader("📤 Upload Histopathological Image")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    input_image = preprocess_image(image)

    with col2:
        st.write("### 🔍 Prediction Result")

        try:
            if model_option == "Feature Fusion":
                fusion_model, densenet_model, inception_model = load_fusion_models()
                features = extract_features(image, densenet_model, inception_model)
                preds = fusion_model.predict(features)
                probs = preds[0]

            elif model_option == "DenseNet121":
                model = load_densenet()
                preds = model.predict(input_image)
                probs = preds[0]

            elif model_option == "InceptionV3":
                model = load_inception()
                preds = model.predict(input_image)
                probs = preds[0]

            elif model_option == "Swin Transformer":
                model = load_swin_model()
                img = preprocess_swin(image)

                with torch.no_grad():
                    outputs = model(img)
                    probs = torch.softmax(outputs, dim=1).numpy()[0]

            predicted_class = class_names[np.argmax(probs)]
            confidence = np.max(probs)

            st.success(f"### ✅ {predicted_class}")
            st.write(f"Confidence: {confidence*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
