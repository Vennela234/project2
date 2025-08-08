import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model
from PIL import Image
import os
import gdown

# --- CONFIG ---
img_height, img_width = 224, 224
class_labels = ['bacterial', 'covid-19', 'normal', 'viral']
conv_layer_name = 'conv2d'

# --- MODEL PATH SETUP ---
MODEL_PATH = "model.h5"
GDRIVE_FILE_ID = "1Ddp0REfCmivIcxlVpnErJLzL_CV8kVia" 
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/file/d/1Ddp0REfCmivIcxlVpnErJLzL_CV8kVia/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_cnn_model()

# --- HELPER FUNCTIONS ---
def preprocess_image(img):
    img = img.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        output = predictions[:, pred_index]

    grads = tape.gradient(output, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs, weights.numpy())
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (img_width, img_height))
    return cam, pred_index.numpy()

def display_gradcam(original_img, heatmap, alpha=0.4):
    img = np.array(original_img.resize((img_width, img_height)))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img.astype(np.uint8), 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# --- STREAMLIT UI ---
st.title("🩺 Pneumonia Chest X-Ray Classifier with Grad-CAM")
st.write("Upload a chest X-ray image. The model will predict the disease type and show a Grad-CAM explanation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    heatmap, pred_index = make_gradcam_heatmap(img_array, model, conv_layer_name)
    prediction_label = class_labels[pred_index]

    st.markdown(f"### 🧠 Predicted Class: `{prediction_label}`")

    overlay = display_gradcam(image, heatmap)
    st.image(overlay, caption="Grad-CAM Explanation", use_column_width=True)
