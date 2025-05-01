import streamlit as st
import gdown
import torch
import os
from arsitektur_clouddeeplabv3 import CloudDeepLabV3Plus  # pastikan file ini sudah ada di GitHub repo

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk download dan cache CloudDeepLabV3+
@st.cache_resource
def load_segmentation_model():
    file_id = "14uQx6dGlV8iCJdQqhWZ6KczfQa7XuaEA"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "clouddeeplabv3.pth"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    model = CloudDeepLabV3Plus()
    model.load_state_dict(torch.load(output, map_location="cpu"))
    model.eval()
    return model

# Fungsi untuk download dan cache YOLOv8
@st.cache_resource
def load_classification_model():
    file_id = "1qG1nvsCBPxOPtiE2Po8yDS521SjfZisI"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "yolov8_cls.pt"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    model = YOLO(output)
    return model

# Load models
seg_model = load_segmentation_model()
cls_model = load_classification_model()

# Mapping nama kelas
class_names = [
    "cumulus", "altocumulus", "cirrus", "clearsky",
    "stratocumulus", "cumulonimbus", "mixed"
]

# Streamlit UI
st.title("🌤️ Cloud Detection App")

uploaded_file = st.file_uploader("Upload Gambar Langit", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Simpan gambar upload
    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize + Padding gambar ke 512x512 (seg_model input)
    input_seg = cv2.resize(img_rgb, (512, 512))
    input_seg = input_seg.transpose(2, 0, 1) / 255.0
    input_seg = torch.from_numpy(input_seg).float().unsqueeze(0)

    # Prediksi segmentasi
    with torch.no_grad():
        output = seg_model(input_seg)["out"].squeeze(0).squeeze(0).numpy()

    mask = (output > 0.5).astype(np.uint8)

    # Hitung tutupan awan
    coverage = (mask.sum() / mask.size) * 100

    # Prediksi klasifikasi
    temp_path = "temp_uploaded_img.jpg"
    cv2.imwrite(temp_path, img)
    result = cls_model.predict(temp_path, verbose=False)[0]
    pred_idx = result.probs.top1
    pred_conf = result.probs.data[pred_idx].item()
    pred_label = class_names[pred_idx]

    # Interpretasi tutupan
    if coverage <= 10:
        sky_condition = "Clear"
    elif coverage <= 30:
        sky_condition = "Mostly Clear"
    elif coverage <= 70:
        sky_condition = "Partly Cloudy"
    elif coverage <= 90:
        sky_condition = "Mostly Cloudy"
    else:
        sky_condition = "Cloudy"

    # Tampilkan hasil
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_rgb, caption="Gambar Asli")

    with col2:
        st.image(mask * 255, caption=f"Tutupan Awan: {coverage:.1f}%")

    with col3:
        st.image(img_rgb, caption=f"Awan: {pred_label} ({pred_conf*100:.1f}%)\nCuaca: {sky_condition}")
