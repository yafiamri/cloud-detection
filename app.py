import streamlit as st
import torch
import gdown
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
from arsitektur_clouddeeplabv3 import CloudDeepLabV3Plus
import io
from datetime import datetime
from fpdf import FPDF
from zipfile import ZipFile
import matplotlib.pyplot as plt

# === Konfigurasi halaman ===
st.set_page_config(page_title="Deteksi Awan AI", layout="wide")

# === CSS Styling ===
st.markdown("""
<style>
    body {
        color: #f0f0f0;
        background-color: #0e1117;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# === Model Loader ===
@st.cache_resource
def load_segmentation_model():
    file_id = "14uQx6dGlV8iCJdQqhWZ6KczfQa7XuaEA"
    output = "clouddeeplabv3.pth"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", output, quiet=False)
    model = CloudDeepLabV3Plus()
    model.load_state_dict(torch.load(output, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_classification_model():
    file_id = "1qG1nvsCBPxOPtiE2Po8yDS521SjfZisI"
    output = "yolov8_cls.pt"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", output, quiet=False)
    model = YOLO(output)
    return model

# === Preprocessing ===
def resize_with_padding(image, target_size=512):
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError("Ukuran gambar tidak valid: 0 piksel ditemukan.")
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    delta_w, delta_h = target_size - new_w, target_size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded = ImageOps.expand(resized, padding, fill=(0, 0, 0))
    return padded, padding, (new_w, new_h)

def detect_circle_roi(image_np):
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    ((x, y), radius) = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    center, radius = (int(x), int(y)), int(radius)
    mask_circle = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask_circle, center, radius, 1, -1)
    coverage_ratio = (mask_circle * (thresh > 0)).sum() / (np.pi * radius**2)
    return mask_circle if coverage_ratio > 0.85 else None

# === PDF Export ===
def buat_pdf_hasil(nama_file, timestamp, sky_condition, coverage, oktaf, top_preds_str, img1, img2, img3):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hasil Analisis Deteksi Awan Berbasis AI", ln=1, align="C")
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=(f"Nama Berkas: {nama_file}\n"
                               f"Waktu Analisis: {timestamp} WIB\n"
                               f"Kondisi Langit: {sky_condition}\n"
                               f"Tutupan Awan: {coverage:.2f}% (sekitar {oktaf} oktaf)\n"
                               f"Jenis Awan Terdeteksi:\n{top_preds_str}"))
    for idx, (title, img) in enumerate(zip(["Gambar Asli (dengan ROI)", "Predicted Mask", "Overlay"], [img1, img2, img3])):
        img_path = f"temp_vis_{idx}.png"
        img.save(img_path)
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0, 8, txt=title, ln=1)
        pdf.image(img_path, w=170)
    output_path = f"hasil_{nama_file}.pdf"
    pdf.output(output_path)
    return output_path

# === Demo File Class ===
class DemoFile(io.BytesIO):
    def __init__(self, content, name):
        super().__init__(content)
        self.name = name

# === Sidebar Input ===
demo_gambar_paths = ["demo_cumulus.jpg", "demo_cirrus.jpg", "demo_mixed.jpg"]
st.sidebar.header("🖼️ Gambar Contoh")
selected_demo = st.sidebar.selectbox("Pilih gambar contoh", ["(Tidak menggunakan demo)"] + demo_gambar_paths)
if selected_demo != "(Tidak menggunakan demo)":
    with open(selected_demo, "rb") as f:
        uploaded_files = [DemoFile(f.read(), selected_demo)]
else:
    uploaded_files = st.file_uploader("Unggah Gambar Langit", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# === Load Models ===
seg_model = load_segmentation_model()
cls_model = load_classification_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

# === Process ===
if st.button("▶️ Proses") and uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        image = Image.open(uploaded_file).convert("RGB")
        if image.size[0] == 0 or image.size[1] == 0:
            st.error(f"Gambar '{filename}' tidak valid atau memiliki ukuran 0x0.")
            continue

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_resized, padding, resized_dim = resize_with_padding(image)
        image_np = np.array(image_resized) / 255.0

        try:
            roi_mask = detect_circle_roi(image_np)
            if roi_mask is None:
                h, w = resized_dim[::-1]
                if h <= 0 or w <= 0:
                    raise ValueError(f"Ukuran padding tidak valid: {resized_dim}")
                roi_mask = np.pad(np.ones((h, w), dtype=np.uint8),
                                  ((padding[1], padding[3]), (padding[0], padding[2])), constant_values=0)
        except Exception as e:
            st.error(f"Gagal membentuk ROI mask untuk gambar {filename}: {e}")
            continue

        input_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"].squeeze().numpy()
        mask = (output > 0.5).astype(np.uint8)
        cloud_area = (mask * roi_mask).sum()
        roi_area = roi_mask.sum()
        coverage = 100 * cloud_area / roi_area
        oktaf = int(round((coverage / 100) * 8))

        if coverage <= 10:
            sky_condition = "Cerah"
        elif coverage <= 30:
            sky_condition = "Sebagian Cerah"
        elif coverage <= 70:
            sky_condition = "Sebagian Berawan"
        elif coverage <= 90:
            sky_condition = "Berawan"
        else:
            sky_condition = "Mendung"

        temp_path = "temp.jpg"
        image.save(temp_path)
        result = cls_model.predict(temp_path, verbose=False)[0]
        probs = result.probs.data.tolist()
        top_preds = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
        top_preds_str = "\n".join([f"- {label} ({conf*100:.1f}%)" for label, conf in top_preds if conf > 0.05])

        contours, _ = cv2.findContours((roi_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original_img = Image.fromarray(cv2.drawContours((image_np * 255).astype(np.uint8), contours, -1, (255,255,0), 2))
        mask_img_pil = Image.fromarray(cv2.drawContours(cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2RGB), contours, -1, (255,255,0), 2))
        overlay_np = np.where((mask * roi_mask)[:, :, None] == 1, (1 - 0.4) * image_np + 0.4 * np.array([1, 0, 0]), image_np)
        overlay_img = Image.fromarray(cv2.drawContours((overlay_np * 255).astype(np.uint8), contours, -1, (255,255,0), 2))
        draw = ImageDraw.Draw(overlay_img)
        draw.text((10, 490), "AI-Based Cloud Detection by Yafi Amri", fill=(255, 255, 255))

        st.image(image, caption="🖼️ Gambar Asli", use_column_width=True)
        st.subheader(f"📄 Hasil Analisis: {filename}")
        st.markdown(f"""
        🕒 **Waktu Analisis:** {timestamp} WIB  
        ⛅ **Kondisi Langit:** {sky_condition}  
        ☁️ **Tutupan Awan:** {coverage:.2f}% (sekitar {oktaf} oktaf)  
        🌥️ **Jenis Awan Terdeteksi:**  
        {top_preds_str}
        """)

        st.image(original_img, caption="Original (ROI)")
        st.image(mask_img_pil, caption="Predicted Mask")
        st.image(overlay_img, caption="Overlay")

        pdf_file = buat_pdf_hasil(filename, timestamp, sky_condition, coverage, oktaf, top_preds_str, original_img, mask_img_pil, overlay_img)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Unduh PDF", data=f.read(), file_name=pdf_file, mime="application/pdf")
