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

@st.cache_resource
def load_classification_model():
    file_id = "1qG1nvsCBPxOPtiE2Po8yDS521SjfZisI"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "yolov8_cls.pt"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    model = YOLO(output)
    return model

def resize_with_padding(image, target_size=512):
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded = ImageOps.expand(resized, padding, fill=(0, 0, 0))
    return padded, padding, (new_w, new_h)

def detect_circle_roi(image_np):
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)
    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circle, center, radius, 1, -1)
    coverage_ratio = (mask_circle * (thresh > 0)).sum() / (np.pi * radius**2)
    if coverage_ratio > 0.85:
        return mask_circle
    else:
        return None

# === Fungsi ekspor PDF ===
def buat_pdf_hasil(nama_file, timestamp, sky_condition, coverage, oktaf, top_preds_str, img1, img2, img3):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Hasil Analisis Deteksi Awan Berbasis AI", ln=1, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=(
        f"Nama Berkas: {nama_file}\n"
        f"Waktu Analisis: {timestamp} WIB\n"
        f"Kondisi Langit: {sky_condition}\n"
        f"Tutupan Awan: {coverage:.2f}% (sekitar {oktaf} oktaf)\n"
        f"Jenis Awan Terdeteksi:\n{top_preds_str}"
    ))
    pdf.ln(5)

    for idx, (title, img) in enumerate(zip([
        "Gambar Asli (dengan ROI)",
        "Predicted Mask", 
        "Overlay"
    ], [img1, img2, img3])):
        img_path = f"temp_vis_{idx}.png"
        img.save(img_path)
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0, 8, txt=title, ln=1)
        pdf.image(img_path, w=170)
        pdf.ln(5)

    pdf_output = f"hasil_{nama_file}.pdf"
    pdf.output(pdf_output)
    return pdf_output

# === Demo Gambar Contoh ===
demo_gambar_paths = [
    "demo_cumulus.jpg",
    "demo_cirrus.jpg",
    "demo_mixed.jpg"
]

st.sidebar.header("🖼️ Gambar Contoh")
selected_demo = st.sidebar.selectbox("Pilih gambar contoh untuk diuji", options=["(Tidak menggunakan demo)"] + demo_gambar_paths)

if selected_demo != "(Tidak menggunakan demo)":
    with open(selected_demo, "rb") as f:
        demo_bytes = f.read()
    uploaded_files = [io.BytesIO(demo_bytes)]
    for uf in uploaded_files:
        uf.name = selected_demo  # agar filename tetap sesuai

# === Streamlit App ===

# Atur konfigurasi layout dan dark mode
st.set_page_config(page_title="Deteksi Awan AI", layout="wide")

css_style = """
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
"""

st.markdown(css_style, unsafe_allow_html=True)

        # Tambahkan tombol unduh PDF untuk setiap hasil
        pdf_file = buat_pdf_hasil(
            filename, timestamp, sky_condition, coverage, oktaf, top_preds_str,
            original_img, mask_img_pil, overlay_img
        )

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="⬇️ Unduh Hasil sebagai PDF",
                data=f.read(),
                file_name=pdf_file,
                mime="application/pdf"
            )

    # Zip semua PDF
    from zipfile import ZipFile
    zip_path = "semua_hasil_pdf.zip"
    with ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir():
            if file.startswith("hasil_") and file.endswith(".pdf"):
                zipf.write(file)

    with open(zip_path, "rb") as z:
        st.download_button(
            label="⬇️ Unduh Semua PDF sebagai ZIP",
            data=z.read(),
            file_name=zip_path,
            mime="application/zip"
        )

    # === Visualisasi Ringkasan ===
    st.markdown("## 📊 Ringkasan Hasil Analisis")
    if not df.empty:
        import matplotlib.pyplot as plt

        # Histogram tutupan awan
        fig1, ax1 = plt.subplots()
        ax1.hist(df["Tutupan Awan (%)"], bins=8, color='skyblue', edgecolor='black')
        ax1.set_title("Distribusi Tutupan Awan")
        ax1.set_xlabel("Tutupan Awan (%)")
        ax1.set_ylabel("Jumlah Citra")
        st.pyplot(fig1)

        # Pie chart jenis awan top-1
        fig2, ax2 = plt.subplots()
        cloud_counts = df["Jenis Awan Top-1"].value_counts()
        ax2.pie(cloud_counts, labels=cloud_counts.index, autopct='%1.1f%%', startangle=140)
        ax2.set_title("Proporsi Jenis Awan Top-1")
        st.pyplot(fig2)

    # === Unduh Hasil CSV ===
    st.markdown("## 📥 Unduh Hasil Rekapitulasi")
    df = pd.DataFrame(results)
    st.download_button("⬇️ Unduh Hasil sebagai CSV", df.to_csv(index=False).encode("utf-8"), "hasil_deteksi_awan.csv", "text/csv")
