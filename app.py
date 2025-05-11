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
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Deteksi Awan AI", layout="wide")
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: #f0f0f0;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def export_pdf(nama_file, timestamp, sky_condition, coverage, oktaf, top_preds_str, img1, img2, img3):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Hasil Analisis Deteksi Awan Berbasis AI", ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.ln(8)
    pdf.multi_cell(0, 8, f"Nama Berkas: {nama_file}\nWaktu Analisis: {timestamp} WIB\nKondisi Langit: {sky_condition}\nTutupan Awan: {coverage:.2f}% (sekitar {oktaf} oktaf)\n\nJenis Awan Terdeteksi:\n{top_preds_str.replace('<b>', '').replace('</b>', '')}")

    for i, (title, img) in enumerate(zip([
        "Gambar Asli (ROI)", "Prediksi Masking Awan", "Overlay Prediksi"
    ], [img1, img2, img3])):
        img_path = f"temp_vis_{i}.png"
        img.save(img_path)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, title, ln=True)
        pdf.image(img_path, w=180)

    output_path = f"hasil_{nama_file.replace('.', '_')}.pdf"
    pdf.output(output_path)
    return output_path

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

seg_model = load_segmentation_model()
cls_model = load_classification_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

demo_gambar_paths = ["demo_cumulus.jpg", "demo_cirrus.jpg", "demo_mixed.jpg"]
st.sidebar.header("🖼️ Gambar Contoh")
selected_demo = st.sidebar.selectbox("Pilih gambar contoh", ["(Tidak menggunakan demo)"] + demo_gambar_paths)
if selected_demo != "(Tidak menggunakan demo)":
    with open(selected_demo, "rb") as f:
        uploaded_files = [io.BytesIO(f.read())]
        uploaded_files[0].name = selected_demo
else:
    uploaded_files = st.file_uploader("Unggah Gambar Langit", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

roi_option = st.sidebar.radio("Pilih Metode ROI:", ["Otomatis (Lingkaran)", "Manual (Kotak)", "Manual (Poligon)"])

if uploaded_files:
    st.subheader("🖼️ Pratinjau Gambar")
    cols = st.columns(len(uploaded_files))
    for i, f in enumerate(uploaded_files):
        img = Image.open(f).convert("RGB")
        cols[i].image(img, caption=f.name, width=180)

    uploaded_file = uploaded_files[0]
    filename = uploaded_file.name
    image = Image.open(uploaded_file).convert("RGB")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w, h = image.size
    target_size = 512
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    delta_w, delta_h = target_size - new_w, target_size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    image_resized = ImageOps.expand(resized, padding, fill=(0, 0, 0))
    image_np = np.array(image_resized) / 255.0

    mask_circle = np.ones((target_size, target_size), dtype=np.uint8)
    manual_mask = None

    if roi_option == "Otomatis (Lingkaran)":
        gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ((x, y), radius) = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        center, radius = (int(x), int(y)), int(radius)
        mask_circle = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask_circle, center, radius, 1, -1)

    elif roi_option in ["Manual (Kotak)", "Manual (Poligon)"]:
        st.warning("Silakan gambar ROI pada kanvas di bawah ini")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            background_image=image_resized.copy(),
            update_streamlit=True,
            height=target_size,
            width=target_size,
            drawing_mode="polygon" if roi_option == "Manual (Poligon)" else "rect",
            key=f"canvas_{filename}_{roi_option.replace(' ', '_')}"
        )

        if canvas_result and canvas_result.json_data and canvas_result.json_data["objects"]:
            manual_mask = np.zeros((target_size, target_size), dtype=np.uint8)
            for obj in canvas_result.json_data["objects"]:
                if roi_option == "Manual (Kotak)":
                    left = int(obj["left"])
                    top = int(obj["top"])
                    width = int(obj["width"])
                    height = int(obj["height"])
                    manual_mask[top:top+height, left:left+width] = 1
                elif roi_option == "Manual (Poligon)":
                    points = obj["path"]
                    coords = [point for point in points if isinstance(point, list) and len(point) == 2]
                    if len(coords) >= 3:
                        # Format ke bentuk (N, 1, 2) untuk fillPoly
                        poly = np.array([[int(p[0]), int(p[1])] for p in coords], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(manual_mask, [poly], 1)

        if st.button("▶️ Proses"):
            if roi_option in ["Manual (Kotak)", "Manual (Poligon)"] and manual_mask is not None:
                mask_circle = manual_mask
            else:
                mask_circle = np.ones((target_size, target_size), dtype=np.uint8)

        input_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"].squeeze().numpy()
        mask = (output > 0.5).astype(np.uint8)

        cloud_area = (mask * mask_circle).sum()
        roi_area = mask_circle.sum()
        coverage = 100 * cloud_area / roi_area if roi_area != 0 else 0
        oktaf = int(round((coverage / 100) * 8))

        temp_path = "temp_input.jpg"
        image.save(temp_path)
        result = cls_model.predict(temp_path, verbose=False)[0]
        probs = result.probs.data.tolist()
        top_preds = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
        top_preds_str = "\n".join([f"- <b>{label}</b> ({conf*100:.1f}%)" if i == 0 else f"- {label} ({conf*100:.1f}%)"
                                    for i, (label, conf) in enumerate(top_preds) if conf > 0.05])

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

        contours, _ = cv2.findContours((mask_circle * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original_img = Image.fromarray(cv2.drawContours((image_np * 255).astype(np.uint8), contours, -1, (255,255,0), 2))
        mask_img_pil = Image.fromarray(cv2.drawContours(cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2RGB), contours, -1, (255,255,0), 2))
        overlay_np = np.where((mask * mask_circle)[:, :, None] == 1, (1 - 0.4) * image_np + 0.4 * np.array([1, 0, 0]), image_np)
        overlay_img = Image.fromarray(cv2.drawContours((overlay_np * 255).astype(np.uint8), contours, -1, (255,255,0), 2))

        col1, col2, col3 = st.columns(3)
        col1.image(original_img, caption="Original (ROI)")
        col2.image(mask_img_pil, caption="Predicted Mask")
        col3.image(overlay_img, caption="Overlay")

        st.markdown(f"""
        ### 📄 Hasil Analisis: `{filename}`
        🕒 **Waktu Analisis:** {timestamp} WIB  
        ⛅ **Kondisi Langit:** {sky_condition}  
        ☁️ **Tutupan Awan:** {coverage:.2f}% (sekitar {oktaf} oktaf)  
        🌥️ **Jenis Awan Terdeteksi:**  
        {top_preds_str}
        """, unsafe_allow_html=True)

        pdf_file = export_pdf(filename, timestamp, sky_condition, coverage, oktaf, top_preds_str, original_img, mask_img_pil, overlay_img)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Unduh Laporan PDF", data=f.read(), file_name=pdf_file, mime="application/pdf")

        st.success("✅ Analisis selesai.")
