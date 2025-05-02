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

st.title("☁️ Aplikasi Deteksi Awan Berbasis AI")
st.markdown("Unggah satu atau lebih gambar langit, lalu klik **Proses** untuk mendeteksi tutupan dan jenis awan secara otomatis.")

seg_model = load_segmentation_model()
cls_model = load_classification_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

uploaded_files = st.file_uploader("Unggah Gambar Langit", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
process = st.button("▶️ Proses")

results = []

if process and uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        image = Image.open(uploaded_file).convert("RGB")
        is_square = image.size[0] == image.size[1]
        image_resized, padding, resized_dim = resize_with_padding(image, target_size=512)
        image_np = np.array(image_resized) / 255.0

        h, w, _ = image_np.shape
        roi_mask = np.ones((h, w), dtype=np.uint8)
        circle_mask = detect_circle_roi(image_np)
        if circle_mask is not None:
            roi_mask = circle_mask
        elif not is_square:
            left, top, right, bottom = padding
            roi_mask[:, :] = 0
            roi_mask[top:top+resized_dim[1], left:left+resized_dim[0]] = 1

        roi_area = roi_mask.sum()
        if roi_area == 0:
            roi_mask[:, :] = 1

        input_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"].squeeze().numpy()
        mask = (output > 0.5).astype(np.uint8)
        cloud_area = (mask * roi_mask).sum()
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
        pred_label, pred_conf = top_preds[0]

        roi_contour = (roi_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(roi_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        original_np = (image_np * 255).astype(np.uint8)
        original_np = cv2.drawContours(original_np, contours, -1, (255, 255, 0), thickness=2)
        original_img = Image.fromarray(original_np)

        mask_img = (mask * 255).astype(np.uint8)
        mask_img_color = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        mask_img_color = cv2.drawContours(mask_img_color, contours, -1, (255, 255, 0), thickness=2)
        mask_img_pil = Image.fromarray(mask_img_color)

        overlay = image_np.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 1.0
        alpha = 0.4
        blended = np.where((mask * roi_mask)[:, :, None] == 1,
                           (1 - alpha) * overlay + alpha * red,
                           overlay)
        overlay_img = Image.fromarray((blended * 255).astype(np.uint8))
        overlay_np = np.array(overlay_img)
        overlay_np = cv2.drawContours(overlay_np, contours, -1, (255, 255, 0), thickness=2)
        overlay_img = Image.fromarray(overlay_np)
        draw = ImageDraw.Draw(overlay_img)
        draw.text((10, 490), "AI-Based Cloud Detection by Yafi Amri", fill=(255, 255, 255))

        st.image(image, caption="🖼️ Gambar Asli (tanpa padding atau ROI)", use_column_width=True)

        st.subheader(f"📄 Hasil Analisis: {filename}")
        top_preds_str = "\n".join([
            f"- {label} ({conf*100:.1f}% tingkat kepercayaan)"
            for label, conf in top_preds if conf > 0.05
        ])
        st.markdown(f"""
        🕒 **Waktu Analisis:** {timestamp} WIB  
        ⛅ **Kondisi Langit:** {sky_condition}  
        ☁️ **Tutupan Awan:** {coverage:.2f}% (sekitar {oktaf} oktaf)  
        🌥️ **Jenis Awan Terdeteksi:**  
        {top_preds_str}
        """)

        col1, col2, col3 = st.columns(3)
        col1.image(original_img, caption="Original (dengan Padding dan ROI)")
        col2.image(mask_img_pil, caption=f"Predicted Mask\nCloud Coverage (ROI): {coverage:.2f}%")
        col3.image(overlay_img, caption="Overlay with ROI")

        results.append({
            "Nama Berkas": filename,
            "Waktu Analisis": timestamp,
            "Tutupan Awan (%)": round(coverage, 2),
            "Oktaf": oktaf,
            "Kondisi Langit": sky_condition,
            "Jenis Awan Top-1": pred_label,
            "Tingkat Kepercayaan (%)": round(pred_conf * 100, 2)
        })

    df = pd.DataFrame(results)
    st.download_button("⬇️ Unduh Hasil sebagai CSV", df.to_csv(index=False).encode("utf-8"), "hasil_deteksi_awan.csv", "text/csv")
