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

# ======================== Model Loader ========================
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

# ======================== Preprocessing ========================
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

# ======================== Streamlit UI ========================
st.title("☁️ AI-Based Cloud Detection App")
st.markdown("Upload satu atau lebih gambar langit, lalu klik **Proses** untuk mendeteksi tutupan dan jenis awan secara otomatis.")

seg_model = load_segmentation_model()
cls_model = load_classification_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

uploaded_files = st.file_uploader("Upload Gambar Langit", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
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

        # ROI otomatis
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

        # Segmentasi
        input_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"].squeeze().numpy()
        mask = (output > 0.5).astype(np.uint8)
        cloud_area = (mask * roi_mask).sum()
        coverage = 100 * cloud_area / roi_area

        # Klasifikasi
        temp_path = "temp.jpg"
        image.save(temp_path)
        result = cls_model.predict(temp_path, verbose=False)[0]
        pred_idx = result.probs.top1
        pred_conf = result.probs.data[pred_idx].item()
        pred_label = class_names[pred_idx]

        # Interpretasi cuaca
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

        # Overlay
        overlay = image_np.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 1.0
        alpha = 0.4
        blended = np.where((mask * roi_mask)[:, :, None] == 1,
                           (1 - alpha) * overlay + alpha * red,
                           overlay)
        overlay_img = Image.fromarray((blended * 255).astype(np.uint8))
        draw = ImageDraw.Draw(overlay_img)
        draw.text((10, 490), "AI-Based Cloud Detection by Yafi Amri", fill=(255, 255, 255))

        # Tampilan
        st.subheader(f"📷 {filename}")
        col1, col2 = st.columns(2)
        col1.image(image, caption="Gambar Asli")
        col2.image(overlay_img, caption=f"Coverage: {coverage:.2f}% | {sky_condition}\nCloud: {pred_label} ({pred_conf*100:.1f}%)")

        results.append({
            "Filename": filename,
            "Timestamp": timestamp,
            "Cloud Coverage (%)": round(coverage, 2),
            "Sky Condition": sky_condition,
            "Cloud Class": pred_label,
            "Confidence (%)": round(pred_conf * 100, 2)
        })

    df = pd.DataFrame(results)
    st.download_button("⬇️ Download Hasil sebagai CSV", df.to_csv(index=False).encode("utf-8"), "cloud_detection_results.csv", "text/csv")
