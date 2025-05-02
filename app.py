import streamlit as st
import gdown
import torch
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
from arsitektur_clouddeeplabv3 import CloudDeepLabV3Plus
import json

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

# Load models
seg_model = load_segmentation_model()
cls_model = load_classification_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

# App layout
st.title("☁️ AI-Based Cloud Detection App")
st.markdown("Upload satu atau lebih gambar langit, pilih metode ROI, lalu klik **Proses** untuk mendeteksi awan.")

roi_method = st.radio("Metode ROI", ["Otomatis", "Manual - Lingkaran", "Manual - Kotak", "Manual - Poligon", "Load ROI JSON"], horizontal=True)
uploaded_files = st.file_uploader("Upload Gambar Langit", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
process = st.button("▶️ Proses")

results = []

if process and uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        image_bytes = uploaded_file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((512, 512))
        image_np = np.array(img_pil) / 255.0
        roi_mask = np.zeros((512, 512), dtype=np.uint8)

        if roi_method == "Otomatis":
            roi_mask[:, :] = 1

        elif roi_method == "Load ROI JSON":
            roi_json = st.file_uploader("Unggah file ROI JSON", type=["json"], key=f"json_{filename}")
            if roi_json:
                roi_data = json.load(roi_json)
                for obj in roi_data["objects"]:
                    if obj["type"] == "circle":
                        x = int(obj["left"] + obj["radius"])
                        y = int(obj["top"] + obj["radius"])
                        r = int(obj["radius"])
                        cv2.circle(roi_mask, (x, y), r, 1, -1)
                    elif obj["type"] == "rect":
                        x, y = int(obj["left"]), int(obj["top"])
                        w, h = int(obj["width"]), int(obj["height"])
                        cv2.rectangle(roi_mask, (x, y), (x + w, y + h), 1, -1)
                    elif obj["type"] == "polygon":
                        pts = [(int(p["x"]), int(p["y"])) for p in obj["points"]]
                        cv2.fillPoly(roi_mask, [np.array(pts, np.int32)], 1)

        else:
            draw_mode = "circle" if "Lingkaran" in roi_method else "rect"
            if "Poligon" in roi_method:
                draw_mode = "polygon"

            bg_path = f"canvas_bg_{filename}.png"
            img_pil.save(bg_path)

            canvas_image_np = np.array(img_pil)

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 0, 0.3)",
                stroke_color="#FFFF00",
                background_color="#000000",
                drawing_mode=draw_mode,
                height=512,
                width=512,
                update_streamlit=True,
                key=f"canvas_{filename}"
            )

            if canvas_result.json_data:
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "circle":
                        x = int(obj["left"] + obj["radius"])
                        y = int(obj["top"] + obj["radius"])
                        r = int(obj["radius"])
                        cv2.circle(roi_mask, (x, y), r, 1, -1)
                    elif obj["type"] == "rect":
                        x, y = int(obj["left"]), int(obj["top"])
                        w, h = int(obj["width"]), int(obj["height"])
                        cv2.rectangle(roi_mask, (x, y), (x + w, y + h), 1, -1)
                    elif obj["type"] == "polygon":
                        pts = [(int(p["x"]), int(p["y"])) for p in obj["points"]]
                        cv2.fillPoly(roi_mask, [np.array(pts, np.int32)], 1)
                with open(f"roi_drawn_{filename}.json", "w") as f:
                    json.dump(canvas_result.json_data, f)
            else:
                st.warning(f"Silakan gambar ROI terlebih dahulu untuk {filename}.")
                continue

        # Segmentasi
        input_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"].squeeze().numpy()
        mask = (output > 0.5).astype(np.uint8)
        cloud_area = (mask * roi_mask).sum()
        roi_area = roi_mask.sum()
        coverage = 100 * cloud_area / roi_area if roi_area > 0 else 0

        # Klasifikasi
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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

        # Overlay + watermark
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

        # Display
        st.subheader(f"📷 {filename}")
        col1, col2 = st.columns(2)
        col1.image(img_rgb, caption="Gambar Asli")
        col2.image(overlay_img, caption=f"Coverage: {coverage:.2f}% | {sky_condition}\nCloud: {pred_label} ({pred_conf*100:.1f}%)")

        # Simpan ke hasil
        results.append({
            "Filename": filename,
            "Timestamp": timestamp,
            "Cloud Coverage (%)": round(coverage, 2),
            "Sky Condition": sky_condition,
            "Cloud Class": pred_label,
            "Confidence (%)": round(pred_conf * 100, 2)
        })

    # Unduh hasil
    df = pd.DataFrame(results)
    st.download_button("⬇️ Download Hasil sebagai CSV", df.to_csv(index=False).encode("utf-8"), "cloud_detection_results.csv", "text/csv")
