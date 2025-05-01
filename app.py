import streamlit as st
import gdown
import torch
import os
from arsitektur_clouddeeplabv3 import CloudDeepLabV3Plus
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json

# =================== MODEL LOADING =================== #
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

seg_model = load_segmentation_model()
cls_model = load_classification_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

# =================== UI START =================== #
st.title("🌤️ Cloud Detection App")

uploaded_file = st.file_uploader("Upload Gambar Langit", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((512, 512))
    image_np = np.array(img_pil) / 255.0

    # ROI OPTIONS
    st.markdown("### Pilih Metode ROI")
    roi_method = st.radio("Metode ROI", ["Otomatis", "Manual - Lingkaran", "Manual - Kotak", "Load ROI JSON"], horizontal=True)

    roi_mask = np.zeros((512, 512), dtype=np.uint8)

    if roi_method.startswith("Manual"):
        draw_mode = "circle" if "Lingkaran" in roi_method else "rect"
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.3)",
            stroke_color="#FFFF00",
            background_image=img_pil,
            height=512,
            width=512,
            drawing_mode=draw_mode,
            update_streamlit=True,
            key="canvas",
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

            with open("roi_drawn.json", "w") as f:
                json.dump(canvas_result.json_data, f)
        else:
            st.warning("Silakan gambar ROI terlebih dahulu.")

    elif roi_method == "Load ROI JSON":
        roi_json = st.file_uploader("Unggah file ROI JSON", type=["json"])
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

    else:
        roi_mask[:, :] = 1

    # SEGMENTATION
    input_seg = image_np.transpose(2, 0, 1)
    input_tensor = torch.from_numpy(input_seg).float().unsqueeze(0)
    with torch.no_grad():
        output = seg_model(input_tensor)["out"].squeeze().numpy()
    mask = (output > 0.5).astype(np.uint8)

    cloud_area = (mask * roi_mask).sum()
    roi_area = roi_mask.sum()
    coverage = 100 * cloud_area / roi_area if roi_area > 0 else 0

    # CLASSIFICATION
    temp_path = "temp_uploaded_img.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    result = cls_model.predict(temp_path, verbose=False)[0]
    pred_idx = result.probs.top1
    pred_conf = result.probs.data[pred_idx].item()
    pred_label = class_names[pred_idx]

    # INTERPRETATION
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

    # VISUAL OUTPUT
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption="Gambar Asli")
    with col2:
        st.image(mask * 255, caption=f"Mask Awan\nTutupan: {coverage:.1f}%")
    with col3:
        overlay = image_np.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 1.0
        alpha = 0.4
        blend = np.where((mask * roi_mask)[:, :, None] == 1,
                         (1 - alpha) * overlay + alpha * red,
                         overlay)
        st.image((blend * 255).astype(np.uint8),
                 caption=f"Awan: {pred_label} ({pred_conf*100:.1f}%)\nCuaca: {sky_condition}")
