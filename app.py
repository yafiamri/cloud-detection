import streamlit as st
import torch, cv2, numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageDraw
from datetime import datetime
import gdown, os
from ultralytics import YOLO
from arsitektur_clouddeeplabv3 import CloudDeepLabV3Plus

def resize_with_padding(image, target=512):
    w, h = image.size
    scale = target / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h))
    pad = (target - new_w) // 2, (target - new_h) // 2
    padded = ImageOps.expand(resized, (pad[0], pad[1], target - new_w - pad[0], target - new_h - pad[1]), fill=(0,0,0))
    return padded, (pad[0], pad[1], target - new_w - pad[0], target - new_h - pad[1]), (new_w, new_h)

def detect_circle_roi(img_np):
    gray = cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(largest)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
    ratio = (mask * (thresh > 0)).sum() / (np.pi * r**2)
    return mask if ratio > 0.85 else None

@st.cache_resource
def load_seg_model():
    fid = "14uQx6dGlV8iCJdQqhWZ6KczfQa7XuaEA"
    if not os.path.exists("clouddeeplabv3.pth"):
        gdown.download(f"https://drive.google.com/uc?export=download&id={fid}", "clouddeeplabv3.pth")
    model = CloudDeepLabV3Plus()
    model.load_state_dict(torch.load("clouddeeplabv3.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_cls_model():
    fid = "1qG1nvsCBPxOPtiE2Po8yDS521SjfZisI"
    if not os.path.exists("yolov8_cls.pt"):
        gdown.download(f"https://drive.google.com/uc?export=download&id={fid}", "yolov8_cls.pt")
    return YOLO("yolov8_cls.pt")

seg_model = load_seg_model()
cls_model = load_cls_model()
class_names = ["cumulus", "altocumulus", "cirrus", "clearsky", "stratocumulus", "cumulonimbus", "mixed"]

st.title("☁️ Cloud Detection (ROI Otomatis)")
files = st.file_uploader("Upload Gambar", type=["jpg","png"], accept_multiple_files=True)

if st.button("▶️ Proses") and files:
    hasil = []
    for f in files:
        img = Image.open(f).convert("RGB")
        name = f.name
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_sq = img.width == img.height
        padded, pad, dims = resize_with_padding(img)
        img_np = np.array(padded) / 255.0
        h, w = img_np.shape[:2]

        roi = detect_circle_roi(img_np) or np.ones((h, w), dtype=np.uint8)
        if not is_sq and roi is None:
            roi[:, :] = 0
            l, t, _, _ = pad
            roi[t:t+dims[1], l:l+dims[0]] = 1

        roi_area = roi.sum()
        input_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).float()
        with torch.no_grad():
            out = seg_model(input_tensor)["out"]
            pred_mask = (torch.sigmoid(out) > 0.5).squeeze().numpy()
        masked = pred_mask * roi
        coverage = 100 * masked.sum() / roi_area

        temp = "temp.jpg"
        padded.save(temp)
        cls_res = cls_model.predict(temp, verbose=False)[0]
        idx = cls_res.probs.top1
        conf = cls_res.probs.data[idx].item()
        label = class_names[idx]
        interpret = ["Clear", "Mostly Clear", "Partly Cloudy", "Mostly Cloudy", "Cloudy"][
            min(int(coverage // 20), 4)]

        overlay = img_np.copy()
        red = np.zeros_like(overlay); red[:,:,0] = 1
        alpha = 0.4
        blend = np.where(masked[:,:,None]==1, (1-alpha)*overlay + alpha*red, overlay)
        blend_img = Image.fromarray((blend*255).astype(np.uint8))
        draw = ImageDraw.Draw(blend_img)
        draw.text((10, 490), "AI-Based Cloud Detection by Yafi Amri", fill=(255,255,255))

        st.subheader(f"📷 {name}")
        col1, col2 = st.columns(2)
        col1.image(padded, caption="Original (Padded)")
        col2.image(blend_img, caption=f"{label} ({conf*100:.1f}%) | {interpret}\nCloud Coverage: {coverage:.2f}%")

        hasil.append({
            "Filename": name,
            "Timestamp": ts,
            "Cloud Coverage (%)": round(coverage, 2),
            "Sky Condition": interpret,
            "Cloud Class": label,
            "Confidence (%)": round(conf*100, 2)
        })

    df = pd.DataFrame(hasil)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "cloud_detection_results.csv", "text/csv")
