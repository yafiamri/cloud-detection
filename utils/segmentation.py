# utils/segmentation.py
import torch
import numpy as np
import cv2
from models.clouddeeplabv3_architecture import CloudDeepLabV3Plus

def load_segmentation_model(weight_path="models/clouddeeplabv3.pth"):
    model = CloudDeepLabV3Plus()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def prepare_input_tensor(image_np):
    img_resized = cv2.resize((image_np * 255).astype(np.uint8), (512, 512)) / 255.0
    input_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float()
    return input_tensor

def predict_segmentation(model, input_tensor):
    with torch.no_grad():
        pred = model(input_tensor)["out"].squeeze().numpy()
    return (pred > 0.5).astype(np.uint8)

def detect_circle_roi(image_np):
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    return mask

def canvas_to_mask(canvas_result, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if canvas_result and canvas_result.json_data:
        canvas_h, canvas_w = canvas_result.image_data.shape[:2]
        scale_x = width / canvas_w
        scale_y = height / canvas_h
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                l = int(obj["left"] * scale_x)
                t = int(obj["top"] * scale_y)
                w = int(obj["width"] * scale_x)
                h = int(obj["height"] * scale_y)
                mask[t:t+h, l:l+w] = 1
            elif obj["type"] == "path" and obj.get("path"):
                coords = [[int(item[1] * scale_x), int(item[2] * scale_y)]
                          for item in obj["path"] if isinstance(item, list) and len(item) >= 3]
                if len(coords) >= 3:
                    poly = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [poly], 1)
            elif obj["type"] == "line":
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                x1 = int(round((left + float(obj["x1"])) * scale_x))
                y1 = int(round((top + float(obj["y1"])) * scale_y))
                x2 = int(round((left + float(obj["x2"])) * scale_x))
                y2 = int(round((top + float(obj["y2"])) * scale_y))
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
                radius = int(round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2))
                if 0 <= cx < width and 0 <= cy < height and radius > 0:
                    cv2.circle(mask, (cx, cy), radius, 1, -1)
    return mask