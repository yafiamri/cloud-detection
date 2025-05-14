# pages/detect.py
import streamlit as st
import os, io, cv2, base64
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from utils.segmentation import load_segmentation_model, prepare_input_tensor, predict_segmentation, detect_circle_roi, canvas_to_mask
from utils.classification import load_classification_model, predict_classification
from utils.layout import apply_global_styles, display_image_grid, section_divider, small_caption, render_result
from utils.image import load_demo_images, load_uploaded_images
from utils.download import download_controller
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Deteksi Awan", layout="wide")

apply_global_styles()

st.image("assets/logo_transparent.png", width=100)
st.title("☁️ Deteksi Tutupan dan Jenis Awan")
st.write("Pilih gambar citra langit, sesuaikan area pengamatan, lalu biarkan sistem menghitung **tutupan awan** dan mengenali **jenis awan** secara otomatis.")

@st.cache_resource
def get_models():
    return load_segmentation_model(), load_classification_model()

seg_model, cls_model = get_models()

if "report_data" not in st.session_state:
    st.session_state["report_data"] = []

demo_images = load_demo_images()
demo_names = [d[0] for d in demo_images]
selected_demos = st.multiselect("Pilih gambar demo:", demo_names, default=[])

uploaded_files = st.file_uploader(
    "Atau unggah gambar langit:",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=True
)

# Gabungkan gambar dari demo dan upload
demo_buffers = []
for fname, img in demo_images:
    if fname in selected_demos:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.name = fname
        demo_buffers.append(buf)

uploaded_files = demo_buffers + (uploaded_files if uploaded_files else [])

if uploaded_files:
    images = load_uploaded_images(uploaded_files)
    display_image_grid([img for _, img in images], [name for name, _ in images])

    section_divider()
    st.header("🎯 Region of Interest (ROI)")
    roi_selections, canvas_results = {}, {}
    mode = st.radio("Pilih cara menentukan ROI:", ["Otomatis untuk Semua Gambar", "Manual untuk Setiap Gambar"])
    
    if mode == "Otomatis untuk Semua Gambar":
        for name, img in images:
            roi_selections[name] = "Otomatis"
            canvas_results[name] = None
    else:
        for name, img in images:
            with st.expander(f"ROI untuk {name}"):
                roi_type = st.selectbox("Metode ROI", ["Otomatis", "Manual (Kotak)", "Manual (Poligon)", "Manual (Lingkaran Klik)"], key=f"roi_{name}")
                roi_selections[name] = roi_type
                if roi_type != "Otomatis":
                    w, h = img.size
                    ratio = h / w
                    new_w = 640
                    new_h = int(new_w * ratio)
                    resized = img.resize((new_w, new_h))
                    canvas = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=2,
                        background_image=resized,
                        update_streamlit=True,
                        height=new_h,
                        width=new_w,
                        drawing_mode="rect" if "Kotak" in roi_type else "polygon" if "Poligon" in roi_type else "line",
                        key="canvas_" + name
                    )
                    small_caption(f"Canvas: {new_w}×{new_h} (rasio asli dipertahankan)")
                    canvas_results[name] = canvas
                else:
                    small_caption("Mask ROI akan dideteksi otomatis berdasarkan area terang berbentuk lingkaran.")
                    canvas_results[name] = None

    section_divider()
    st.header("🧠 Analisis Gambar")
    
    if st.button("🚀 Jalankan Analisis"):
        with st.spinner("⏳ Analisis sedang diproses..."):
            results = []
            for name, img in images:
                np_img = np.array(img) / 255.0
                h, w = img.size
                roi_type = roi_selections[name]
                canvas = canvas_results[name]
                mask_roi = detect_circle_roi(np_img) if roi_type == "Otomatis" else canvas_to_mask(canvas, img.height, img.width)
                mask_roi = cv2.resize(mask_roi.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
    
                tensor = prepare_input_tensor(np_img)
                pred = predict_segmentation(seg_model, tensor)
                pred = cv2.resize(pred.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
    
                awan = (pred * mask_roi).sum()
                total = mask_roi.sum()
                coverage = 100 * awan / total if total > 0 else 0
                oktaf = int(round((coverage / 100) * 8))
                kondisi = ["Cerah", "Sebagian Cerah", "Sebagian Berawan", "Berawan", "Hampir Tertutup", "Tertutup"][min(oktaf // 2, 5)]
    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("temps/history/images", exist_ok=True)
                os.makedirs("temps/history/masks", exist_ok=True)
                os.makedirs("temps/history/overlays", exist_ok=True)
                img_path = f"temps/history/images/{timestamp}_{name}"
                mask_path = f"temps/history/masks/{timestamp}_{name}"
                overlay_path = f"temps/history/overlays/{timestamp}_{name}"
                cv2.imwrite(mask_path, pred * 255)
                img.save(img_path)
    
                preds = predict_classification(cls_model, img_path)
                jenis = preds[0][0]
    
                overlay_np = np.where((pred * mask_roi)[..., None] == 1, 0.6 * np_img + 0.4 * np.array([1, 0, 0]), np_img)
                overlay_img = Image.fromarray((overlay_np * 255).astype(np.uint8))
                draw = ImageDraw.Draw(overlay_img)
                text = "© AI-Based Cloud Detection"
                font_size = 12
                margin = 10
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                text_width, text_height = draw.textsize(text, font=font)
                x = overlay_img.width - text_width - margin
                y = overlay_img.height - text_height - margin
                
                draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)
                overlay_img.save(overlay_path)
    
                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name,
                    "original_path": img_path,
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "coverage": coverage,
                    "oktaf": oktaf,
                    "kondisi_langit": kondisi,
                    "jenis_awan": jenis,
                    "top_preds": preds
                }
                results.append(result)
    
            st.session_state["report_data"] = results\
            
            riwayat_path = "temps/history/riwayat.csv"
            os.makedirs("temps/history", exist_ok=True)
            
            write_header = not os.path.exists(riwayat_path)
            existing = pd.read_csv(riwayat_path) if not write_header else pd.DataFrame()
            
            new_entries = []
            for r in results:
                top_preds_str = "; ".join([f"{label} ({round(prob*100, 1)}%)" for label, prob in r["top_preds"]])
                new_entries.append({
                    "timestamp": r["timestamp"],
                    "nama_gambar": r["name"],
                    "original_path": r["original_path"],
                    "mask_path": r["mask_path"],
                    "overlay_path": r["overlay_path"],
                    "coverage": round(r["coverage"], 2),
                    "oktaf": r["oktaf"],
                    "kondisi_langit": r["kondisi_langit"],
                    "jenis_awan": r["jenis_awan"],
                    "top_preds": top_preds_str
                })
            
            df = pd.concat([existing, pd.DataFrame(new_entries)], ignore_index=True)
            df.to_csv(riwayat_path, index=False)

report_data = st.session_state.get("report_data", [])
if "report_data" in st.session_state and st.session_state["report_data"]:
    section_divider()
    for r in st.session_state["report_data"]:
        render_result(r)
    st.success("✅ Analisis selesai.")
    st.markdown("---")
    st.markdown("### 📥 Unduh Hasil Analisis")
    download_controller(st.session_state["report_data"], context="deteksi")

# Footer sidebar
st.sidebar.markdown(
    "<hr style='margin-top:350px;margin-bottom:10px'>"
    "<p style='font-size:12px;text-align:center;'>"
    "Dikembangkan oleh <b>Yafi Amri</b><br>"
    "Mahasiswa Meteorologi ITB 2021<br>"
    "</p>", 
    unsafe_allow_html=True
)