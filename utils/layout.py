# utils/layout.py
import streamlit as st
import os
from PIL import Image

def display_image_grid(images, captions, columns=4):
    cols = st.columns(columns)
    for i, (img, caption) in enumerate(zip(images, captions)):
        with cols[i % columns]:
            st.image(img, caption=caption, use_container_width=True)

def result_header(text):
    st.markdown(f"<div class='result-header'>{text}</div>", unsafe_allow_html=True)

def section_divider():
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

def small_caption(text):
    st.markdown(f"<div class='small-font'>{text}</div>", unsafe_allow_html=True)

def apply_global_styles():
    st.markdown("""
        <style>
            html, body {
                font-family: 'Segoe UI', sans-serif;
            }
            section[data-testid="stSidebar"] .css-1d391kg, .css-1v3fvcr {
                font-size: 1.1rem;
                font-weight: 500;
            }
            h1 {
                font-size: 2.2rem;
                color: #2c3e50;
            }
            h2 {
                font-size: 1.5rem;
                color: #34495e;
            }
            .small-font {font-size:13px; color:gray;}
            .result-header {font-size:20px; font-weight:bold; margin-top:1em;}
            .section-divider {border-top: 1px solid #bbb; margin-top: 2em; margin-bottom: 1em;}
        </style>
    """, unsafe_allow_html=True)

def render_result(r):
    st.subheader(f"📊 Hasil: {r['name']}")

    # Baca ulang gambar dari path jika tidak tersedia sebagai objek
    image_ori = Image.open(r["original_path"]) if os.path.exists(r["original_path"]) else None
    image_overlay = Image.open(r["overlay_path"]) if os.path.exists(r["overlay_path"]) else None

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if image_ori:
            st.image(image_ori, caption="Gambar Asli")
        else:
            st.warning("❗ Gambar asli tidak ditemukan.")
    with col_g2:
        if image_overlay:
            st.image(image_overlay, caption="Overlay Segmentasi Awan")
        else:
            st.warning("❗ Overlay tidak ditemukan.")

    with st.container():
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("#### 🖼️ Hasil Segmentasi")
            st.markdown(f"""
            <style>
            table {{
                width: 100%;
            }}
            </style>
            <table>
                <thead>
                    <tr><th>Parameter</th><th>Nilai</th></tr>
                </thead>
                <tbody>
                    <tr><td>Tutupan Awan (%)</td><td>{r['coverage']:.2f}%</td></tr>
                    <tr><td>Tutupan Awan (oktaf)</td><td>{r['oktaf']} oktaf</td></tr>
                    <tr><td>Kondisi Langit</td><td>{r['kondisi_langit']}</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
    
        with col_t2:
            st.markdown("#### 🌥️ Jenis Awan Terdeteksi:")
            table_md = f"""
            <style>
            table {{
                width: 100%;
            }}
            </style>
            <table>
                <thead>
                    <tr><th>Klasifikasi Awan</th><th>Confidence</th></tr>
                </thead>
                <tbody>
            """
            for i, (label, score) in enumerate(r["top_preds"]):
                confidence = f"{score * 100:.1f}%"
                if i == 0:
                    table_md += f"<tr><td>{label}</td><td>{confidence}</td></tr>"
                else:
                    table_md += f"<tr><td>{label}</td><td>{confidence}</td></tr>"
            table_md += "</tbody></table>"
            st.markdown(table_md, unsafe_allow_html=True)