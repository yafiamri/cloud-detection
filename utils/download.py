# utils/download.py
import streamlit as st
import base64
from utils.export import export_csv, export_pdf, export_zip

def download_controller(data, context="detect"):
    if not data:
        st.warning("Tidak ada data untuk diunduh.")
        return
    
    # Inisialisasi
    selected_format = None
    nama_pengguna = ""
    download_ready = False
    error_msg = None
    href = ""

    # Baris 1: Format + Nama
    row1_col1, row1_col2, _ = st.columns([0.2, 0.3, 0.5])
    with row1_col1:
        selected_format = st.selectbox("Format", ["PDF", "CSV", "ZIP"], key=f"format_{context}")
    with row1_col2:
        if selected_format == "PDF":
            nama_pengguna = st.text_input("Nama untuk Laporan PDF", value="Yafi Amri", key=f"nama_{context}")
        else:
            st.markdown("<br>", unsafe_allow_html=True)

    # Baris 2: Tombol + Link
    row2_col1, row2_col2, _ = st.columns([0.2, 0.3, 0.5])
    with row2_col1:
        clicked = st.button("📤 Buat & Unduh", key=f"btn_{context}")
    with row2_col2:
        if clicked:
            try:
                if selected_format == "CSV":
                    csv_data = export_csv(data)
                    b64 = base64.b64encode(csv_data).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="hasil_{context}.csv">💾 Unduh CSV</a>'
                    download_ready = True

                elif selected_format == "ZIP":
                    zip_path = export_zip(data)
                    with open(zip_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="gambar_{context}.zip">📦 Unduh ZIP</a>'
                    download_ready = True

                elif selected_format == "PDF":
                    pdf_path = export_pdf(data, nama_pengguna)
                    with open(pdf_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="laporan_{context}.pdf">📄 Unduh PDF</a>'
                    download_ready = True

            except Exception as e:
                error_msg = str(e)

        if href:
            st.markdown(href, unsafe_allow_html=True)

    # Baris 3: Notifikasi
    if download_ready:
        st.success("✅ Unduhan siap! Klik tautan di atas.")
    elif error_msg:
        st.error("❌ Gagal membuat file.")
        st.exception(error_msg)