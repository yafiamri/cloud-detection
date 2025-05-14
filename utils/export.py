# utils/export.py
import os
import pandas as pd
import zipfile
from io import BytesIO
from fpdf import FPDF
from datetime import datetime

def safe_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")

def export_csv(data):
    """Ekspor data ke CSV dalam format konsisten."""
    rows = []
    for item in data:
        rows.append({
            "timestamp": item.get("timestamp", ""),
            "nama_gambar": item.get("name", ""),
            "coverage": round(item.get("coverage", 0), 2),
            "oktaf": item.get("oktaf", 0),
            "jenis_awan": item.get("jenis_awan", ""),
            "kondisi_langit": item.get("kondisi_langit", "")
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def export_pdf(data, nama_pengguna, output_path=None):
    """Buat PDF laporan dari hasil analisis awan."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    logo_path = "assets/logo_transparent.png"

    # Style configuration
    primary_color = (41, 128, 185)  # Warna biru Streamlit
    secondary_color = (100, 100, 100)
    
    # Cover Page
    pdf.add_page()
    pdf.image(logo_path, x=(pdf.w - 100)/2, y=40, w=100)
    pdf.set_y(160)
    pdf.set_font("helvetica", 'B', 22)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 12, "Laporan Hasil Deteksi Awan Berbasis AI", ln=1, align='C')
    
    pdf.ln(5)
    pdf.set_text_color(*secondary_color)
    # Disusun oleh
    label = "Disusun oleh: "
    value = nama_pengguna
    pdf.set_font("helvetica", 'B', 14)
    label_w = pdf.get_string_width(label)
    pdf.set_x((pdf.w - pdf.get_string_width(label + value)) / 2)
    pdf.cell(label_w, 8, label, ln=0)
    pdf.set_font("helvetica", '', 14)
    pdf.cell(0, 8, value, ln=1)
    # Dicetak pada
    label = "Dicetak pada: "
    value = datetime.now().strftime('%Y-%m-%d %H:%M WIB')
    pdf.set_font("helvetica", 'B', 14)
    label_w = pdf.get_string_width(label)
    pdf.set_x((pdf.w - pdf.get_string_width(label + value)) / 2)
    pdf.cell(label_w, 8, label, ln=0)
    pdf.set_font("helvetica", '', 14)
    pdf.cell(0, 8, value, ln=1)
    pdf.ln(5)

    pdf.set_y(-25)
    pdf.set_font("helvetica", '', 8)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 5, "© AI-Based Cloud Detection", 0, 1, 'L', link="https://clouddetection.streamlit.app/")
    pdf.set_font("helvetica", 'I', 8)
    pdf.set_text_color(*secondary_color)
    pdf.cell(0, 5, "Yafi Amri - Meteorologi ITB", 0, 0, 'L')
    
    # Konten untuk setiap hasil analisis
    for item in data:
        pdf.add_page()
        
        # Header
        pdf.set_text_color(*primary_color)
        pdf.set_font("helvetica", 'B', 16)
        pdf.cell(40, 10, "Hasil Analisis:", ln=0)
        pdf.set_font("helvetica", '', 16)
        pdf.cell(0, 10, f" {item.get('name', 'Unknown')}", ln=1, border=0)
        pdf.set_text_color(*secondary_color)
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(40, 8, "Waktu Analisis:", ln=0, border=0)
        pdf.set_font("helvetica", '', 12)
        pdf.cell(0, 8, f" {item.get('timestamp', '')}", ln=1)
        pdf.ln(5)

        # Gambar side-by-side
        img_width = 90  # Lebar gambar (mm)
        spacing = 10    # Jarak antar gambar

        # Gambar original
        try:
            img_path = item.get("original_path")
            if img_path and os.path.exists(img_path):
                pdf.image(img_path, x=10, y=40, w=img_width)
                pdf.set_xy(10, 40 + img_width + 5)
                pdf.set_font("helvetica", 'I', 10)
                pdf.cell(img_width, 5, "Gambar Asli", ln=1, align='C')
        except:
            pdf.set_xy(10, 40)
            pdf.set_font("helvetica", 'I', 10)
            pdf.multi_cell(img_width, 5, "Gambar asli tidak ditemukan")
        
        # Gambar overlay
        try:
            overlay_path = item.get("overlay_path")
            overlay_img = item.get("overlay_img")
            
            if overlay_path and os.path.exists(overlay_path):
                pdf.image(overlay_path, x=10+img_width+spacing, y=40, w=img_width)
                pdf.set_xy(10+img_width+spacing, 40 + img_width + 5)
                pdf.cell(img_width, 5, "Overlay Segmentasi", ln=1, align='C')
            elif overlay_img:
                buf = BytesIO()
                overlay_img.save(buf, format="JPEG", quality=95)
                buf.seek(0)
                pdf.image(buf, x=10+img_width+spacing, y=40, w=img_width)
                pdf.set_xy(10+img_width+spacing, 40 + img_width + 5)
                pdf.cell(img_width, 5, "Overlay Segmentasi", ln=1, align='C')
        except:
            pdf.set_xy(10+img_width+spacing, 40)
            pdf.multi_cell(img_width, 5, "Overlay tidak ditemukan")

        # Tabel Hasil Segmentasi
        pdf.set_y(40 + img_width + 15)  # Posisi di bawah gambar
        pdf.set_font("helvetica", 'B', 16)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 10, "Hasil Segmentasi", ln=1)
        col_widths_seg = [95, 95]
        
        # Header tabel
        pdf.set_fill_color(*primary_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(col_widths_seg[0], 10, "Parameter", 1, 0, 'C', fill=True)
        pdf.cell(col_widths_seg[1], 10, "Nilai", 1, 1, 'C', fill=True)
        
        # Isi tabel
        pdf.set_text_color(*secondary_color)
        pdf.set_font("helvetica", '', 12)
        rows = [
            ("Tutupan Awan", f"{item.get('coverage', 0):.2f}%"),
            ("Nilai Oktaf", f"{item.get('oktaf', 0)}"),
            ("Kondisi Langit", item.get('kondisi_langit', ''))
        ]
        for row in rows:
            pdf.cell(col_widths_seg[0], 10, row[0], 1, 0, 'C')
            pdf.cell(col_widths_seg[1], 10, row[1], 1, 1, 'C')

        # Tabel Klasifikasi Awan
        pdf.ln(5)
        pdf.set_font("helvetica", 'B', 16)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 10, "Hasil Klasifikasi", ln=1)
        col_widths_cls = [95, 95]
        
        # Header
        pdf.set_fill_color(*primary_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(col_widths_cls[0], 8, "Jenis Awan", 1, 0, 'C', fill=True)
        pdf.cell(col_widths_cls[1], 8, "Confidence", 1, 1, 'C', fill=True)
        
        # Isi tabel
        pdf.set_text_color(*secondary_color)
        top_preds = item.get("top_preds", [])
        if isinstance(top_preds, str):
            parsed = []
            for line in top_preds.split(";"):
                line = line.strip()
                if "(" in line and ")" in line:
                    try:
                        label = line.split("(")[0].strip()
                        score_str = line.split("(")[1].replace(")", "").replace("%", "").strip()
                        score = float(score_str) / 100
                        parsed.append((label, score))
                    except:
                        continue
            top_preds = parsed
        for i, (label, score) in enumerate(top_preds):
            # Gunakan bold untuk prediksi utama
            if i == 0:
                pdf.set_font("helvetica", 'B', 12)
            else:
                pdf.set_font("helvetica", '', 12)
            
            pdf.cell(col_widths_cls[0], 10, label, 1, 0, 'C')
            pdf.cell(col_widths_cls[1], 10, f"{score*100:.1f}%", 1, 1, 'C')

        # Footer
        pdf.set_y(-25)
        pdf.set_font("helvetica", '', 8)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 5, "© AI-Based Cloud Detection", 0, 1, 'L', link="https://clouddetection.streamlit.app/")
        pdf.set_font("helvetica", 'I', 8)
        pdf.set_text_color(*secondary_color)
        pdf.cell(0, 5, "Yafi Amri - Meteorologi ITB", 0, 0, 'L')
        pdf.set_text_color(*secondary_color)
        pdf.cell(0, 5, f"Halaman {pdf.page_no()-1}", 0, 0, 'R')

    # Save output
    if output_path is None:
        os.makedirs("temps", exist_ok=True)
        output_path = f"temps/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    
    pdf.output(output_path)
    return output_path

def export_zip(data, output_path=None):
    """Kompresi gambar original dan overlay ke dalam ZIP."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for item in data:
            ori = item.get("original_path", "")
            over = item.get("overlay_path", "")
            if os.path.exists(ori):
                zipf.write(ori, arcname=f"original/{os.path.basename(ori)}")
            if os.path.exists(over):
                zipf.write(over, arcname=f"overlay/{os.path.basename(over)}")
    
    if output_path is None:
        os.makedirs("temps", exist_ok=True)
        output_path = f"temps/export_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"

    with open(output_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return output_path