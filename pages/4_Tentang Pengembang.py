# pages/about.py
import streamlit as st
from utils.layout import apply_global_styles

st.set_page_config(page_title="Tentang Pengembang", layout="wide")

apply_global_styles()

st.image("assets/logo_transparent.png", width=100)
st.title("👨‍💻 Tentang Pembuat Aplikasi")

st.markdown("""
Halo! Saya **Yafi Amri**, mahasiswa Program Studi Meteorologi ITB angkatan 2021.  
Aplikasi ini dikembangkan sebagai bagian dari **Tugas Akhir Purwarupa** dengan judul:

### 📌 *Pengembangan Sistem Pendeteksian Awan Berbasis Artificial Intelligence*

---

**Tujuan Proyek:**  
Merancang sistem berbasis AI untuk:
- Melakukan segmentasi tutupan awan (dengan CloudDeepLabV3+)
- Melakukan klasifikasi jenis awan (dengan YOLOv8)
- Memberikan hasil analisis awan yang objektif, cepat, dan konsisten

**Lingkup Implementasi:**
- Input: Citra langit (upload/manual/kamera)
- Proses: Preprocessing → Segmentasi → Klasifikasi
- Output: Visualisasi tutupan awan (% & oktaf), jenis awan, ekspor PDF/CSV

---

**Kontak (opsional):**
- 🔗 GitHub: [github.com/yafiamri](https://github.com/yafiamri)
- 🌐 LinkedIn: [linkedin.com/in/yafiamri](https://linkedin.com/in/yafiamri)

Terima kasih telah menggunakan aplikasi ini! 🌤️
""")

# Footer sidebar
st.sidebar.markdown(
    "<hr style='margin-top:350px;margin-bottom:10px'>"
    "<p style='font-size:12px;text-align:center;'>"
    "Dikembangkan oleh <b>Yafi Amri</b><br>"
    "Mahasiswa Meteorologi ITB 2021<br>"
    "</p>", 
    unsafe_allow_html=True
)