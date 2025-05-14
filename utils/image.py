# utils/image.py
from PIL import Image
import os, zipfile, tempfile


def load_uploaded_images(uploaded_files):
    """Membaca gambar dari file upload atau ZIP."""
    images = []
    for file in uploaded_files:
        if file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                    for name in os.listdir(tmpdir):
                        if name.lower().endswith(('jpg', 'jpeg', 'png')):
                            img_path = os.path.join(tmpdir, name)
                            images.append((name, Image.open(img_path).convert("RGB")))
        else:
            images.append((file.name, Image.open(file).convert("RGB")))
    return images


def load_demo_images(demo_dir="assets/demo"):
    """Memuat gambar contoh dari folder demo_dir."""
    return [
        (f, Image.open(os.path.join(demo_dir, f)).convert("RGB"))
        for f in os.listdir(demo_dir)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ]