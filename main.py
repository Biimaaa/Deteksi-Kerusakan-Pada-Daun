import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# === FUNGSI DETEKSI ===
def deteksi_daun_dan_kerusakan_from_img(img):
    img = cv2.resize(img, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_daun = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_daun = cv2.morphologyEx(mask_daun, cv2.MORPH_CLOSE, kernel)
    mask_daun = cv2.morphologyEx(mask_daun, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_daun, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, 0.0, "NoLeaf"

    daun = max(contours, key=cv2.contourArea)
    mask_daun_final = np.zeros_like(mask_daun)
    cv2.drawContours(mask_daun_final, [daun], -1, 255, -1)

    daun_area = cv2.bitwise_and(img, img, mask=mask_daun_final)
    hsv_daun = cv2.cvtColor(daun_area, cv2.COLOR_BGR2HSV)

    lower_yellow, upper_yellow = np.array([15, 50, 70]), np.array([35, 255, 255])
    lower_brown, upper_brown = np.array([5, 30, 20]), np.array([25, 255, 200])
    lower_dark, upper_dark = np.array([0, 0, 0]), np.array([180, 255, 70])

    mask_yellow = cv2.inRange(hsv_daun, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(hsv_daun, lower_brown, upper_brown)
    mask_dark = cv2.inRange(hsv_daun, lower_dark, upper_dark)

    mask_rusak_warna = cv2.bitwise_or(mask_yellow, mask_brown)
    mask_rusak_warna = cv2.bitwise_or(mask_rusak_warna, mask_dark)

    inverse_mask = cv2.bitwise_not(mask_daun_final)
    x, y, w, h = cv2.boundingRect(daun)
    mask_crop = inverse_mask[y:y+h, x:x+w]
    mask_daun_crop = mask_daun_final[y:y+h, x:x+w]
    mask_lubang = cv2.bitwise_and(mask_crop, mask_daun_crop)
    full_mask_lubang = np.zeros_like(mask_daun_final)
    full_mask_lubang[y:y+h, x:x+w] = mask_lubang

    mask_rusak_total = cv2.bitwise_or(mask_rusak_warna, full_mask_lubang)
    mask_rusak_total = cv2.bitwise_and(mask_rusak_total, mask_daun_final)
    mask_rusak_total = cv2.morphologyEx(mask_rusak_total, cv2.MORPH_OPEN, kernel)
    mask_rusak_total = cv2.morphologyEx(mask_rusak_total, cv2.MORPH_CLOSE, kernel)

    contours_rusak, _ = cv2.findContours(mask_rusak_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask_rusak_total)
    for c in contours_rusak:
        if cv2.contourArea(c) > 150:
            cv2.drawContours(mask_filtered, [c], -1, 255, -1)

    hasil = img.copy()
    hasil[mask_filtered > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(img, 0.7, hasil, 0.4, 0)

    total_daun = np.sum(mask_daun_final > 0)
    total_rusak = np.sum(mask_filtered > 0)
    persentase = (total_rusak / total_daun * 100) if total_daun > 0 else 0

    return mask_daun_final, mask_filtered, overlay, persentase, "OK"


# === KONFIGURASI HALAMAN STREAMLIT ===
st.set_page_config(page_title="Deteksi Kerusakan Daun 🌿", layout="wide")

# === GAYA TAMBAHAN CSS ===
st.markdown("""
<style>
body {
    background-color: #f3fff3;
}
.title {
    text-align: center;
    font-size: 40px;
    color: #2E7D32;
    font-weight: bold;
}
.subtext {
    text-align: center;
    font-size: 16px;
    color: #4E944F;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 18px;
    font-weight: bold;
}
footer {
    text-align: center;
    font-size: 14px;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown('<div class="title">🌿 Sistem Deteksi Kerusakan Daun</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Analisis bercak kuning, coklat, hitam, dan lubang pada daun tanaman</div>', unsafe_allow_html=True)
st.markdown("---")

# === PILIH SUMBER GAMBAR ===
option = st.radio("📷 Pilih sumber gambar:", ["Upload Gambar", "Gunakan Kamera"])

# === PROSES DETEKSI ===
def tampilkan_hasil(img):
    mask_daun, mask_rusak, overlay, persen, status = deteksi_daun_dan_kerusakan_from_img(img)
    if status == "NoLeaf":
        st.error("❌ Tidak ada daun terdeteksi pada gambar.")
        return

    # --- Kotak Hasil ---
    if persen < 10:
        warna_bg, teks = "#2E7D32", "✅ Daun sehat"
    elif persen < 40:
        warna_bg, teks = "#FBC02D", "⚠️ Kerusakan ringan"
    else:
        warna_bg, teks = "#C62828", "🚨 Kerusakan berat"

    st.markdown(f"""
    <div class="result-box" style="background-color:{warna_bg};">
        {teks}<br>Persentase area rusak: {persen:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # --- Tampilkan Visual ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Gambar Asli"); axes[0].axis('off')
    axes[1].imshow(mask_daun, cmap='gray')
    axes[1].set_title("Mask Daun"); axes[1].axis('off')
    axes[2].imshow(mask_rusak, cmap='gray')
    axes[2].set_title("Area Rusak"); axes[2].axis('off')
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Hasil Deteksi ({persen:.2f}%)"); axes[3].axis('off')
    st.pyplot(fig)


# === OPSI UPLOAD ===
if option == "Upload Gambar":
    file = st.file_uploader("📁 Unggah gambar daun (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if file:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("🔍 Menganalisis gambar..."):
            tampilkan_hasil(img)

# === OPSI KAMERA ===
elif option == "Gunakan Kamera":
    cam_image = st.camera_input("📸 Ambil foto daun menggunakan kamera Anda")
    if cam_image:
        file_bytes = np.frombuffer(cam_image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("🔍 Menganalisis hasil foto..."):
            tampilkan_hasil(img)

# === FOOTER ===
st.markdown("""
<footer>
🌱 Dikembangkan dengan OpenCV, NumPy, Matplotlib, dan Streamlit | © 2025 Deteksi Daun Otomatis
</footer>
""", unsafe_allow_html=True)
