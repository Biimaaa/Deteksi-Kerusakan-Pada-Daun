import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# KONFIGURASI DASAR STREAMLIT
# -------------------------------
st.set_page_config(page_title="Deteksi Area Rusak Daun", layout="wide")
st.title("🌿 Sistem Deteksi Kerusakan Daun")
st.write("""
Program ini mendeteksi area **kerusakan daun** (perubahan warna kuning, coklat, hitam, serta lubang daun) 
menggunakan kombinasi **HSV Thresholding**, **Morphology**, dan **analisis deviasi warna rata-rata daun**.
""")

# -------------------------------
# FUNGSI UTAMA DETEKSI
# -------------------------------
def deteksi_daun_dan_kerusakan(img, kernel_size=5, cluster_sensitivitas=30):
    # Resize dan konversi ke HSV
    img = cv2.resize(img, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1️⃣ Segmentasi daun (warna hijau dominan)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_daun = cv2.inRange(hsv, lower_green, upper_green)

    # 2️⃣ Bersihkan masker daun
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_daun = cv2.morphologyEx(mask_daun, cv2.MORPH_CLOSE, kernel)
    mask_daun = cv2.morphologyEx(mask_daun, cv2.MORPH_OPEN, kernel)

    # 3️⃣ Ambil kontur daun terbesar
    contours, _ = cv2.findContours(mask_daun, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, 0.0, "NoLeaf"
    daun = max(contours, key=cv2.contourArea)
    mask_daun_final = np.zeros_like(mask_daun)
    cv2.drawContours(mask_daun_final, [daun], -1, 255, -1)

    # 4️⃣ Analisis warna rata-rata daun (zona hijau sehat)
    hsv_daun = cv2.bitwise_and(hsv, hsv, mask=mask_daun_final)
    hue_mean = np.mean(hsv_daun[:, :, 0][mask_daun_final > 0])
    sat_mean = np.mean(hsv_daun[:, :, 1][mask_daun_final > 0])
    val_mean = np.mean(hsv_daun[:, :, 2][mask_daun_final > 0])

    # 5️⃣ Hitung deviasi dari warna rata-rata daun sehat
    diff_hue = cv2.absdiff(hsv[:, :, 0], np.full_like(hsv[:, :, 0], hue_mean, dtype=np.uint8))
    diff_sat = cv2.absdiff(hsv[:, :, 1], np.full_like(hsv[:, :, 1], sat_mean, dtype=np.uint8))
    diff_val = cv2.absdiff(hsv[:, :, 2], np.full_like(hsv[:, :, 2], val_mean, dtype=np.uint8))

    # 6️⃣ Deteksi area warna abnormal berdasarkan deviasi
    mask_abnormal = ((diff_hue > cluster_sensitivitas) | 
                     (diff_sat > cluster_sensitivitas) | 
                     (diff_val > cluster_sensitivitas)).astype(np.uint8) * 255

    # Batasi area abnormal hanya di daun
    mask_abnormal = cv2.bitwise_and(mask_abnormal, mask_daun_final)

    # 7️⃣ Warna spesifik untuk rusak (kuning, coklat, hitam)
    lower_yellow = np.array([15, 50, 70])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_brown = np.array([5, 30, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 70])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    mask_rusak_warna = cv2.bitwise_or(mask_yellow, mask_brown)
    mask_rusak_warna = cv2.bitwise_or(mask_rusak_warna, mask_dark)

    # 8️⃣ Gabungkan deviasi + warna rusak
    mask_rusak_total = cv2.bitwise_or(mask_rusak_warna, mask_abnormal)
    mask_rusak_total = cv2.bitwise_and(mask_rusak_total, mask_daun_final)

    # 9️⃣ Perapian morfologi dan filter noise kecil
    mask_rusak_total = cv2.morphologyEx(mask_rusak_total, cv2.MORPH_OPEN, kernel)
    mask_rusak_total = cv2.morphologyEx(mask_rusak_total, cv2.MORPH_CLOSE, kernel)
    contours_rusak, _ = cv2.findContours(mask_rusak_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask_rusak_total)
    for c in contours_rusak:
        if cv2.contourArea(c) > 200:  # hilangkan noise kecil
            cv2.drawContours(mask_filtered, [c], -1, 255, -1)

    # 🔟 Overlay hasil akhir
    hasil = img.copy()
    hasil[mask_filtered > 0] = [255, 0, 0]  # area rusak = merah
    overlay = cv2.addWeighted(img, 0.7, hasil, 0.4, 0)

    # 11️⃣ Hitung persentase rusak
    total_daun = np.sum(mask_daun_final > 0)
    total_rusak = np.sum(mask_filtered > 0)
    persen = (total_rusak / total_daun * 100) if total_daun > 0 else 0

    return mask_daun_final, mask_filtered, overlay, persen, "OK"

# -------------------------------
# ANTARMUKA STREAMLIT
# -------------------------------
uploaded_file = st.file_uploader("📤 Unggah Gambar Daun", type=["jpg", "jpeg", "png"])
kernel_size = st.slider("🧩 Ukuran Kernel Morphology", 1, 10, 5)
cluster_sensitivitas = st.slider("🎚️ Sensitivitas Deteksi Deviasi Warna", 10, 80, 30)

if uploaded_file is not None:
    # Baca gambar dari Streamlit uploader
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    mask_daun, mask_rusak, overlay, persen, status = deteksi_daun_dan_kerusakan(
        img, kernel_size, cluster_sensitivitas
    )

    if status == "NoLeaf":
        st.warning("❌ Tidak ada daun terdeteksi dalam gambar.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 Gambar Asli")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("🔴 Hasil Deteksi Kerusakan")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown(f"### Persentase Area Rusak: **{persen:.2f}%**")
        st.caption("Area berwarna Biru menunjukkan bagian daun yang mengalami kerusakan.")

        if st.button("💾 Simpan Hasil Deteksi"):
            cv2.imwrite("hasil_deteksi_daun.jpg", overlay)
            st.success("✅ Hasil disimpan sebagai `hasil_deteksi_daun.jpg`.")
