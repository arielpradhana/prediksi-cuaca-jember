<div align="center">

# 🌤️ Prediksi Cuaca Kota Jember

### Sistem Cerdas Klasifikasi Kondisi Cuaca Berbasis Machine Learning
#### Mengadopsi Metodologi **CRISP-DM** secara Penuh

<br>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Buka%20Aplikasi-4CAF50?style=for-the-badge)](https://prediksicuacajember.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI%20Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/Lisensi-MIT-blue?style=for-the-badge)](LICENSE)

<br>

> **☀️ Cerah · ⛅ Berawan · 🌧️ Hujan**
> — Prediksi kondisi cuaca harian Kota Jember secara akurat dengan kekuatan Machine Learning.

</div>

---

## 📌 Tentang Proyek

Cuaca adalah faktor esensial yang memengaruhi beragam aktivitas masyarakat di Kota Jember — mulai dari **sektor agrikultur**, **mobilitas harian**, hingga **perencanaan kegiatan**. Ketidakpastian cuaca yang tidak terprediksi dapat menimbulkan kerugian yang signifikan.

Proyek ini hadir sebagai solusi: sebuah sistem klasifikasi Machine Learning yang dibangun secara sistematis menggunakan standar industri **CRISP-DM** *(Cross-Industry Standard Process for Data Mining)*, dari tahap pemahaman bisnis hingga evaluasi model yang terstruktur.

```
Bisnis → Data → Preprocessing → Modeling → Evaluasi → Deployment
```

---

## ✨ Fitur Unggulan

| Modul | Ikon | Deskripsi |
|---|:---:|---|
| **Eksplorasi Data** | 📊 | Visualisasi interaktif: Pie Chart, Histogram, Heatmap Korelasi & Boxplot |
| **Pipeline Preprocessing** | ⚙️ | Demo pembersihan data, standarisasi, dan split data latih/uji |
| **Komparasi Model** | 🤖 | Duel performa 3 algoritma: Random Forest vs Decision Tree vs KNN |
| **Prediksi Otomatis** | 📅 | Ambil data real-time via **Open-Meteo API** untuk ramalan hari ke depan |
| **Prediksi Manual** | ✏️ | Simulasi input bebas — masukkan suhu, angin, kelembaban sesukamu |

---

## 🤖 Model Machine Learning

Tiga algoritma dikompetisikan untuk menemukan yang terbaik:

```
┌─────────────────────────────────────────────────────┐
│  🌲 Random Forest   ──▶  Ensemble, Robust, Akurat   │
│  🌿 Decision Tree   ──▶  Interpretatif, Sederhana   │
│  📍 K-NN            ──▶  Berbasis Jarak & Kemiripan │
└─────────────────────────────────────────────────────┘
         ↓  Pemenang disimpan sebagai model.pkl  ↓
```

> Model terbaik (Random Forest) dipilih berdasarkan akurasi, precision, recall, dan F1-Score tertinggi dari hasil cross-validation.

---

## 📂 Struktur Direktori

```
PROJECT-CUACA-BMKG/
│
├── 📁 app/
│   └── app.py                  # 🖥️  Script utama Streamlit UI
│
├── 📁 crawling/
│   └── crawl_bmkg.py           # 🕷️  Ekstraksi data cuaca via API
│
├── 📁 dataset/
│   └── data_cuaca.csv          # 📋  Dataset mentah hasil crawling
│
├── 📁 laporan/
│   └── laporan_final.docx      # 📄  Laporan CRISP-DM komprehensif
│
├── 📁 model/
│   ├── label_encoder.pkl       # 🏷️  Enkoder label → angka
│   ├── model.pkl               # 🧠  Model utama (Random Forest)
│   └── scaler.pkl              # 📐  Scaler standarisasi data input
│
├── 📁 notebook/
│   └── analisis.ipynb          # 🔬  EDA, Eksperimen & Training (Jupyter)
│
├── 📄 README.md                # 📖  Dokumentasi proyek (file ini)
└── 📄 requirements.txt         # 📦  Daftar dependensi Python
```

---

## 🚀 Menjalankan di Lokal

Ikuti 3 langkah mudah ini untuk menjalankan aplikasi di komputermu sendiri:

### 1️⃣ Clone Repository

```bash
git clone https://github.com/arielpradhana/prediksi-cuaca-jember.git
cd prediksi-cuaca-jember
```

> 💡 Sesuaikan URL di atas dengan URL repositori GitHub kamu jika melakukan fork.

### 2️⃣ Install Dependensi

Sangat disarankan menggunakan **Virtual Environment**:

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# Install semua pustaka
pip install -r requirements.txt
```

### 3️⃣ Jalankan Aplikasi

```bash
streamlit run app/app.py
```

Aplikasi akan otomatis terbuka di browser pada: **`http://localhost:8501`** 🎉

---

## 🛠️ Stack Teknologi

<div align="center">

| Kategori | Teknologi |
|:---:|:---:|
| **Bahasa** | Python 3.8+ |
| **Web UI** | Streamlit |
| **Machine Learning** | Scikit-Learn |
| **Manipulasi Data** | Pandas · NumPy |
| **Visualisasi** | Matplotlib · Seaborn |
| **Sumber Data** | Open-Meteo API |

</div>

---

## 📊 Metodologi CRISP-DM

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  Business│────▶│   Data   │────▶│   Data   │
  │Understanding    Understanding    Preparation│
  └──────────┘     └──────────┘     └──────────┘
                                          │
  ┌──────────┐     ┌──────────┐          ▼
  │Deployment│◀────│Evaluation│◀────┌──────────┐
  │          │     │          │     │ Modeling │
  └──────────┘     └──────────┘     └──────────┘
```

Setiap tahapan CRISP-DM diimplementasikan secara eksplisit, terdokumentasi di `laporan/laporan_final.docx`, dan dapat ditelusuri di `notebook/analisis.ipynb`.

---

## 🤝 Kontribusi

Proyek ini bersifat **Open Source** dan terbuka untuk kolaborasi!

1. 🍴 **Fork** repositori ini
2. 🌿 Buat branch baru: `git checkout -b fitur/nama-fitur`
3. ✅ Commit perubahanmu: `git commit -m 'Tambah fitur: nama-fitur'`
4. 📤 Push ke branch: `git push origin fitur/nama-fitur`
5. 🔁 Buat **Pull Request**

Ide pengembangan yang disambut: algoritma baru (XGBoost, SVM, LSTM), fitur notifikasi cuaca, integrasi data BMKG langsung, atau optimasi UI.

---

## 📜 Lisensi

Proyek ini dirilis di bawah lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan dengan tetap menyertakan atribusi.

---

<div align="center">


⭐ **Jika proyek ini bermanfaat, jangan lupa beri bintang!** ⭐

[![GitHub stars](https://img.shields.io/github/stars/arielpradhana/prediksi-cuaca-jember?style=social)](https://github.com/arielpradhana/prediksi-cuaca-jember)

</div>
