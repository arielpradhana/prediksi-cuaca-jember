🌤️ Prediksi Cuaca Kota JemberSistem Cerdas Klasifikasi Kondisi Cuaca Berbasis Machine Learning dengan Pendekatan Metodologi CRISP-DM✨ Lihat Demo Langsung (Live App) ✨ 🚀📌 Tentang ProyekCuaca adalah faktor esensial yang memengaruhi beragam aktivitas masyarakat, khususnya di Kota Jember. Mulai dari sektor agrikultur, mobilitas, hingga perencanaan harian. Proyek ini bertujuan untuk membangun sebuah model klasifikasi Machine Learning yang dapat memprediksi kondisi cuaca harian (☀️ Cerah, ⛅ Berawan, 🌧️ Hujan) secara akurat.Pengembangan sistem ini sepenuhnya mengadopsi standar metodologi CRISP-DM (Cross-Industry Standard Process for Data Mining), memastikan setiap tahapan dari pemahaman bisnis hingga evaluasi terstruktur dengan baik.✨ Fitur UtamaAplikasi antarmuka web interaktif (dibangun dengan Streamlit) ini dilengkapi dengan navigasi menu layaknya SaaS premium:📊 Eksplorasi Data Interaktif: Menampilkan visualisasi data historis secara detail (Pie chart, Histogram, Heatmap korelasi, Boxplot).⚙️ Pipeline Preprocessing: Demonstrasi proses pembersihan data, standarisasi, hingga pembagian porsi latih/uji.🤖 Komparasi Model Mesin: Menghadapkan 3 algoritma unggulan (Random Forest, Decision Tree, KNN) untuk melihat performa terbaik.📅 Prediksi Otomatis (API): Mengambil data secara real-time via Open-Meteo API untuk meramalkan cuaca beberapa hari ke depan.✏️ Prediksi Manual (Sintetis): Simulasi ramalan cuaca dengan memasukkan metrik suhu, angin, dan kelembaban secara mandiri.📂 Struktur DirektoriProyek ini diorganisasi sedemikian rupa agar modular dan mudah dipelihara:PROJECT-CUACA-BMKG/
├── app/
│   └── app.py                  # Script utama aplikasi Streamlit UI
├── crawling/
│   └── crawl_bmkg.py           # Script ekstraksi data cuaca API
├── dataset/
│   └── data_cuaca.csv          # Dataset mentah hasil crawling
├── laporan/
│   └── laporan_final.docx      # Dokumen laporan komprehensif CRISP-DM
├── model/
│   ├── label_encoder.pkl       # Enkoder untuk merubah label ke angka
│   ├── model.pkl               # Model Machine Learning utama (Random Forest)
│   └── scaler.pkl              # Scaler untuk standardisasi data input
├── notebook/
│   └── analisis.ipynb          # Eksperimen, EDA, & training model (Jupyter)
├── README.md                   # Dokumentasi proyek (File ini)
└── requirements.txt            # Daftar pustaka/dependensi Python
🚀 Panduan Menjalankan di Komputer Lokal (Localhost)Jika Anda ingin mencoba meremix atau menjalankan aplikasi ini di komputer Anda sendiri, ikuti langkah-langkah mudah berikut:1. Clone Repository inigit clone https://github.com/USERNAME_ANDA/prediksi-cuaca-jember.git
cd prediksi-cuaca-jember
(Catatan: Sesuaikan URL di atas dengan URL GitHub Anda jika di-fork)2. Instalasi DependensiSangat disarankan menggunakan Virtual Environment. Instal pustaka pendukung via requirements.txt:pip install -r requirements.txt
3. Eksekusi Aplikasi StreamlitGunakan terminal untuk meluncurkan server lokal Streamlit. Pastikan Anda menjalankan perintah ini dari root folder proyek:streamlit run app/app.py
Aplikasi akan secara otomatis terbuka di browser melalui http://localhost:8501.🛠️ Stack Teknologi TerapanBahasa Inti: Python 3.8+Pengembangan Web UI: StreamlitPusat Komputasi ML: Scikit-LearnPemrosesan Matriks: Pandas & NumPyGrafik & Visualisasi: Matplotlib & SeabornSumber Aliran Data (API): Open-Meteo API🤝 Kontribusi & ModifikasiSistem ini bersifat terbuka (Open Source). Jangan ragu untuk melakukan Fork, eksplorasi kode, dan mengirimkan Pull Request jika Anda memiliki ide pengembangan algoritma yang lebih mutakhir!
