import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, '../model/model.pkl')
scaler_path = os.path.join(current_dir, '../model/scaler.pkl')
le_path = os.path.join(current_dir, '../model/label_encoder.pkl')

model  = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
le     = pickle.load(open(le_path, 'rb'))

if "df" not in st.session_state:
    st.session_state.df = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Perbaikan global typography yang aman, tidak merusak komponen internal Streamlit */
html, body, div, span, p, h1, h2, h3, h4, h5, h6, label, button, input {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Base App Background */
.stApp {
    background-color: #FAFAFD; 
}

/* ─── SIDEBAR MODERN MENU ─── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #F0EAF8;
    padding-top: 1rem;
}

/* Menyembunyikan navigasi default jika ada */
[data-testid="stSidebarNav"] { display: none !important; }

/* Menghilangkan bulat/lingkaran (radio circle) pada menu */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Menyesuaikan layout menu setelah bulatan dihilangkan */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    width: 100%;
    border-radius: 14px;
    padding: 14px 18px !important;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
    background-color: transparent;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background-color: #F8F5FC;
}

/* Styling Text di dalam Menu (Default) */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] p {
    margin: 0 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #6B5E82;
}

/* ─── GENERAL TYPOGRAPHY & LAYOUT ─── */
h1, h2, h3, h4, p { color: #2D1B4E; }

.section-title {
    color: #2D1B4E;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 2rem;
    margin-bottom: 1.2rem;
    letter-spacing: -0.02em;
}

.hero {
    padding: 4rem 3rem;
    background: linear-gradient(135deg, #F8F5FC 0%, #EAE0F5 100%);
    border-radius: 32px;
    box-shadow: 0 12px 40px rgba(79, 53, 132, 0.05);
    border: 1px solid #EAE0F5;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.7) 0%, transparent 60%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -20px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(211,197,234,0.4) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { 
    font-size: 3.5rem; 
    font-weight: 800; 
    color: #2D1B4E; 
    margin: 0; 
    line-height: 1.1;
    letter-spacing: -0.03em;
    position: relative;
    z-index: 2;
}
.hero p { 
    color: #554572; 
    font-size: 1.15rem; 
    margin-top: 1.2rem; 
    max-width: 650px;
    line-height: 1.6;
    font-weight: 500;
    position: relative;
    z-index: 2;
}

/* ─── CARDS & CONTAINERS ─── */
.info-box {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    color: #4F3584;
    font-size: 1.05rem;
    box-shadow: 0 8px 24px rgba(79, 53, 132, 0.03);
    border: 1px solid #F0EAF8;
    line-height: 1.6;
}

.crisp-step {
    background: #FFFFFF;
    border-radius: 24px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    border-left: 0;
    box-shadow: 0 8px 24px rgba(79, 53, 132, 0.03);
    border: 1px solid #F0EAF8;
    line-height: 1.6;
    color: #6B5E82;
    position: relative;
    overflow: hidden;
}
.crisp-step::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 6px;
    background: #4F3584;
}
.crisp-step h4 {
    color: #2D1B4E;
    margin: 0 0 0.8rem 0;
    font-size: 1.25rem;
    font-weight: 800;
}

/* ─── EVALUATION CARDS ─── */
.model-card {
    background: #FFFFFF;
    border-radius: 28px;
    padding: 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 32px rgba(79, 53, 132, 0.04);
    border: 1px solid #F0EAF8;
}
.model-title { 
    font-size: 1.4rem; 
    font-weight: 800; 
    color: #2D1B4E; 
    margin-bottom: 0.5rem; 
}
.model-acc {
    font-size: 3.5rem; 
    font-weight: 800;
    color: #4F3584;
    margin-bottom: 1.5rem;
    letter-spacing: -0.03em;
    line-height: 1;
}
.best-badge {
    display: inline-block; 
    padding: 0.5rem 1.2rem;
    background: #2D1B4E;
    border-radius: 50px; 
    font-size: 0.85rem;
    font-weight: 700; 
    color: #FFFFFF; 
    margin-left: 1rem;
    vertical-align: middle;
    letter-spacing: 0.02em;
}

.metric-row { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
.metric-box {
    flex: 1; 
    min-width: 90px;
    background: #FDFBFF;
    border-radius: 18px; 
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #F0EAF8;
}
.metric-box .ml { color: #6B5E82; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.4rem;}
.metric-box .mv { color: #2D1B4E; font-size: 1.4rem; font-weight: 800; }

.class-row { display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap; }
.class-box {
    flex: 1; 
    min-width: 120px;
    border-radius: 18px; 
    padding: 1.2rem;
    text-align: center; 
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.5;
}

/* ─── RESULT CARDS IN PREDICTION ─── */
.result-card {
    background: #FFFFFF;
    border-radius: 24px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 10px 30px rgba(79, 53, 132, 0.04);
    color: #2D1B4E;
    font-weight: 600;
    border: 1px solid #F0EAF8;
    position: relative;
    overflow: hidden;
}
.result-cerah::after   { content:''; position:absolute; left:0; top:0; bottom:0; width:8px; background:#F59E0B; }
.result-berawan::after { content:''; position:absolute; left:0; top:0; bottom:0; width:8px; background:#A78BFA; }
.result-hujan::after   { content:''; position:absolute; left:0; top:0; bottom:0; width:8px; background:#3B82F6; }

.stat-row { display:flex; gap:1.5rem; margin-bottom:1.5rem; }
.stat-box {
    flex:1; 
    background: #FFFFFF;
    border-radius: 24px; 
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(79, 53, 132, 0.04);
    text-align:center;
    border: 1px solid #F0EAF8;
}
.stat-box .label { color: #6B5E82; font-size: 1.05rem; font-weight: 700;}
.stat-box .value { color: #2D1B4E; font-size: 2.2rem; font-weight: 800; margin-top: 0.5rem;}

.mode-badge {
    display: inline-block; 
    padding: 0.6rem 1.5rem;
    border-radius: 50px; 
    font-size: 0.95rem; 
    font-weight: 700;
    margin-bottom: 1.5rem;
    background: #FDFBFF; 
    border: 1px solid #D3C5EA;
    color: #4F3584;
}

/* ─── STREAMLIT UI OVERRIDES ─── */
/* Primary Button Action */
div.stButton > button {
    background-color: #FFFFFF !important;
    color: #4F3584 !important;
    border: 1px solid #D3C5EA !important;
    border-radius: 50px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #F8F5FC !important;
    border-color: #4F3584 !important;
}

/* ─── PERBAIKAN TOMBOL UPLOAD DATASET ─── */
/* Mengatasi teks bertumpuk akibat input file bawaan browser yang bocor */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    border: 2px dashed #D3C5EA !important;
}
[data-testid="stFileUploader"] section {
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] input[type="file"] {
    opacity: 0 !important;
    color: transparent !important;
}
[data-testid="stFileUploader"] button {
    background-color: #4F3584 !important;
    color: #FFFFFF !important;
    border-radius: 50px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(79, 53, 132, 0.2) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #382463 !important;
}

/* Special styling for "Prediksi Sekarang" buttons to make them pop */
button[key^="btn_"] {
    background-color: #4F3584 !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(79, 53, 132, 0.25) !important;
}
button[key^="btn_"]:hover {
    background-color: #382463 !important;
    box-shadow: 0 8px 25px rgba(79, 53, 132, 0.35) !important;
}

/* Tab Component di Prediction */
div[data-testid="stTabs"] button {
    color: #6B5E82 !important;
    background: transparent !important;
    font-size: 1.05rem !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px 12px 0 0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4F3584 !important;
    border-bottom: 3px solid #4F3584 !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR NAVIGATION ───────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#2D1B4E; font-weight:800; font-size: 1.6rem; margin-bottom: 1.5rem; margin-top:0.5rem;'>Navigasi Menu</h2>", unsafe_allow_html=True)
    menu_options = [
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluasi",
        "Prediksi"
    ]
    menu_choice = st.radio("Pilih Menu:", menu_options, label_visibility="collapsed")

    # CSS Dinamis untuk Menu Aktif (Dijamin pasti nyala sesuai pilihan)
    menu_index = menu_options.index(menu_choice)
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div:nth-child({menu_index + 1}) label {{
        background-color: #4F3584 !important;
        box-shadow: 0 4px 15px rgba(79, 53, 132, 0.2);
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div:nth-child({menu_index + 1}) p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Prediksi Cuaca Jember</h1>
    <p>Bangun pemahaman cuaca yang lebih baik dengan sistem prediksi berbasis Machine Learning melalui pendekatan metodologi CRISP-DM yang akurat dan responsif.</p>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD DATASET (Menggunakan komponen asli bawaan Streamlit) ──────
st.markdown('<p class="section-title">Upload Dataset</p>', unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns([3, 1])
with upload_col1:
    uploaded_file = st.file_uploader(
        "Pilih file CSV hasil crawling",
        type=["csv"],
        help="Format dataset: kondisi, suhu_max, suhu_min, suhu_rata, curah_hujan, kecepatan_angin, kelembaban_max, kelembaban_min"
    )
with upload_col2:
    if st.session_state.df is not None:
        st.write("")
        st.write("")
        if st.button("Ganti Dataset", use_container_width=True):
            st.session_state.df = None
            st.rerun()

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"Dataset berhasil dimuat — {len(st.session_state.df)} baris, {len(st.session_state.df.columns)} kolom")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

df = st.session_state.df

if df is None:
    st.markdown("""
    <div class="info-box">
        <b>Data Belum Tersedia</b><br><br>
        Silakan upload file CSV hasil crawling di kotak atas untuk membuka
        akses ke eksplorasi data, metrik statistik, visualisasi lanjutan, perbandingan akurasi, dan performa
        model pada panel navigasi di sebelah kiri. Navigasi <b>Prediksi</b>
        tetap dapat diakses secara langsung karena mesin prediksi telah dilatih sebelumnya.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MENU 1 — BUSINESS UNDERSTANDING
# ══════════════════════════════════════════════════════
if menu_choice == "Business Understanding":
    st.markdown('<p class="section-title">Business Understanding</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Latar Belakang</b><br>
        Cuaca merupakan faktor esensial yang mempengaruhi beragam aktivitas masyarakat Kota Jember,
        mulai dari sektor agrikultur, mobilitas transportasi, hingga kegiatan rumah tangga sehari-hari. Prediksi kondisi cuaca
        yang presisi dapat memperkuat masyarakat dalam pengambilan keputusan taktis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="crisp-step">
        <h4>Tujuan Bisnis</h4>
        Membangun sistem cerdas prediksi kondisi cuaca harian Kota Jember
        (Cerah, Berawan, Hujan) menggunakan Machine Learning untuk
        mempermudah perencanaan rutinitas masyarakat.
    </div>
    <div class="crisp-step">
        <h4>Rumusan Masalah</h4>
        Bagaimana memodelkan klasifikasi kondisi cuaca harian Kota Jember
        bersumber pada variabel suhu, tingkat kelembaban, dan laju angin
        dengan memanfaatkan performa algoritma Machine Learning?
    </div>
    <div class="crisp-step">
        <h4>Kriteria Keberhasilan</h4>
        Model ditargetkan mampu mengidentifikasi kondisi cuaca dengan metrik akurasi
        di atas batas minimal 70% pada tahap pengujian data.
    </div>
    <div class="crisp-step">
        <h4>Sumber Data Terpusat</h4>
        Catatan histori cuaca Kota Jember diekstraksi dari Open-Meteo API
        (jaringan stasiun metereologi WMO/BMKG) secara kontinu melalui metode
        automasi terprogram yang melingkupi periode observasi 2023–2024.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MENU 2 — DATA UNDERSTANDING
# ══════════════════════════════════════════════════════
elif menu_choice == "Data Understanding":
    st.markdown('<p class="section-title">Data Understanding</p>', unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="info-box">
            Kumpulan data tidak ditemukan. Mohon unggah dokumen dataset CSV Anda pada modul di atas 
            untuk merender tampilan sampel data, tinjauan statistik deskriptif, rasio data yang hilang, serta bagan visualisasi.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Info umum
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sampel", f"{len(df)} Baris")
        col2.metric("Dimensi", f"{len(df.columns)} Fitur")
        col3.metric("Rentang", "2023–2024")
        col4.metric("Kategori", "3 Kelas")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tampilkan data
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Sampel Kumpulan Data</p>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Statistik deskriptif
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Analisis Statistik Deskriptif</p>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Missing values
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Kekurangan Nilai (Missing Values)</p>', unsafe_allow_html=True)
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Dimensi Fitur", "Total Kosong"]
        st.dataframe(missing, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualisasi
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Eksplorasi Visualisasi</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Distribusi label
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            kondisi_count = df['kondisi'].value_counts()
            colors = ['#A78BFA', '#F59E0B', '#3B82F6']
            ax.pie(kondisi_count, labels=kondisi_count.index,
                   autopct='%1.1f%%', colors=colors,
                   textprops={'color':'#2D1B4E', 'fontweight': 'bold'})
            ax.set_title('Persentase Kondisi Cuaca', color='#2D1B4E', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with col2:
            # Histogram suhu
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            ax.hist(df['suhu_rata'], bins=20, color='#4F3584', edgecolor='#FFFFFF', alpha=0.9)
            ax.set_title('Persebaran Suhu Rata-rata', color='#2D1B4E', fontweight='bold')
            ax.set_xlabel('Derajat Suhu (°C)', color='#6B5E82')
            ax.set_ylabel('Frekuensi Kemunculan', color='#6B5E82')
            ax.tick_params(colors='#6B5E82')
            for spine in ax.spines.values(): spine.set_edgecolor('#F0EAF8')
            st.pyplot(fig)
            plt.close()

        # Heatmap korelasi
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Matriks Korelasi (Heatmap)</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        cols = ['suhu_max','suhu_min','suhu_rata','curah_hujan','kecepatan_angin','kelembaban_max','kelembaban_min']
        sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', ax=ax,
                    cmap='Purples', linewidths=0.5,
                    annot_kws={'color':'#2D1B4E', 'size':9})
        ax.tick_params(colors='#6B5E82')
        ax.set_title('Hubungan Linier Antar Variabel', color='#2D1B4E', fontweight='bold')
        st.pyplot(fig)
        plt.close()

        # Boxplot suhu per kondisi
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Sebaran Suhu berdasarkan Kondisi</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        df.boxplot(column='suhu_rata', by='kondisi', ax=ax,
                   patch_artist=True,
                   boxprops=dict(facecolor='#F6F2FC', color='#4F3584'),
                   medianprops=dict(color='#4F3584', linewidth=2),
                   whiskerprops=dict(color='#4F3584'),
                   capprops=dict(color='#4F3584'),
                   flierprops=dict(markerfacecolor='#4F3584'))
        ax.set_title('Suhu Rata-rata per Kelas Cuaca', color='#2D1B4E', fontweight='bold')
        ax.set_xlabel('Klasifikasi', color='#6B5E82')
        ax.set_ylabel('Suhu (°C)', color='#6B5E82')
        ax.tick_params(colors='#6B5E82')
        plt.suptitle('')
        for spine in ax.spines.values(): spine.set_edgecolor('#F0EAF8')
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════
# MENU 3 — DATA PREPARATION
# ══════════════════════════════════════════════════════
elif menu_choice == "Data Preparation":
    st.markdown('<p class="section-title">Data Preparation</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="crisp-step">
        <h4>1. Imputasi Data Kosong (Missing Values)</h4>
        Bila ditemukan, celah data akan disubstitusi menggunakan nilai rata-rata kalkulatif (mean imputation)
        pada masing-masing dimensi numerik.
    </div>
    <div class="crisp-step">
        <h4>2. Pemilihan Fitur Induk</h4>
        Dimensi fitur primer yang dilibatkan: suhu_max, suhu_min, suhu_rata,
        kecepatan_angin, kelembaban_max, kelembaban_min.
        Parameter tanggal dan curah_hujan dieliminasi sebagai input fitur prediktif.
    </div>
    <div class="crisp-step">
        <h4>3. Pengkodean Label Kategori</h4>
        Transformasi teks klasifikasi cuaca menjadi format komputasi numerikal:<br>
        • Berawan → 0 &nbsp;|&nbsp; Cerah → 1 &nbsp;|&nbsp; Hujan → 2
    </div>
    <div class="crisp-step">
        <h4>4. Standardisasi Skala (StandardScaler)</h4>
        Setiap parameter diformat menggunakan fungsi StandardScaler untuk mereduksi disparitas skala
        (mean=0, deviasi standar=1), menjamin keseimbangan kontribusi setiap fitur pada model.
    </div>
    <div class="crisp-step">
        <h4>5. Pemisahan Korpus (Train/Test Split)</h4>
        Korpus data disegmentasi menjadi porsi 80% pelatihan (584 observasi) dan
        20% pengujian independen (147 observasi) menggunakan teknik stratified 
        demi mempertahankan proporsi kelas aslinya.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="info-box">
            Sisipkan dataset terlebih dulu untuk memuat visualisasi sebaran metrik hasil kalkulasi
            dan pencuplikan data usai tahap preparasi.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Visualisasi distribusi label
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Distribusi Label Pasca-Prep</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        kondisi_count = df['kondisi'].value_counts()
        col1.metric("Berawan", f"{kondisi_count.get('Berawan', 0)} Baris")
        col2.metric("Cerah",   f"{kondisi_count.get('Cerah', 0)} Baris")
        col3.metric("Hujan",  f"{kondisi_count.get('Hujan', 0)} Baris")

        st.markdown("<br>", unsafe_allow_html=True)

        # Sample data setelah preprocessing
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Cuplikan Matriks Pasca-Prep (Scaled)</p>', unsafe_allow_html=True)
        fitur = ['suhu_max','suhu_min','suhu_rata','kecepatan_angin','kelembaban_max','kelembaban_min']
        df_prep = df[fitur].copy()
        df_scaled_show = pd.DataFrame(
            scaler.transform(df_prep),
            columns=[f"{c} (scaled)" for c in fitur]
        )
        st.dataframe(df_scaled_show.head(10).round(4), use_container_width=True)

# ══════════════════════════════════════════════════════
# MENU 4 — MODELING
# ══════════════════════════════════════════════════════
elif menu_choice == "Modeling":
    st.markdown('<p class="section-title">Modeling Architecture</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Tiga arsitektur algoritma klasifikasi dirakit dan dikomparasikan rasio keberhasilannya
        guna mengerucut pada performa mesin komputasi cuaca yang paling optimal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="crisp-step">
        <h4>Decision Tree Classifier</h4>
        Skema arsitektur binar percabangan yang memecah data bersandar pada metrik evaluasi fitur
        paling dominan. Strukturnya transparan secara logika, meski berpotensi pada isu overfitting.<br><br>
        <b>Konfigurasi Parameter:</b> random_state=42
    </div>
    <div class="crisp-step">
        <h4>Random Forest Classifier</h4>
        Kolektif pohon keputusan simultan yang memproses prediksi secara kohesif (ensemble).
        Infrastruktur ini lebih persisten dalam mereduksi noise dan menyajikan akurasi kumulatif teratas.<br><br>
        <b>Konfigurasi Parameter:</b> n_estimators=100, random_state=42
    </div>
    <div class="crisp-step">
        <h4>K-Nearest Neighbors (KNN)</h4>
        Metode spasial klasifikasi proksimitas yang melebeli entri data mengacu pada
        jarak kedekatan dengan tetangga observasi (K) di dimensi ruang multi-fitur.<br><br>
        <b>Konfigurasi Parameter:</b> n_neighbors=5
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="info-box">
            Sisipkan dataset Anda ke sistem untuk merender representasi diagram
            skala komparasi akurasi ketiga arsitektur ini.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Perbandingan akurasi visual
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Rasio Komparasi Akurasi</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        models_name = ['Decision Tree', 'Random Forest', 'KNN']
        accuracies  = [65.99, 78.91, 74.15]
        colors      = ['#D3C5EA', '#4F3584', '#A78BFA']
        bars = ax.bar(models_name, accuracies, color=colors, edgecolor='#FFFFFF', alpha=0.9)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val}%', ha='center', color='#2D1B4E', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Persentase Akurasi (%)', color='#6B5E82')
        ax.set_title('Uji Reliabilitas Mesin', color='#2D1B4E', fontweight='bold')
        ax.tick_params(colors='#6B5E82')
        for spine in ax.spines.values(): spine.set_edgecolor('#F0EAF8')
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════
# MENU 5 — EVALUASI
# ══════════════════════════════════════════════════════
elif menu_choice == "Evaluasi":
    st.markdown('<p class="section-title">Evaluasi Performa Sistem</p>', unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="info-box">
            Mohon sertakan dataset di awal agar laporan lengkap evaluasi 
            (presisi, daya recall, skor harmoni F1) masing-masing komputasi dapat ditampilkan.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Decision Tree
        st.markdown("""
        <div class="model-card">
            <div class="model-title">Decision Tree</div>
            <div class="model-acc">65.99%</div>
            <div class="metric-row">
                <div class="metric-box"><div class="ml">Precision</div><div class="mv">0.50</div></div>
                <div class="metric-box"><div class="ml">Recall</div><div class="mv">0.47</div></div>
                <div class="metric-box"><div class="ml">F1-Score</div><div class="mv">0.48</div></div>
                <div class="metric-box"><div class="ml">Support</div><div class="mv">147</div></div>
            </div>
            <div class="class-row">
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Berawan<br>P:0.75 R:0.79 F1:0.77
                </div>
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Cerah<br>P:0.20 R:0.20 F1:0.20
                </div>
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Hujan<br>P:0.55 R:0.43 F1:0.48
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Random Forest
        st.markdown("""
        <div class="model-card" style="border: 2px solid #4F3584;">
            <div class="model-title">Random Forest <span class="best-badge">Model Terpilih</span></div>
            <div class="model-acc">78.91%</div>
            <div class="metric-row">
                <div class="metric-box"><div class="ml">Precision</div><div class="mv">0.65</div></div>
                <div class="metric-box"><div class="ml">Recall</div><div class="mv">0.54</div></div>
                <div class="metric-box"><div class="ml">F1-Score</div><div class="mv">0.56</div></div>
                <div class="metric-box"><div class="ml">Support</div><div class="mv">147</div></div>
            </div>
            <div class="class-row">
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Berawan<br>P:0.80 R:0.94 F1:0.86
                </div>
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Cerah<br>P:0.33 R:0.07 F1:0.11
                </div>
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Hujan<br>P:0.81 R:0.61 F1:0.69
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # KNN
        st.markdown("""
        <div class="model-card">
            <div class="model-title">K-Nearest Neighbors (KNN)</div>
            <div class="model-acc">74.15%</div>
            <div class="metric-row">
                <div class="metric-box"><div class="ml">Precision</div><div class="mv">0.57</div></div>
                <div class="metric-box"><div class="ml">Recall</div><div class="mv">0.48</div></div>
                <div class="metric-box"><div class="ml">F1-Score</div><div class="mv">0.50</div></div>
                <div class="metric-box"><div class="ml">Support</div><div class="mv">147</div></div>
            </div>
            <div class="class-row">
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Berawan<br>P:0.77 R:0.91 F1:0.83
                </div>
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Cerah<br>P:0.25 R:0.07 F1:0.11
                </div>
                <div class="class-box" style="background:#FDFBFF; color:#4F3584; border: 1px solid #F0EAF8;">
                    Hujan<br>P:0.68 R:0.46 F1:0.55
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Kesimpulan evaluasi
        st.markdown('<p class="section-title" style="font-size:1.4rem;">Sintesis Kelayakan</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Arsitektur <b>Random Forest</b> dikonfirmasi sebagai instrumen optimal dengan verifikasi performa <b>78.91%</b>,
            lebih superior bila disandingkan dengan parameter Decision Tree (65.99%) dan limitasi KNN (74.15%).
            Eksekusi Random Forest membuktikan stabilitas unggul pada penentuan kelas Berawan (F1: 0.86)
            maupun identifikasi Hujan (F1: 0.69). Model valid ini telah diadopsi sebagai instrumen tunggal 
            pada konsol Prediksi.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MENU 6 — PREDIKSI
# ══════════════════════════════════════════════════════
elif menu_choice == "Prediksi":
    st.markdown('<p class="section-title">Konsol Prediksi Komprehensif</p>', unsafe_allow_html=True)

    mode = st.tabs(["Sistem Otomasi Kalender", "Input Variabel Manual"])

    # ── SUB TAB 1 — TANGGAL ──────────────────────────
    with mode[0]:
        st.markdown('<div class="mode-badge">Ekstraksi Variabel Cuaca via Jalur API Jaringan</div>', unsafe_allow_html=True)

        def ambil_data(tgl_mulai, tgl_selesai):
            hari_ini = date.today()
            hasil = {k: [] for k in ["time","temperature_2m_max","temperature_2m_min",
                                      "temperature_2m_mean","windspeed_10m_max",
                                      "relative_humidity_2m_max","relative_humidity_2m_min"]}
            daily_params = ["temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                            "precipitation_sum","windspeed_10m_max",
                            "relative_humidity_2m_max","relative_humidity_2m_min"]

            if tgl_mulai <= hari_ini:
                r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                    "latitude": -8.1845, "longitude": 113.6680,
                    "start_date": str(tgl_mulai),
                    "end_date": str(min(tgl_selesai, hari_ini)),
                    "daily": daily_params, "timezone": "Asia/Jakarta"
                })
                if r.status_code == 200:
                    d = r.json()["daily"]
                    for k in hasil: hasil[k].extend(d.get(k, []))

            if tgl_selesai > hari_ini:
                r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                    "latitude": -8.1845, "longitude": 113.6680,
                    "start_date": str(max(tgl_mulai, hari_ini + timedelta(days=1))),
                    "end_date": str(tgl_selesai),
                    "daily": daily_params,
                    "timezone": "Asia/Jakarta", "forecast_days": 16
                })
                if r.status_code == 200:
                    d = r.json()["daily"]
                    for k in hasil: hasil[k].extend(d.get(k, []))

            return hasil if hasil["time"] else None

        col1, col2 = st.columns(2)
        with col1:
            tgl_mulai = st.date_input("Tanggal Observasi Awal", value=date.today(),
                                       min_value=date(2020,1,1),
                                       max_value=date.today()+timedelta(days=15))
        with col2:
            tgl_selesai = st.date_input("Tanggal Observasi Akhir", value=date.today()+timedelta(days=2),
                                         min_value=date(2020,1,1),
                                         max_value=date.today()+timedelta(days=15))

        if tgl_selesai < tgl_mulai:
            st.error("Rentan tanggal yang diset invalid. Tanggal akhir harus setelah tanggal mulai.")
        else:
            jumlah = (tgl_selesai - tgl_mulai).days + 1
            st.info(f"Target Komputasi: Rentang {jumlah} hari peninjauan")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Jalankan Prediksi Sistem", key="btn_tanggal", use_container_width=True):
                with st.spinner("Sinkronisasi jaringan variabel observasi..."):
                    data = ambil_data(tgl_mulai, tgl_selesai)

                if not data:
                    st.error("Gangguan sinkronisasi jaringan. Mohon eksekusi ulang permintaannya.")
                else:
                    st.success(f"Sinkronisasi rampung. Melakukan komputasi untuk {jumlah} siklus harian.")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="section-title" style="font-size:1.4rem;">Output Hasil Prediksi</p>', unsafe_allow_html=True)

                    cerah = berawan = hujan = 0
                    for i in range(len(data["time"])):
                        tgl   = data["time"][i]
                        smax  = data["temperature_2m_max"][i]       or 32.0
                        smin  = data["temperature_2m_min"][i]       or 24.0
                        srata = data["temperature_2m_mean"][i]      or 28.0
                        angin = data["windspeed_10m_max"][i]        or 15.0
                        kmax  = data["relative_humidity_2m_max"][i] or 85.0
                        kmin  = data["relative_humidity_2m_min"][i] or 60.0

                        inp   = scaler.transform([[smax, smin, srata, angin, kmax, kmin]])
                        hasil = le.inverse_transform(model.predict(inp))[0]

                        if hasil == "Cerah":
                            hasil_display = "☀️ Cerah"
                            cls = "result-cerah"; cerah += 1
                        elif hasil == "Berawan":
                            hasil_display = "⛅ Berawan"
                            cls = "result-berawan"; berawan += 1
                        else:
                            hasil_display = "🌧️ Hujan"
                            cls = "result-hujan";  hujan += 1

                        st.markdown(f"""
                        <div class="result-card {cls}">
                            <div style="flex:1">
                                <div style="font-size:0.95rem; color:#6B5E82; font-weight:700; margin-bottom: 0.4rem;">{tgl}</div>
                                <div style="font-size: 1.3rem; font-weight: 800;">{hasil_display}</div>
                            </div>
                            <div style="font-size:0.9rem; color:#6B5E82; text-align:right; line-height: 1.6; font-weight:500;">
                                Suhu: {smin}° – {smax}° C<br>
                                Kelembaban: {kmin}% – {kmax}%<br>
                                Laju Angin: {angin} km/h
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if jumlah > 1:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<p class="section-title" style="font-size:1.4rem;">Kumulasi Tren</p>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="stat-row">
                            <div class="stat-box"><div class="label">☀️ Peluang Cerah</div><div class="value">{cerah} Hari</div></div>
                            <div class="stat-box"><div class="label">⛅ Peluang Berawan</div><div class="value">{berawan} Hari</div></div>
                            <div class="stat-box"><div class="label">🌧️ Peluang Hujan</div><div class="value">{hujan} Hari</div></div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── SUB TAB 2 — MANUAL ───────────────────────────
    with mode[1]:
        st.markdown('<div class="mode-badge">Input Dimensi Cuaca Sintetis Secara Mandiri</div>', unsafe_allow_html=True)

        jumlah_hari = st.radio("Jumlah Siklus Simulasi", [1, 3, 7], horizontal=True,
                                format_func=lambda x: f"{x} Siklus Hari")
        st.markdown("<br>", unsafe_allow_html=True)

        inputs = []
        for i in range(jumlah_hari):
            tgl   = date.today() + timedelta(days=i)
            label = "Siklus Terkini" if i == 0 else f"Siklus Lanjutan +{i}"
            st.markdown(f"<h3 style='font-size:1.15rem; color:#4F3584; font-weight: 800; margin-bottom: 1.2rem; border-bottom:1px solid #F0EAF8; padding-bottom:0.8rem;'>{label} — {tgl.strftime('%d %b %Y')}</h3>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                smax  = st.number_input("Suhu Maksimum (°C)",     20.0, 45.0,  32.0, 0.1, key=f"m_smax{i}")
                smin  = st.number_input("Suhu Minimum (°C)",     15.0, 35.0,  24.0, 0.1, key=f"m_smin{i}")
            with c2:
                srata = st.number_input("Rata-rata Suhu (°C)",    15.0, 40.0,  28.0, 0.1, key=f"m_srat{i}")
                angin = st.number_input("Laju Angin (km/h)",         0.0,100.0,  15.0, 0.1, key=f"m_angin{i}")
            with c3:
                kmax  = st.number_input("Titik Lembab Max (%)", 30.0,100.0,  85.0, 0.1, key=f"m_kmax{i}")
                kmin  = st.number_input("Titik Lembab Min (%)", 20.0,100.0,  60.0, 0.1, key=f"m_kmin{i}")

            inputs.append([smax, smin, srata, angin, kmax, kmin])
            if i < jumlah_hari - 1:
                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Jalankan Prediksi Sintetis", key="btn_manual", use_container_width=True):
            st.markdown('<p class="section-title" style="font-size:1.4rem;">Output Hasil Prediksi</p>', unsafe_allow_html=True)
            cerah = berawan = hujan = 0

            for i, inp in enumerate(inputs):
                tgl   = date.today() + timedelta(days=i)
                label = "Siklus Terkini" if i == 0 else f"Siklus Lanjutan +{i}"
                hasil = le.inverse_transform(model.predict(scaler.transform([inp])))[0]

                if hasil == "Cerah":
                    hasil_display = "☀️ Cerah"
                    cls = "result-cerah"; cerah += 1
                elif hasil == "Berawan":
                    hasil_display = "⛅ Berawan"
                    cls = "result-berawan"; berawan += 1
                else:
                    hasil_display = "🌧️ Hujan"
                    cls = "result-hujan";  hujan += 1

                st.markdown(f"""
                <div class="result-card {cls}">
                    <div>
                        <div style="font-size:0.95rem; font-weight:700; color:#6B5E82; margin-bottom: 0.4rem;">{label} — {tgl.strftime('%d %b %Y')}</div>
                        <div style="font-size: 1.3rem; font-weight: 800;">{hasil_display}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if jumlah_hari > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-title" style="font-size:1.4rem;">Kumulasi Tren</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-box"><div class="label">☀️ Peluang Cerah</div><div class="value">{cerah} Hari</div></div>
                    <div class="stat-box"><div class="label">⛅ Peluang Berawan</div><div class="value">{berawan} Hari</div></div>
                    <div class="stat-box"><div class="label">🌧️ Peluang Hujan</div><div class="value">{hujan} Hari</div></div>
                </div>
                """, unsafe_allow_html=True)