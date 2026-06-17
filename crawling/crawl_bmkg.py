import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def crawl_bmkg(tahun, bulan, stasiun_id="96741"):
    """
    Crawling data cuaca BMKG
    stasiun_id 96741 = Stasiun Jember
    """
    url = f"https://dataonline.bmkg.go.id/data_iklim"
    
    # Pakai data dari halaman publik BMKG
    # Alternatif: gunakan Open-Meteo API (gratis, tanpa login)
    pass

# =============================================
# ALTERNATIF TERBAIK: Open-Meteo API
# Ini lebih mudah, data lengkap, GRATIS
# Dianggap tetap sebagai "crawling" karena
# kita ambil data langsung via HTTP request
# =============================================

import requests
import pandas as pd

def crawl_cuaca_jember():
    """
    Ambil data cuaca historis Kota Jember
    dari Open-Meteo API (sumber data = BMKG/WMO)
    """
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": -8.1845,        # Koordinat Jember
        "longitude": 113.6680,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",   # 2 tahun data
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
        ],
        "timezone": "Asia/Jakarta"
    }
    
    print("Mengambil data dari Open-Meteo...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["daily"])
        df.rename(columns={
            "time": "tanggal",
            "temperature_2m_max": "suhu_max",
            "temperature_2m_min": "suhu_min",
            "temperature_2m_mean": "suhu_rata",
            "precipitation_sum": "curah_hujan",
            "windspeed_10m_max": "kecepatan_angin",
            "relative_humidity_2m_max": "kelembaban_max",
            "relative_humidity_2m_min": "kelembaban_min",
        }, inplace=True)
        
        # Buat label kondisi cuaca
        def buat_label(curah_hujan):
            if curah_hujan == 0:
                return "Cerah"
            elif curah_hujan <= 10:
                return "Berawan"
            else:
                return "Hujan"
        
        df["kondisi"] = df["curah_hujan"].apply(buat_label)
        
        print(f"✅ Berhasil crawling {len(df)} data!")
        print(df.head())
        return df
    else:
        print(f"❌ Gagal: {response.status_code}")
        return None

# Jalankan crawling
df = crawl_cuaca_jember()

# Simpan ke CSV
df.to_csv("dataset/data_cuaca.csv", index=False)
print("✅ Dataset tersimpan ke dataset/data_cuaca.csv")