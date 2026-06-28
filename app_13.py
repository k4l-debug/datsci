import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io  
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================================================================
# AMANKAN CONFIG: Terapkan tema Seaborn Global & Matikan Auto-Crop Sistem
# =========================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['savefig.bbox'] = None
plt.rcParams['figure.autolayout'] = False

# Setup halaman biar melebar kesamping (dashboard style)
st.set_page_config(page_title="Prediksi Harga Properti", layout="wide")

# =========================================================================
# SUNTIKAN CSS REVISI: Judul Adaptif Otomatis Mengikuti Tema Terang/Gelap
# =========================================================================
st.markdown("""
    <style>
    /* Paksa kontainer utama metrik jadi flex kolom rata tengah */
    [data-testid="stMetric"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        width: 100% !important;
    }
    
    /* Rata tengahin judul besaran */
    [data-testid="stMetricLabel"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    /* FIX ADAPTIF: Hapus hardcode warna putih agar otomatis ikut tema browser */
    [data-testid="stMetricLabel"] > div {
        font-size: 1.1rem !important; /* Huruf judul tetep diperbesar */
        font-weight: 600 !important;   /* Tetep tebal dan tegas */
    }
    
    /* Rata tengahin angka dan matikan paksa potong titik-titik (...) */
    [data-testid="stMetricValue"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.55rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOAD & TRAIN (LOG TRANSFORM + MULTI-PIPE)
# ==========================================
@st.cache_resource
def load_and_train_all():
    # Load absolute raw data untuk tracking perbandingan
    df_raw = pd.read_csv('case7_house_price.csv')
    
    df = df_raw.copy()
    df = df.dropna(subset=['harga_idr']).copy()
    
    # KUNCI 1: Bersihkan typo karakter '@' dulu sebelum konversi nan
    def pre_clean(x):
        if pd.isna(x): return np.nan
        return str(x).lower().replace('@', 'a').strip()
    df['kondisi'] = df['kondisi'].apply(pre_clean)
    
    # KUNCI 2: Jalankan Imputasi Modus DULUAN biar nilai 'nan' asli keisi data normal
    cat_cols = ['kota', 'kondisi', 'garasi', 'jenis_sertifikat', 'dekat_sekolah', 'dekat_rs', 'dekat_mall']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.capitalize().replace({'Nan': np.nan, 'None': np.nan})
        df[col] = df[col].fillna(df[col].mode()[0])

    df['jenis_sertifikat'] = df['jenis_sertifikat'].replace({'Hgb': 'Shgb'})
        
    # KUNCI 3: Baru kelompokkan ke kategori final (Kata 'Lainnya' gak akan muncul dari nan lagi)
    def final_clean_kondisi(x):
        x = str(x).lower()
        if 'baru' in x: return 'Baru'
        if 'bagus' in x: return 'Bagus'
        if 'cukup' in x: return 'Cukup'
        if 'renov' in x: return 'Perlu Renovasi'
        return 'Bagus' # Fallback langsung ke modus teraman jika ada sisa aneh
    df['kondisi'] = df['kondisi'].apply(final_clean_kondisi)

    X = df.drop(columns=['property_id', 'harga_idr', 'harga_miliar'], errors='ignore')
    y = np.log1p(df['harga_idr'])
    
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols_ohe = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor_scaled = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols_ohe)
    ])
    
    preprocessor_raw = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols_ohe)
    ])
    
    models = {
        "Linear Regression": (LinearRegression(), preprocessor_scaled),
        "Ridge": (Ridge(alpha=1.0), preprocessor_scaled),
        "Lasso": (Lasso(alpha=0.1), preprocessor_scaled),
        "Random Forest": (RandomForestRegressor(random_state=42), preprocessor_raw),
        "XGBoost": (XGBRegressor(random_state=42), preprocessor_raw),
        "LGBM": (LGBMRegressor(random_state=42, verbose=-1), preprocessor_raw),
        "CatBoost": (CatBoostRegressor(random_state=42, verbose=0), preprocessor_raw)
    }
    
    trained_models = {}
    results = []
    predictions_dict = {}  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, (model, pre) in models.items():
        pipe = Pipeline([('pre', pre), ('reg', model)])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe
        
        # Hitung R^2 jujur (skala Rupiah asli balik dari Log)
        y_pred_log = pipe.predict(X_test)
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred_log)
        
        score = r2_score(y_test_real, y_pred_real)
        # REVISI: Mengubah key dari "R^2" menjadi pangkat murni "R²"
        results.append({"Model": name, "R²": score})
        
        # Simpan nilai riil test untuk keperluan plot analitik residual
        predictions_dict[name] = {
            "actual": y_test_real.values,
            "predicted": y_pred_real
        }
        
    # REVISI: Sorting berdasarkan kolom baru "R²"
    return trained_models, df_raw, df, pd.DataFrame(results).sort_values("R²", ascending=False), predictions_dict
   
models, df_raw, df_clean, df_results, predictions_dict = load_and_train_all()

# ==========================================
# 2. UI STREAMLIT (TABS INTERAKTIF)
# ==========================================
st.title("🏡 Dashboard Analisis & Kalkulator Harga Properti")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard EDA", "🧹 Data Preparation", "🤖 Model Comparison", "🔮 Estimasi Harga"])

# --- TAB 1: DASHBOARD EDA ---
with tab1:
    st.header("Dashboard Analisis Ekplorasi Data (EDA)")
    
    # Metrik Atas (Layout 5 Kolom: Menggunakan Harga Tertinggi & Terendah)
    m1, m2, m3, m4, m5 = st.columns(5, gap="large")
    m1.metric("Total Sampel Rumah", f"{len(df_clean):,} unit")
    m2.metric("Rata-rata Harga", f"Rp {df_clean['harga_idr'].mean():,.0f}")
    m3.metric("Harga Tertinggi", f"Rp {df_clean['harga_idr'].max():,.0f}")
    m4.metric("Harga Terendah", f"Rp {df_clean['harga_idr'].min():,.0f}")
    m5.metric("Lokasi Terbanyak", df_clean['kota'].mode()[0])
    
    st.markdown("---")
    
    # Menampilkan kondisi data mentah sebelum cleaning
    st.subheader("⚠️ Masalah Kualitas Pada Dataset Mentah (Raw Data Issues)")
    col_raw1, col_raw2 = st.columns(2)
    
    with col_raw1:
        st.write("**1. Log Missing Value pada Dataset Awal:**")
        missing_raw = df_raw.isnull().sum()
        df_missing_raw = pd.DataFrame({"Nama Kolom": missing_raw.index, "Jumlah Baris Kosong (NaN)": missing_raw.values})
        st.dataframe(df_missing_raw, use_container_width=True, hide_index=True, height=561)
        
    with col_raw2:
        st.write("**2. Log Nilai Unik Kolom Kategorikal (Banyak Typo & Inkonsistensi):**")
        cat_cols_track = ['kota', 'kondisi', 'garasi', 'jenis_sertifikat', 'dekat_sekolah', 'dekat_rs', 'dekat_mall']
        raw_output_text = ""
        for col in cat_cols_track:
            if col in df_raw.columns:
                raw_output_text += f"--- Nilai Unik di Kolom '{col}' ---\n{df_raw[col].unique().tolist()}\n\n"
        st.text(raw_output_text)
        
    st.markdown("---")
    
    # Filter Interaktif
    st.subheader("Interactive Data Filter")
    pilihan_kota = st.multiselect("Pilih Kota yang Ingin Dianalisis:", options=df_clean['kota'].unique(), default=df_clean['kota'].unique())
    df_filtered = df_clean[df_clean['kota'].isin(pilihan_kota)].copy()
    
    # Pembagian data ke skala Miliar biar teks kaku 1e10 hilang selamanya
    df_filtered['harga_miliar'] = df_filtered['harga_idr'] / 1e9

    # =========================================================================
    # BARIS 1: COLS TERPISAH + SEABORN STYLE + KOTAK PUTIH KEMBAR SIAM 100% RATA
    # =========================================================================
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        
        sns.histplot(data=df_filtered, x='harga_miliar', kde=True, ax=ax1, color='#2ca02c', alpha=0.5)
        sns.despine(left=True, bottom=True)
        
        ax1.set_title("Distribusi Harga Rumah Asli (Miliar Rp)", fontsize=11, fontweight='bold', pad=15)
        ax1.set_xlabel("Harga (Miliar Rp)", fontsize=9.5)
        ax1.set_ylabel("Count", fontsize=9.5)
        
        fig1.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.95)
        
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=200, bbox_inches=None) 
        st.image(buf1, use_container_width=True)
        plt.close(fig1)
        
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        
        sns.barplot(data=df_filtered, x='kota', y='harga_miliar', ax=ax2, color='#1f77b4', errorbar=None)
        sns.despine(left=True, bottom=True)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        ax2.set_title("Rata-rata Harga Jual Berdasarkan Wilayah", fontsize=11, fontweight='bold', pad=15)
        ax2.set_xlabel("Kota", fontsize=9.5)
        ax2.set_ylabel("Rata-rata Harga (Miliar Rp)", fontsize=9.5)
        
        fig2.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.95)
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=200, bbox_inches=None)
        st.image(buf2, use_container_width=True)
        plt.close(fig2)
        
    st.markdown("---")
    
    # =========================================================================
    # BARIS 2: COLS TERPISAH + SEABORN STYLE + KOTAK PUTIH KEMBAR SIAM 100% RATA
    # =========================================================================
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Matriks Korelasi Variabel Numerik")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        
        num_features = df_filtered.select_dtypes(include=['float64', 'int64']).drop(columns=['harga_miliar'], errors='ignore')
        sns.heatmap(num_features.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3, cbar=False)
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        
        fig3.subplots_adjust(bottom=0.32, top=0.88, left=0.30, right=0.95)
        
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', dpi=200, bbox_inches=None)
        st.image(buf3, use_container_width=True)
        plt.close(fig3)
        
    with col4:
        st.subheader("Deteksi Rentang Outlier Fitur (Skala Standar)")
        fig4, ax4 = plt.subplots(figsize=(6, 5)) 
        
        fitur_box = ['jarak_pusat_kota_km', 'harga_idr', 'tahun_dibangun', 'jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'jumlah_lantai', 'luas_bangunan_m2', 'luas_tanah_m2']
        scaler_box = StandardScaler()
        data_scaled = pd.DataFrame(scaler_box.fit_transform(df_filtered[fitur_box]), columns=fitur_box)
        
        # Boxplot Seaborn murni
        sns.boxplot(data=data_scaled, ax=ax4, palette='Set2')
        sns.despine(left=True, bottom=True)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_ylabel("Nilai Standar (Z-Score)", fontsize=9.5)
        
        fig4.subplots_adjust(bottom=0.32, top=0.88, left=0.24, right=0.95)
        
        buf4 = io.BytesIO()
        fig4.savefig(buf4, format='png', dpi=200, bbox_inches=None)
        st.image(buf4, use_container_width=True)
        plt.close(fig4)

    st.markdown("---")
    
    # Rekap Analisis Outlier IQR Data Asli (Fitur Numerik)
    st.subheader("📋 Ringkasan Deteksi Outlier Metode IQR (Data Filtered)")
    num_cols_iqr = df_filtered.select_dtypes(include=['float64', 'int64']).columns.drop('harga_miliar', errors='ignore')
    iqr_summary = []
    for col in num_cols_iqr:
        q1 = df_filtered[col].quantile(0.25)
        q3 = df_filtered[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        total_outliers = ((df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)).sum()
        pct_outliers = (total_outliers / len(df_filtered)) * 100
        iqr_summary.append({
            "Nama Variabel": col,
            "Batas Bawah": f"{lower_bound:,.1f}",
            "Batas Atas": f"{upper_bound:,.1f}",
            "Jumlah Outlier": f"{total_outliers} unit",
            "Persentase Outlier": f"{pct_outliers:.2f}%"
        })
    st.table(pd.DataFrame(iqr_summary))

    st.markdown("---")
    if st.checkbox("Tampilkan Tabel Ringkasan Statistik Deskriptif Lengkap"):
        st.write("Statistik Fitur Numerik Ter-filter:")
        st.dataframe(df_filtered.describe())


# --- TAB 2: DATA PREPARATION ---
with tab2:
    st.header("Data Preparation & Cleaning Log")
    
    # --------------------------------=========================================
    # SEKSI 1: Handling Missing Value
    # --------------------------------=========================================
    st.subheader("🧼 1. Komparasi Sebelum & Sesudah Handling Missing Value")
    missing_comp = pd.DataFrame({
        "Nama Fitur/Kolom": df_raw.columns,
        "Missing Value (Sebelum)": df_raw.isnull().sum().values,
        "Missing Value (Sesudah)": 0 
    })
    st.table(missing_comp)
    
    st.markdown("---")
    
    # --------------------------------=========================================
    # SEKSI 2: Analisis Transformasi Logaritma
    # --------------------------------=========================================
    st.subheader("🔄 2. Analisis Transformasi Logaritma pada Target (Harga)")
    
    # Plot Bentuk Distribusi (Histogram Atas)
    fig_log, (ax_bef, ax_aft) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_clean['harga_idr'], kde=True, ax=ax_bef, color='orange')
    ax_bef.set_title("Sebelum Log (Skewed Positif)")
    ax_bef.set_xlabel("Harga Asli (Rp)")
    
    sns.histplot(np.log1p(df_clean['harga_idr']), kde=True, ax=ax_aft, color='purple')
    ax_aft.set_title("Sesudah Log (Mendekati Normal)")
    ax_aft.set_xlabel("Log(Harga)")
    sns.despine(fig=fig_log)
    fig_log.tight_layout()
    st.pyplot(fig_log, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Plot Boxplot (Berada di Bawah Histogram)
    fig_box_log, ax_box_log = plt.subplots(figsize=(14, 5))
    sns.boxplot(x=np.log1p(df_clean['harga_idr']), ax=ax_box_log, color='plum')
    ax_box_log.set_title("Boxplot Distribusi 'harga_idr' (Skala Logaritma)")
    ax_box_log.set_xlabel("Log(Harga)")
    sns.despine(fig=fig_box_log)
    fig_box_log.tight_layout()
    st.pyplot(fig_box_log, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabel Hasil Cek Outlier Setelah Log
    st.write("**📋 Ringkasan Deteksi Outlier Target 'harga_idr' Setelah Skala Logaritma**")
    log_target = np.log1p(df_clean['harga_idr'])
    q1_log = log_target.quantile(0.25)
    q3_log = log_target.quantile(0.75)
    iqr_log = q3_log - q1_log
    lb_log = q1_log - (1.5 * iqr_log)
    ub_log = q3_log + (1.5 * iqr_log)
    
    outliers_log_count = ((log_target < lb_log) | (log_target > ub_log)).sum()
    pct_outliers_log = (outliers_log_count / len(df_clean)) * 100
    
    log_summary_data = [{
        "Nama Variabel": "harga_idr (Setelah Transformasi Log)",
        "Batas Bawah (Skala Log)": f"{lb_log:.4f}",
        "Batas Atas (Skala Log)": f"{ub_log:.4f}",
        "Jumlah Outlier": f"{outliers_log_count} unit",
        "Persentase Outlier": f"{pct_outliers_log:.2f}%"
    }]
    st.table(pd.DataFrame(log_summary_data))
    
    # Tabel Hasil Cek Outlier Sebelum Log
    st.write("**📋 Ringkasan Deteksi Outlier Target 'harga_idr' Sebelum Skala Logaritma**")
    raw_target = df_clean['harga_idr']
    q1_raw = raw_target.quantile(0.25)
    q3_raw = raw_target.quantile(0.75)
    iqr_raw = q3_raw - q1_raw
    lb_raw = q1_raw - (1.5 * iqr_raw)
    ub_raw = q3_raw + (1.5 * iqr_raw)
    
    outliers_raw_count = ((raw_target < lb_raw) | (raw_target > ub_raw)).sum()
    pct_outliers_raw = (outliers_raw_count / len(df_clean)) * 100
    
    raw_summary_data = [{
        "Nama Variabel": "harga_idr (Sebelum Transformasi Log)",
        "Batas Bawah (Rupiah)": f"Rp {lb_raw:,.1f}",
        "Batas Atas (Rupiah)": f"Rp {ub_raw:,.1f}",
        "Jumlah Outlier": f"{outliers_raw_count} unit",
        "Persentase Outlier": f"{pct_outliers_raw:.2f}%"
    }]
    st.table(pd.DataFrame(raw_summary_data))
    
    st.info("💡 **Insight Penting:** Perhatikan perbandingan kedua tabel di atas! Sebelum dilakukan transformasi log, data target memiliki sebaran pencilan yang sangat agresif akibat rumah-rumah bernilai ekstrem tinggi. Setelah dieksekusi menggunakan logaritma, jangkauan sebaran ditekan secara matematis sehingga model regresi tidak mudah bias.")
    
    st.markdown("---")
    
    # --------------------------------=========================================
    # SEKSI 3: Handling Typo & Inkonsistensi Kategori
    # --------------------------------=========================================
    st.subheader("🔤 3. Komparasi Sebelum & Sesudah Handling Typo & Inkonsistensi Kategori")
    cat_cols_track = ['kota', 'kondisi', 'garasi', 'jenis_sertifikat', 'dekat_sekolah', 'dekat_rs', 'dekat_mall']
    
    for col_cat in cat_cols_track:
        with st.expander(f"🔍 Klik untuk Detail Komparasi Kolom: '{col_cat}'"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**Nilai Unik (Sebelum Cleaning):**")
                st.text(df_raw[col_cat].unique().tolist())
            with col_c2:
                st.write(f"**Nilai Unik (Sesudah Cleaning):**")
                st.text(df_clean[col_cat].unique().tolist())
                
    st.markdown("---")
    st.subheader("Sampel Data Bersih")
    st.write("Menampilkan 5 baris pertama data setelah di-imputasi modus/median and dibersihkan dari karakter typo:")
    st.dataframe(df_clean.head())


# --- TAB 3: MODEL COMPARISON ---
with tab3:
    st.header("Perbandingan Performa Model (R² Score)")
    st.write("Skor di bawah dihitung menggunakan nominal asli (Rupiah) setelah di-inverse dari bentuk logaritma:")
    st.table(df_results)
    # REVISI: Mengubah teks output sukses agar mengambil nama kolom baru "R²"
    st.success(f"Model dengan Akurasi Tertinggi: {df_results.iloc[0]['Model']}")

    st.markdown("---")
    
    st.subheader("📊 Analisis Kesalahan & Residual Prediksi (Error Analysis)")
    pilihan_model_res = st.selectbox("Pilih Model untuk Ditinjau Analisis Erornya:", options=df_results["Model"].tolist())
    
    # Tarik data aktual vs prediksi
    y_aktual = predictions_dict[pilihan_model_res]["actual"]
    y_prediksi = predictions_dict[pilihan_model_res]["predicted"]
    
    # Skala Miliar untuk scatter plots bawaan agar axis-nya estetik lurus rata
    y_aktual_miliar = y_aktual / 1e9
    y_prediksi_miliar = y_prediksi / 1e9
    nilai_residual_miliar = y_aktual_miliar - y_prediksi_miliar
    
    # Hitung nilai persentase meleset absolut untuk fitur baru (Nomor 1 & 2)
    pct_error = (np.abs(y_aktual - y_prediksi) / y_aktual) * 100
    
    # Pengelompokan Kategori Toleransi Akurasi (Nomor 1)
    mepet = np.sum(pct_error < 10)
    wajar = np.sum((pct_error >= 10) & (pct_error <= 20))
    jauh = np.sum(pct_error > 20)
    total_test = len(pct_error)
    
    df_toleransi = pd.DataFrame({
        "Kategori": ["(<10%)", "(10%-20%)", "(>20%)"],
        "Persentase": [(mepet/total_test)*100, (wajar/total_test)*100, (jauh/total_test)*100]
    })
    
    # =========================================================================
    # BARIS 1 RESIDUAL: SCATTER PLOTS (DIJAMIN 100% KEMBAR SIAM SEJAJAR)
    # =========================================================================
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        fig_res1, ax_res1 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=y_aktual_miliar, y=y_prediksi_miliar, ax=ax_res1, color='#1f77b4', alpha=0.6, edgecolor='w')
        garis_ideal = [min(y_aktual_miliar), max(y_aktual_miliar)]
        ax_res1.plot(garis_ideal, garis_ideal, color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
        sns.despine(left=True, bottom=True)
        
        ax_res1.set_title(f"Actual vs. Predicted - {pilihan_model_res}", fontsize=11, fontweight='bold', pad=15)
        ax_res1.set_xlabel("Nilai Aktual (Miliar Rp)", fontsize=9.5)
        ax_res1.set_ylabel("Nilai Prediksi Model (Miliar Rp)", fontsize=9.5)
        ax_res1.legend(fontsize=8.5)
        
        fig_res1.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.95)
        buf_res1 = io.BytesIO()
        fig_res1.savefig(buf_res1, format='png', dpi=200, bbox_inches=None) 
        st.image(buf_res1, use_container_width=True)
        plt.close(fig_res1)
        
    with col_res2:
        fig_res2, ax_res2 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=y_prediksi_miliar, y=nilai_residual_miliar, ax=ax_res2, color='#e377c2', alpha=0.6, edgecolor='w')
        ax_res2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        sns.despine(left=True, bottom=True)
        
        ax_res2.set_title(f"Residual Plot - {pilihan_model_res}", fontsize=11, fontweight='bold', pad=15)
        ax_res2.set_xlabel("Nilai Prediksi Model (Miliar Rp)", fontsize=9.5)
        ax_res2.set_ylabel("Residual / Galat (Miliar Rp)", fontsize=9.5)
        
        fig_res2.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.95)
        buf_res2 = io.BytesIO()
        fig_res2.savefig(buf_res2, format='png', dpi=200, bbox_inches=None) 
        st.image(buf_res2, use_container_width=True)
        plt.close(fig_res2)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # =========================================================================
    # BARIS 2 RESIDUAL: FITUR 1 & FITUR 2 REVISI (DIJAMIN 100% KEMBAR SIAM SEJAJAR)
    # =========================================================================
    col_res3, col_res4 = st.columns(2)
    with col_res3:
        fig_res3, ax_res3 = plt.subplots(figsize=(6, 5))
        
        # Plot Bar Chart Analisis Toleransi Akurasi
        sns.barplot(data=df_toleransi, x='Kategori', y='Persentase', ax=fig_res3.gca(), palette=['#2ca02c', '#ff7f0e', '#d62728'])
        sns.despine(left=True, bottom=True)
        
        ax_res3.set_title("Analisis Toleransi Akurasi (Kecil vs Besar)", fontsize=11, fontweight='bold', pad=15)
        ax_res3.set_xlabel("Kategori Kedekatan Prediksi", fontsize=9.5)
        ax_res3.set_ylabel("Persentase dari Total Data (%)", fontsize=9.5)
        ax_res3.set_ylim(0, 100)
        
        # Tempel label persentase teks di atas bar biar kebaca presisi
        for p in ax_res3.patches:
            ax_res3.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                             ha='center', va='center', fontsize=9, fontweight='bold')
            
        fig_res3.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.95)
        buf_res3 = io.BytesIO()
        fig_res3.savefig(buf_res3, format='png', dpi=200, bbox_inches=None) 
        st.image(buf_res3, use_container_width=True)
        plt.close(fig_res3)
        
    with col_res4:
        fig_res4, ax_res4 = plt.subplots(figsize=(6, 5))
        
        # Plot Histogram Distribusi Persentase Meleset
        sns.histplot(pct_error, kde=True, ax=ax_res4, color='#bcbd22', alpha=0.5)
        sns.despine(left=True, bottom=True)
        
        ax_res4.set_title("Distribusi Persentase Meleset (Percentage Error)", fontsize=11, fontweight='bold', pad=15)
        ax_res4.set_xlabel("Persentase Meleset / Error (%)", fontsize=9.5)
        ax_res4.set_ylabel("Count", fontsize=9.5)
        
        fig_res4.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.95)
        buf_res4 = io.BytesIO()
        fig_res4.savefig(buf_res4, format='png', dpi=200, bbox_inches=None) 
        st.image(buf_res4, use_container_width=True)
        plt.close(fig_res4)


# --- TAB 4: ESTIMASI HARGA OTOMATIS ---
with tab4:
    st.header("🔮 Kalkulator Estimasi Harga Otomatis")
    best_model_name = df_results.iloc[0]['Model']
    best_model = models[best_model_name]
    st.info(f"Kalkulator ini bekerja menggunakan model terbaik otomatis saat ini: **{best_model_name}**")
    
    input_vals = {}
    
    # FIX 1: Buang 'property_id' dari kalkulator karena gak dipake training dan bikin eror median
    feature_columns = df_clean.drop(columns=['property_id', 'harga_idr', 'harga_miliar'], errors='ignore').columns
    
    col_layout1, col_layout2 = st.columns(2)
    for i, col in enumerate(feature_columns):
        if col in df_clean.columns:
            with col_layout1 if i % 2 == 0 else col_layout2:
                # FIX 2: Pake check khusus numeric bawaan pandas biar anti-gagal di server online
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    input_vals[col] = st.number_input(f"Masukkan {col.replace('_', ' ').title()}:", value=float(df_clean[col].median()))
                else:
                    input_vals[col] = st.selectbox(f"Pilih {col.replace('_', ' ').title()}:", options=df_clean[col].unique())
            
    st.markdown("---")
    if st.button("🚀 Hitung Estimasi Harga Rumah"):
        input_df = pd.DataFrame([input_vals])
        input_df = input_df[feature_columns]
        
        log_prediksi = best_model.predict(input_df)[0]
        harga_final = np.expm1(log_prediksi)
        
        st.success(f"### Estimasi Nilai Pasar Properti: Rp {harga_final:,.0f}")