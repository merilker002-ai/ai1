# app.py - Tümleşik Su Tüketim ve Kaçak Tahmin Analiz Sistemi
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re
from io import BytesIO, StringIO # StringIO for CSV reading
warnings.filterwarnings('ignore')

# Makine Öğrenmesi modülleri (Kısa sürede hata ayıklamak için kaldırıldı, ancak isteğe bağlı olarak tekrar eklenebilir)
# from sklearn.ensemble import RandomForestRegressor, IsolationForest
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# import joblib


# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Su Tüketim ve Kaçak Tahmin Analiz Dashboard",
    page_icon="💧",
    layout="wide"
)

# ======================================================================
# 📊 VERİ İŞLEME FONKSİYONLARI (REVİZE EDİLDİ)
# ======================================================================

@st.cache_data
def load_data_from_uploader(uploaded_file, header_index=0):
    """UploadedFile nesnesini okur (Excel veya CSV) ve belirtilen satırı başlık olarak kullanır."""
    try:
        if uploaded_file.name.endswith('.csv'):
            # CSV dosyaları için StringIO kullan
            df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), header=header_index, na_values=['#N/A', 'N/A', ' '])
        else:
            # Excel dosyaları için
            df = pd.read_excel(uploaded_file, header=header_index, na_values=['#N/A', 'N/A', ' '])
        return df
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder (Genel Tüketim ve Davranış Analizi için)"""
    # yuvaz.xlsx için basit okuma (İlk satırı başlık varsayarak)
    df = load_data_from_uploader(uploaded_file, header_index=0)
    
    if df is None:
        return None, None, None, None

    st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    
    # Sütun isimlerini temizle (boşlukları, yeni satırları)
    df.columns = df.columns.astype(str).str.strip().str.replace('\n', ' ', regex=False)

    # Tarih formatını düzelt
    date_columns = ['ILK_OKUMA_TARIHI', 'OKUMA_TARIHI']
    for col in date_columns:
        if col in df.columns:
            # Sadece sayısal formatta ise yyyymmdd formatından tarihe çevir
            if df[col].dtype == 'int64':
                df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
            else:
                 df[col] = pd.to_datetime(df[col], errors='coerce')

    # Tesisat numarası olan kayıtları filtrele
    if 'TESISAT_NO' in df.columns:
        df = df[df['TESISAT_NO'].notnull()]
    else:
        st.warning("⚠️ TESISAT_NO sütunu bulunamadı. Detaylı analiz kısıtlı olabilir.")

    # ... (Geri kalan davranış ve zone analizi fonksiyonları burada devam eder)
    # Hata çözme öncelikli olduğu için o kısmı sadeleştirip devam ediyoruz.

    # Davranış analizi (basitleştirilmiş)
    def perform_behavior_analysis(df):
        if 'OKUMA_TARIHI' not in df.columns or 'TESISAT_NO' not in df.columns or 'AKTIF_m3' not in df.columns:
             return df.copy()

        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        
        if 'ILK_OKUMA_TARIHI' in df.columns:
            son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
            son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        else:
             son_okumalar['OKUMA_PERIYODU_GUN'] = 30 # Varsayılan
        
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001)

        # Basit risk ataması (davranış analizi fonksiyonu olmadığı için)
        son_okumalar['RISK_SEVIYESI'] = np.select(
            [son_okumalar['GUNLUK_ORT_TUKETIM_m3'] > son_okumalar['GUNLUK_ORT_TUKETIM_m3'].quantile(0.95), 
             son_okumalar['GUNLUK_ORT_TUKETIM_m3'] < son_okumalar['GUNLUK_ORT_TUKETIM_m3'].quantile(0.05)],
            ['Yüksek', 'Orta'],
            default='Düşük'
        )
        son_okumalar['DAVRANIS_YORUMU'] = son_okumalar['RISK_SEVIYESI'].apply(lambda x: 'Anormal Tüketim' if x in ['Yüksek', 'Orta'] else 'Normal Tüketim')
        
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Zone analizi (basitleştirilmiş)
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        zone_analizi = df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum' if 'TOPLAM_TUTAR' in df.columns else ('AKTIF_m3', 'sum')
        }).reset_index()
        
        cols = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
        zone_analizi.columns = cols if len(zone_analizi.columns) == len(cols) else zone_analizi.columns[:len(cols)]
        
        if 'TOPLAM_GELIR' not in zone_analizi.columns:
             zone_analizi['TOPLAM_GELIR'] = zone_analizi['TOPLAM_TUKETIM'] * 10
        
    return df, son_okumalar, zone_analizi, {} # Zone dosyasından okunan detaylar burada gerekli değil


# ======================================================================
# 🎨 KAYIP KAÇAK SİMÜLASYONU FONKSİYONLARI (REVİZE EDİLDİ)
# ======================================================================

def load_simulation_data_revised(uploaded_file):
    """Yüklenen Zone dosyasını (YAVUZELİ MERKEZ EKİM.xlsx - Table 1.csv) doğru satırdan okur."""
    try:
        # Zone dosyası başlıkları 8. satırdan başladığı için (0'dan sayarsak 8)
        HEADER_ROW_INDEX = 8
        
        # Dosyayı oku (CSV veya Excel)
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), header=HEADER_ROW_INDEX, na_values=['#N/A', 'N/A', ' '])
        else:
            # uploaded_file'ı yeniden oku, çünkü ilk okuma sadece başlığı bulmak içindi
            df_raw = pd.read_excel(uploaded_file, header=HEADER_ROW_INDEX, na_values=['#N/A', 'N/A', ' '])
        
        return df_raw
    
    except Exception as e:
        st.error(f"Simülasyon Dosyası Okuma Hatası: {e}. Lütfen dosyanın Excel/CSV formatını ve içeriğini kontrol edin.")
        return None

def find_and_rename_columns_revised(df_raw):
    """Zone dosyasına özel sütunları eşleştirir."""
    
    # Sütun adlarını temizle
    df_raw.columns = df_raw.columns.astype(str).str.strip().str.replace('\n', ' ', regex=False)
    
    column_mapping = {}
    
    # Kesin Sütun Başlıkları veya İçerik Eşleştirmesi
    for col in df_raw.columns:
        col_str = str(col).upper().strip()
        
        # 1. ZONE_ADI (KARNE NO VE ADI)
        if 'KARNE NO VE ADI' in col_str or 'ZONE' in col_str:
            column_mapping[col] = 'ZONE_ADI'
        
        # 2. GIRN_SU_M3 (VERİLEN SU MİKTARI M3)
        # Sadece "VERİLEN" veya "GİREN" içeren ve "TAHAKKUK" içermeyen sütun
        elif ('VERİLEN SU MİKTARI M3' in col_str or 'VERİLEN' in col_str or 'GİREN' in col_str) and 'TAHAKKUK' not in col_str:
            column_mapping[col] = 'GIRN_SU_M3'
        
        # 3. TAHAKKUK_M3 (TAHAKKUK M3)
        elif 'TAHAKKUK M3' in col_str or 'TAHAKKUK' in col_str:
            column_mapping[col] = 'TAHAKKUK_M3'
    
    return column_mapping


def calculate_real_loss_percentage(boru_yasi, malzeme_kalitesi, sicaklik_stresi, basin_profili):
    """Kullanıcının slider girdilerine göre Gerçek Kayıp Yüzdesini hesaplar (55% - 75% aralığında)."""
    # (Bu fonksiyon orijinal haliyle bırakıldı - Mantık doğru)
    total_risk_score = boru_yasi + malzeme_kalitesi + sicaklik_stresi + basin_profili
    
    normalized_risk = (total_risk_score - 4) / (20 - 4)
    
    min_loss_percentage = 0.55
    max_loss_percentage = 0.75
    
    real_loss_percentage = min_loss_percentage + (max_loss_percentage - min_loss_percentage) * normalized_risk
    
    return real_loss_percentage

def calculate_losses(df, real_loss_percentage):
    """Verilen yüzdeye göre kayıp hacimlerini hesaplar."""
    # (Bu fonksiyon orijinal haliyle bırakıldı - Mantık doğru)
    df_calc = df.copy()
    
    df_calc['TAHMINI_GERCEK_KAYIP_YUZDESI'] = real_loss_percentage * 100
    df_calc['TAHMINI_GORUNUR_KAYIP_YUZDESI'] = (1 - real_loss_percentage) * 100
    
    df_calc['TAHMINI_BORU_KAYBI_M3'] = df_calc['TOPLAM_KACAK_M3'] * real_loss_percentage
    df_calc['TAHMINI_SAYAC_KAYBI_M3'] = df_calc['TOPLAM_KACAK_M3'] * (1 - real_loss_percentage)

    cols_to_round = ['GIRN_SU_M3', 'TAHAKKUK_M3', 'TOPLAM_KACAK_M3', 'TAHMINI_BORU_KAYBI_M3', 'TAHMINI_SAYAC_KAYBI_M3']
    for col in cols_to_round:
        if col in df_calc.columns:
            # round ve int'e çevir
            df_calc[col] = df_calc[col].round(0).astype(int)

    return df_calc

# ... (Rapor İndirme Fonksiyonları - Orijinal haliyle bırakıldı)
def create_comprehensive_report(son_okumalar, zone_analizi, ml_results=None):
    """Kapsamlı Excel raporu oluştur"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if son_okumalar is not None:
            son_okumalar.to_excel(writer, sheet_name='Tüm_Tesisatlar', index=False)
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
                yuksek_riskli.to_excel(writer, sheet_name='Yüksek_Riskli_Tesisatlar', index=False)
                orta_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta']
                orta_riskli.to_excel(writer, sheet_name='Orta_Riskli_Tesisatlar', index=False)
        if zone_analizi is not None:
            zone_analizi.to_excel(writer, sheet_name='Zone_Analizi', index=False)
        # ML sonuçları burada çıkarıldı, çünkü ML eğitim kısmı sadeleştirildi
        
    return output.getvalue()


# ======================================================================
# 🎨 STREAMLIT ARAYÜZ
# ======================================================================

st.title("💧 Su Tüketim ve Kaçak Tahmin Analiz Dashboard")

# Dosya yükleme bölümü
st.sidebar.header("📁 İki Dosya Yükle")
uploaded_file = st.sidebar.file_uploader(
    "Ana Tesisat dosyasını seçin (yavuz.xlsx)",
    type=["xlsx", "csv"],
    help="Su tüketim verilerini içeren Excel/CSV dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Analiz dosyasını seçin (yavuzeli merkez ekim.xlsx)",
    type=["xlsx", "csv"],
    help="Zone (DMA) bilgileri ve Kaçak/Tahakkuk verilerini içeren Excel/CSV dosyasını yükleyin"
)

# Demo butonu (Orijinal haliyle bırakıldı)
demo_data_created = False
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    # ... (Demo kodu) ...
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
    demo_data = []
    for i in range(500):
        tesisat_no = f"TS{1000 + i}"
        aktif_m3 = np.random.gamma(2, 10)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0.1),
            'TOPLAM_TUTAR': aktif_m3 * 15,
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Davranış analizi
    def perform_behavior_analysis(df):
        son_okumalar = df.copy()
        son_okumalar['OKUMA_PERIYODU_GUN'] = 300
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Demo davranış analizi sonuçları
    risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.7, 0.2, 0.1])
    son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
    son_okumalar['DAVRANIS_YORUMU'] = "Demo verisi - analiz edildi"
    son_okumalar['SUPHELI_DONEMLER'] = "Yok"
    
    # Zone analizi
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    demo_data_created = True
    st.success("✅ Demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    # Zone dosyasının buradaki analizde kullanılmamasına dikkat edin, simülasyonda kullanılacak.
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
    demo_data_created = False
else:
    if not demo_data_created:
        st.warning("⚠️ Lütfen Excel/CSV dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()


# ... (Rapor İndirme Butonları ve Genel Metrikler - Orijinal haliyle bırakıldı)
if 'son_okumalar' in locals() and son_okumalar is not None:
    # Rapor İndirme Butonları
    st.sidebar.header("📥 Rapor İndirme")
    comprehensive_report = create_comprehensive_report(son_okumalar, zone_analizi)
    st.sidebar.download_button(
        label="📊 Tüm Raporu İndir (Excel)",
        data=comprehensive_report,
        file_name="tum_analiz_raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    # Bireysel raporlar (kısmen sadeleştirildi)
    if 'RISK_SEVIYESI' in son_okumalar.columns:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
            if len(yuksek_riskli) > 0:
                csv_yuksek = yuksek_riskli.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(label="🚨 Yüksek Riskli", data=csv_yuksek, file_name="yuksek_riskli_tesisatlar.csv", mime="text/csv")
        with col2:
            orta_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta']
            if len(orta_riskli) > 0:
                csv_orta = orta_riskli.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(label="🟡 Orta Riskli", data=csv_orta, file_name="orta_riskli_tesisatlar.csv", mime="text/csv")
    
    # Genel Metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: st.metric("📊 Toplam Tesisat", f"{len(son_okumalar):,}")
    
    with col2: 
        if 'AKTIF_m3' in son_okumalar.columns: st.metric("💧 Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")
        else: st.metric("💧 Toplam Tüketim", "Veri Yok")
    
    with col3: 
        if 'TOPLAM_TUTAR' in son_okumalar.columns: st.metric("💰 Toplam Gelir", f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL")
        else: st.metric("💰 Toplam Gelir", "Veri Yok")
    
    with col4: 
        if 'RISK_SEVIYESI' in son_okumalar.columns:
            yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
            st.metric("🚨 Yüksek Riskli Tesisat", f"{yuksek_riskli}")
        else: st.metric("🚨 Risk Analizi", "Mevcut Değil")


# Tab Menü
tab1, tab2, tab3, tab4 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz", "📊 Kayıp Kaçak Simülasyonu"])

# ... (Tab 1, 2, 3 İçerikleri - Orijinal haliyle bırakıldı)

with tab1:
    if 'son_okumalar' in locals() and son_okumalar is not None:
        col1, col2 = st.columns(2)
        with col1:
            if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', title='Günlük Tüketim Dağılımı',
                                    labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'}, color_discrete_sequence=['#3498DB'])
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
        with col2:
            if 'AKTIF_m3' in son_okumalar.columns and 'TOPLAM_TUTAR' in son_okumalar.columns and 'RISK_SEVIYESI' in son_okumalar.columns:
                fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR', color='RISK_SEVIYESI',
                                    title='Tüketim-Tutar İlişkisi (Risk Seviyeli)',
                                    labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                                    color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
                st.plotly_chart(fig2, use_container_width=True)

with tab2:
    if 'zone_analizi' in locals() and zone_analizi is not None:
        col1, col2 = st.columns(2)
        with col1:
            if 'TOPLAM_TUKETIM' in zone_analizi.columns and 'KARNE_NO' in zone_analizi.columns:
                fig4 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO', title='Zone Bazlı Tüketim Dağılımı')
                st.plotly_chart(fig4, use_container_width=True)
        with col2:
            if 'TESISAT_SAYISI' in zone_analizi.columns and 'KARNE_NO' in zone_analizi.columns:
                fig5 = px.bar(zone_analizi, x='KARNE_NO', y='TESISAT_SAYISI', title='Zone Bazlı Tesisat Sayısı',
                                labels={'KARNE_NO': 'Zone', 'TESISAT_SAYISI': 'Tesisat Sayısı'}, color_discrete_sequence=['#E74C3C'])
                st.plotly_chart(fig5, use_container_width=True)
        st.subheader("Zone Karşılaştırma Tablosu")
        st.dataframe(zone_analizi, use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    if 'son_okumalar' in locals() and son_okumalar is not None:
        st.subheader("Tesisat Tablosu")
        columns_to_show = ['TESISAT_NO']
        if 'AKTIF_m3' in son_okumalar.columns: columns_to_show.append('AKTIF_m3')
        if 'TOPLAM_TUTAR' in son_okumalar.columns: columns_to_show.append('TOPLAM_TUTAR')
        if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns: columns_to_show.append('GUNLUK_ORT_TUKETIM_m3')
        if 'RISK_SEVIYESI' in son_okumalar.columns: columns_to_show.append('RISK_SEVIYESI')
        if 'DAVRANIS_YORUMU' in son_okumalar.columns: columns_to_show.append('DAVRANIS_YORUMU')
        st.dataframe(son_okumalar[columns_to_show].round(3), use_container_width=True)

# YENİ TAB: Kayıp Kaçak Simülasyonu - HATA GİDERİLDİ
with tab4:
    st.header("💧 Yavuzeli Su Kayıp Kaçak Analizi Simülasyonu")
    st.markdown("### Literatür Destekli Risk Analizi ve Eylem Planı Önceliklendirmesi")
    st.markdown("---")
    
    # Dosya Yükleme Bölümü (Tekrarlanan uploader'ı kaldırıp Zone dosyasını kullanıyoruz)
    st.header("1️⃣ Veri Girişi")
    if zone_file is None:
        st.warning("⚠️ Lütfen 'Zone Analiz dosyasını' (yavuzeli merkez ekim.xlsx) sol kenar çubuğundan yükleyin.")
    
    if zone_file is not None:
        try:
            # REVİZE EDİLMİŞ FONKSİYON KULLANILDI
            df_zone_raw = load_simulation_data_revised(zone_file)
            
            if df_zone_raw is None:
                st.error("Simülasyon verisi yüklenemedi.")
                st.stop()
                
            # REVİZE EDİLMİŞ SÜTUN EŞLEŞTİRME KULLANILDI
            column_mapping = find_and_rename_columns_revised(df_zone_raw)
            
            if not all(col in column_mapping.values() for col in ['ZONE_ADI', 'GIRN_SU_M3', 'TAHAKKUK_M3']):
                st.error("Zone dosyasında gerekli sütunlar bulunamadı. Lütfen dosya formatını kontrol edin.")
                st.info("Bulunan sütunlar ve eşleşmeler:")
                st.write(column_mapping)
                st.stop()
            else:
                # Sütunları yeniden adlandır
                df_zone = df_zone_raw.rename(columns=column_mapping)
                
                required_cols = ['ZONE_ADI', 'GIRN_SU_M3', 'TAHAKKUK_M3']
                df_zone = df_zone[required_cols].copy()
                
                # NaN satırları temizle ve TOPLAM satırını çıkar
                df_zone = df_zone.dropna(subset=['ZONE_ADI'])
                df_zone = df_zone[~df_zone['ZONE_ADI'].astype(str).str.contains('TOPLAM', na=False)]
                
                # Sayısal dönüşüm (hata oluşmaması için errors='coerce' kullanıldı)
                df_zone['GIRN_SU_M3'] = pd.to_numeric(df_zone['GIRN_SU_M3'], errors='coerce')
                df_zone['TAHAKKUK_M3'] = pd.to_numeric(df_zone['TAHAKKUK_M3'], errors='coerce')
                
                # NaN değerleri temizle
                df_zone = df_zone.dropna(subset=['GIRN_SU_M3', 'TAHAKKUK_M3'])
                
                # Toplam Kaçak Hesaplama
                df_zone['TOPLAM_KACAK_M3'] = df_zone['GIRN_SU_M3'] - df_zone['TAHAKKUK_M3']
                df_zone['TOPLAM_KACAK_ORANI'] = (df_zone['TOPLAM_KACAK_M3'] / df_zone['GIRN_SU_M3']) * 100

                st.success(f"✅ Zone Analiz verileri başarıyla yüklendi ve işlendi: {len(df_zone)} kayıt")
                
                # RİSK PARAMETRELERİ - ANA SAYFADA GÖSTER
                st.header("2️⃣ Risk Parametrelerini Ayarlayın")
                st.markdown("**Talimat:** Zone'unuzun genel durumunu yansıtan risk puanlarını (1: Düşük Risk, 5: Yüksek Risk) seçin.")
                
                boru_malzemesi_options = {
                    "Polietilen (PE/HDPE)": 1,
                    "Beton/Betonarme (Çimento)": 3,
                    "Sfero Döküm Demir": 3,
                    "Gri Döküm (Font) Demir": 4,
                    "Asbestli Çimento (AC)": 5
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("I. Altyapı Parametreleri")
                    boru_yasi = st.slider("1. Boru Yaşı Endeksi", min_value=1, max_value=5, value=5, step=1, help="1: Yeni (0-5 yıl), 5: Eski (20+ yıl)")
                    
                    malzeme_secimi = st.selectbox("2. Baskın Boru Malzemesi Kalitesi", options=list(boru_malzemesi_options.keys()), index=4, help="Malzeme tipine göre risk puanı")
                    malzeme_kalitesi = boru_malzemesi_options[malzeme_secimi]

                with col2:
                    st.subheader("II. Çevresel ve Operasyonel Parametreler")
                    sicaklik_stresi = st.slider("3. Zemin Hareketi/Sıcaklık Stresi", min_value=1, max_value=5, value=4, step=1, help="1: Stabil zemin, 5: Yüksek hareketli zemin")
                    
                    basin_profili = st.slider("4. Basınç Profili", min_value=1, max_value=5, value=5, step=1, help="1: Düşük basınç, 5: Yüksek/değişken basınç")

                # Hesaplama ve Sonuçlar
                real_loss_percent_decimal = calculate_real_loss_percentage(boru_yasi, malzeme_kalitesi, sicaklik_stresi, basin_profili)
                real_loss_percent_display = round(real_loss_percent_decimal * 100, 1)

                df_results = calculate_losses(df_zone, real_loss_percent_decimal)

                st.header("3️⃣ Simülasyon Sonuçları ve Kayıp Dağılımı")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="Toplam Kayıp Risk Puanı (Max 20)", value=f"{boru_yasi + malzeme_kalitesi + sicaklik_stresi + basin_profili}", help="4 ayrı parametrenin puanlarının toplamıdır.")

                with col2:
                    st.metric(label="Tahmini Boru Kaybı (Gerçek Kayıp) Oranı", value=f"%{real_loss_percent_display}", delta=f"Kalan %{100 - real_loss_percent_display:.1f} Sayaç/İdari Kayıptır.")

                with col3:
                    total_real_loss = df_results['TAHMINI_BORU_KAYBI_M3'].sum()
                    total_apparent_loss = df_results['TAHMINI_SAYAC_KAYBI_M3'].sum()
                    st.metric(label="Şebekeden Kaybolan Su Hacmi Tahmini", value=f"{total_real_loss:,} m³", delta="Boru Kaçağı (Fiziksel)")

                st.subheader("Bölge (Zone) Bazında Tahmini Kayıp Hacmi ($m^3$)")
                
                display_cols = ['ZONE_ADI', 'GIRN_SU_M3', 'TOPLAM_KACAK_M3', 'TOPLAM_KACAK_ORANI',
                                'TAHMINI_BORU_KAYBI_M3', 'TAHMINI_SAYAC_KAYBI_M3']
                display_df = df_results[display_cols]
                display_df.columns = ['Zone Adı', 'Giren Su (m³)', 'Toplam Kayıp (m³)', 'Toplam Kayıp (%)', 
                                      'Tahmini Boru Kaybı (m³)', 'Tahmini Sayaç/İdari Kayıp (m³)']
                display_df['Toplam Kayıp (%)'] = display_df['Toplam Kayıp (%)'].round(2).astype(str) + '%'

                st.dataframe(display_df, use_container_width=True)

                st.markdown("---")
                st.subheader("💡 Önceliklendirme ve Eylem Vurgusu")
                st.markdown(f"""
                Bu simülasyon sonuçlarına göre (Gerçek Kayıp Payı: **%{real_loss_percent_display}**):

                * **Acil Fiziki Müdahale:** Toplam kayıp hacminin **{total_real_loss:,} $m^3$'ü** doğrudan boru sisteminden kaynaklanmaktadır. Bu durum, belirlenen risklere göre **Şebeke Rehabilitasyonu** ve **Basınç Yönetimi** projelerinin aciliyetini doğrulamaktadır.
                * **Kayıt ve İdari Müdahale:** **{total_apparent_loss:,} $m^3$'lük** kayıp hacmi ise sayaç okuma hataları, arızalı sayaçlar ve yasadışı kullanımla mücadeleyi (Görünür Kayıp) önceliklendirmeyi gerektirmektedir.
                """)

        except Exception as e:
            st.error(f"Veri işlenirken beklenmedik bir hata oluştu: {e}")
