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
from io import BytesIO
warnings.filterwarnings('ignore')

# Makine Öğrenmesi modülleri
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Su Tüketim ve Kaçak Tahmin Analiz Dashboard",
    page_icon="💧",
    layout="wide"
)

# ======================================================================
# 📊 VERİ İŞLEME FONKSİYONLARI
# ======================================================================

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder"""
    try:
        # Ana veri dosyasını oku
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None, None

    # Tarih formatını düzelt
    date_columns = ['ILK_OKUMA_TARIHI', 'OKUMA_TARIHI']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Tesisat numarası olan kayıtları filtrele
    if 'TESISAT_NO' in df.columns:
        df = df[df['TESISAT_NO'].notnull()]
    
    # Zone veri dosyasını oku ve işle
    zone_analizi = None
    kullanici_zone_verileri = {}
    
    if zone_file is not None:
        try:
            zone_excel_df = pd.read_excel(zone_file)
            st.success(f"✅ Zone veri dosyası başarıyla yüklendi: {len(zone_excel_df)} kayıt")
            
            # Sütun isimlerini temizle
            zone_excel_df.columns = zone_excel_df.columns.str.strip().str.replace('\n', ' ', regex=False)
            
            # Zone verilerini işle - daha esnek sütun eşleştirme
            column_mapping = {}
            for col in zone_excel_df.columns:
                col_clean = str(col).upper().replace(' ', '')
                if 'KARNE' in col_clean or 'ZONE' in col_clean:
                    column_mapping[col] = 'ZONE_ADI'
                elif 'VERİLEN' in col_clean or 'GİREN' in col_clean or 'GIREN' in col_clean or 'SUMIKTARI' in col_clean:
                    column_mapping[col] = 'GIRN_SU_M3'
                elif 'TAHAKKUK' in col_clean:
                    column_mapping[col] = 'TAHAKKUK_M3'
            
            # Sütunları yeniden adlandır
            if column_mapping:
                zone_excel_df = zone_excel_df.rename(columns=column_mapping)
            
            # Gerekli sütunları kontrol et
            required_cols = ['ZONE_ADI', 'GIRN_SU_M3', 'TAHAKKUK_M3']
            available_cols = [col for col in required_cols if col in zone_excel_df.columns]
            
            if len(available_cols) >= 2:  # En az Zone adı ve bir sayısal sütun
                zone_analizi = zone_excel_df[available_cols].copy()
                
                # NaN satırları temizle
                zone_analizi = zone_analizi.dropna(subset=['ZONE_ADI'])
                zone_analizi = zone_analizi[~zone_analizi['ZONE_ADI'].astype(str).str.contains('TOPLAM', na=False)]
                
                # Sayısal dönüşüm
                for col in ['GIRN_SU_M3', 'TAHAKKUK_M3']:
                    if col in zone_analizi.columns:
                        zone_analizi[col] = pd.to_numeric(zone_analizi[col], errors='coerce')
                
                # NaN değerleri temizle
                zone_analizi = zone_analizi.dropna(subset=[col for col in ['GIRN_SU_M3', 'TAHAKKUK_M3'] if col in zone_analizi.columns])
                
                st.success(f"✅ Zone verileri işlendi: {len(zone_analizi)} kayıt")
                
        except Exception as e:
            st.error(f"❌ Zone veri dosyası işlenirken hata: {e}")

    # Davranış analizi fonksiyonları
    def perform_behavior_analysis(df):
        if 'OKUMA_TARIHI' not in df.columns or 'TESISAT_NO' not in df.columns:
            return df
            
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        
        if 'ILK_OKUMA_TARIHI' in df.columns and 'OKUMA_TARIHI' in df.columns:
            son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
            son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        
        if 'AKTIF_m3' in son_okumalar.columns and 'OKUMA_PERIYODU_GUN' in son_okumalar.columns:
            son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
            son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Davranış analizi fonksiyonu
    def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
        if 'TESISAT_NO' not in df.columns or 'AKTIF_m3' not in df.columns:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta"

        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI') if 'OKUMA_TARIHI' in df.columns else df[df['TESISAT_NO'] == tesisat_no]

        if len(tesisat_verisi) < 3:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta"

        tuketimler = tesisat_verisi['AKTIF_m3'].values

        # Sıfır tüketim analizi
        sifir_sayisi = sum(tuketimler == 0)

        # Varyasyon analizi
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0

        # Şüpheli durum tespiti
        suphe_aciklamasi = ""
        suphe_donemleri = []
        risk_seviyesi = "Düşük"

        # 1. Düzensiz sıfır tüketim paterni
        if sifir_sayisi >= 3:
            sifir_indisler = np.where(tuketimler == 0)[0]
            if len(sifir_indisler) >= 3:
                ardisik_olmayan = sum(np.diff(sifir_indisler) > 1) >= 2
                if ardisik_olmayan:
                    suphe_aciklamasi += "Düzensiz sıfır tüketim paterni. "
                    risk_seviyesi = "Yüksek"

        # 2. Ani tüketim değişiklikleri
        if varyasyon_katsayisi > 1.5 and mean_tuketim > 5:
            suphe_aciklamasi += "Tüketimde yüksek dalgalanma. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi

        # 3. Son dönem sıfır tüketim
        if tuketimler[-1] == 0 and len(tuketimler) > 1:
            suphe_aciklamasi += "Son dönem sıfır tüketim. "
            risk_seviyesi = "Yüksek" if sifir_sayisi >= 2 else "Orta"

        # Yorum kütüphanesi
        yorumlar_normal = ["Normal tüketim paterni", "Stabil tüketim alışkanlığı"]
        yorumlar_supheli = [
            "Tüketim alışkanlıklarında değişiklik gözlemleniyor",
            "Düzensiz tüketim paterni dikkat çekici",
            "Tüketim davranışında tutarsızlık mevcut"
        ]

        if not suphe_aciklamasi:
            davranis_yorumu = np.random.choice(yorumlar_normal)
        else:
            davranis_yorumu = np.random.choice(yorumlar_supheli)

        return davranis_yorumu, ", ".join(suphe_donemleri) if suphe_donemleri else "Yok", risk_seviyesi

    # Tüm tesisatlar için davranış analizi yap
    davranis_sonuclari = []
    if 'TESISAT_NO' in son_okumalar.columns:
        for i, (idx, row) in enumerate(son_okumalar.iterrows()):
            yorum, supheli_donemler, risk = tesisat_davranis_analizi(row['TESISAT_NO'], row, df)
            davranis_sonuclari.append({
                'TESISAT_NO': row['TESISAT_NO'],
                'DAVRANIS_YORUMU': yorum,
                'SUPHELI_DONEMLER': supheli_donemler,
                'RISK_SEVIYESI': risk
            })

        davranis_df = pd.DataFrame(davranis_sonuclari)
        son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    return df, son_okumalar, zone_analizi, kullanici_zone_verileri

# ======================================================================
# 🤖 MAKİNE ÖĞRENMESİ KAÇAK TAHMİN MODÜLÜ
# ======================================================================

class LeakagePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        self.feature_importance = {}
    
    def train_model(self, df, feature_columns, target_column='KAÇAK_RİSK_SKORU'):
        """Model eğit"""
        try:
            # Eksik verileri temizle
            df_clean = df.dropna(subset=feature_columns + [target_column])
            
            if len(df_clean) < 10:
                st.error("Eğitim için yeterli veri yok")
                return None
            
            # Özellikler ve hedef
            X = df_clean[feature_columns]
            y = df_clean[target_column]
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Veriyi bölme
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Model eğitme
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Tahminler
            y_pred = model.predict(X_test)
            
            # Anomali tespiti
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = anomaly_detector.fit_predict(X_scaled)
            
            # Feature importance
            feature_imp = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Sonuçları sakla
            self.scalers['main'] = scaler
            self.models['main'] = model
            self.anomaly_detectors['main'] = anomaly_detector
            self.feature_importance['main'] = feature_imp
            
            return {
                'model': model,
                'scaler': scaler,
                'anomaly_detector': anomaly_detector,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'anomalies': anomalies,
                'feature_importance': feature_imp
            }
            
        except Exception as e:
            st.error(f"Model eğitilirken hata: {str(e)}")
            return None

# ======================================================================
# 📥 RAPOR İNDİRME FONKSİYONLARI
# ======================================================================

def create_comprehensive_report(son_okumalar, zone_analizi, ml_results=None):
    """Kapsamlı Excel raporu oluştur"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Tüm Tesisatlar
        if son_okumalar is not None:
            son_okumalar.to_excel(writer, sheet_name='Tüm_Tesisatlar', index=False)
            
            # 2. Yüksek Riskli Tesisatlar
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
                yuksek_riskli.to_excel(writer, sheet_name='Yüksek_Riskli_Tesisatlar', index=False)
                
                # 3. Orta Riskli Tesisatlar
                orta_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta']
                orta_riskli.to_excel(writer, sheet_name='Orta_Riskli_Tesisatlar', index=False)
        
        # 4. Zone Analizi
        if zone_analizi is not None:
            zone_analizi.to_excel(writer, sheet_name='Zone_Analizi', index=False)
        
        # 5. Makine Öğrenmesi Sonuçları
        if ml_results is not None:
            ml_results['feature_importance'].to_excel(writer, sheet_name='ML_Özellik_Önemliliği', index=False)
    
    return output.getvalue()

# ======================================================================
# 🎨 KAYIP KAÇAK SİMÜLASYONU FONKSİYONLARI
# ======================================================================

def calculate_real_loss_percentage(boru_yasi, malzeme_kalitesi, sicaklik_stresi, basin_profili):
    """Kullanıcının slider girdilerine göre Gerçek Kayıp Yüzdesini hesaplar (55% - 75% aralığında)."""
    total_risk_score = boru_yasi + malzeme_kalitesi + sicaklik_stresi + basin_profili
    
    # Riski 4-20 aralığından 0-1 aralığına normalize etme:
    normalized_risk = (total_risk_score - 4) / (20 - 4)
    
    # Yüzdeyi 55% (min) ile 75% (max) arasına ölçekleme:
    min_loss_percentage = 0.55
    max_loss_percentage = 0.75
    
    real_loss_percentage = min_loss_percentage + (max_loss_percentage - min_loss_percentage) * normalized_risk
    
    return real_loss_percentage

def calculate_losses(df_zone, real_loss_percentage):
    """Verilen yüzdeye göre kayıp hacimlerini hesaplar."""
    df_calc = df_zone.copy()
    
    # Toplam Kaçak Hesaplama (eğer yoksa)
    if 'TOPLAM_KACAK_M3' not in df_calc.columns:
        if 'GIRN_SU_M3' in df_calc.columns and 'TAHAKKUK_M3' in df_calc.columns:
            df_calc['TOPLAM_KACAK_M3'] = df_calc['GIRN_SU_M3'] - df_calc['TAHAKKUK_M3']
            df_calc['TOPLAM_KACAK_ORANI'] = (df_calc['TOPLAM_KACAK_M3'] / df_calc['GIRN_SU_M3']) * 100
    
    # Gerçek ve Görünür Kayıp Hesaplamaları
    df_calc['TAHMINI_GERCEK_KAYIP_YUZDESI'] = real_loss_percentage * 100
    df_calc['TAHMINI_GORUNUR_KAYIP_YUZDESI'] = (1 - real_loss_percentage) * 100
    
    df_calc['TAHMINI_BORU_KAYBI_M3'] = df_calc['TOPLAM_KACAK_M3'] * real_loss_percentage
    df_calc['TAHMINI_SAYAC_KAYBI_M3'] = df_calc['TOPLAM_KACAK_M3'] * (1 - real_loss_percentage)

    # Yuvarlama
    cols_to_round = ['GIRN_SU_M3', 'TAHAKKUK_M3', 'TOPLAM_KACAK_M3', 'TAHMINI_BORU_KAYBI_M3', 'TAHMINI_SAYAC_KAYBI_M3']
    for col in cols_to_round:
        if col in df_calc.columns:
            df_calc[col] = df_calc[col].round(0).astype(int)

    return df_calc

# ======================================================================
# 🎨 STREAMLIT ARAYÜZ
# ======================================================================

# Başlık
st.title("💧 Su Tüketim ve Kaçak Tahmin Analiz Dashboard")

# Dosya yükleme bölümü
st.sidebar.header("📁 İki Dosya Yükle")
uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin (yavuz.xlsx)",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin (yavuzeli merkez ekim.xlsx)",
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Demo butonu
demo_data_created = False
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    # Demo verisi oluştur
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
    # Örnek veri oluştur
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
    zone_analizi = pd.DataFrame({
        'ZONE_ADI': ['ZONE1', 'ZONE2', 'ZONE3', 'ZONE4', 'ZONE5'],
        'GIRN_SU_M3': [18666, 19623, 20078, 3968, 17775],
        'TAHAKKUK_M3': [7654, 7375, 7010, 1813, 2134]
    })
    
    demo_data_created = True
    st.success("✅ Demo verisi başarıyla oluşturuldu!")
    st.session_state.demo_data = True
    st.session_state.df_zone = zone_analizi

elif uploaded_file is not None and zone_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
    demo_data_created = False
    
    if zone_analizi is not None:
        st.session_state.df_zone = zone_analizi
        st.session_state.demo_data = False
else:
    if not demo_data_created and 'demo_data' not in st.session_state:
        st.warning("⚠️ Lütfen iki Excel dosyasını da yükleyin veya Demo modunu kullanın")
    st.stop()

# Rapor İndirme Butonları
st.sidebar.header("📥 Rapor İndirme")

if 'son_okumalar' in locals() and son_okumalar is not None:
    # Kapsamlı rapor
    comprehensive_report = create_comprehensive_report(son_okumalar, zone_analizi)
    st.sidebar.download_button(
        label="📊 Tüm Raporu İndir (Excel)",
        data=comprehensive_report,
        file_name="tum_analiz_raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Genel Metrikler
if 'son_okumalar' in locals() and son_okumalar is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Toplam Tesisat", f"{len(son_okumalar):,}")
    
    with col2:
        if 'AKTIF_m3' in son_okumalar.columns:
            st.metric("💧 Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")
        else:
            st.metric("💧 Toplam Tüketim", "Veri Yok")
    
    with col3:
        if 'TOPLAM_TUTAR' in son_okumalar.columns:
            st.metric("💰 Toplam Gelir", f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL")
        else:
            st.metric("💰 Toplam Gelir", "Veri Yok")
    
    with col4:
        if 'RISK_SEVIYESI' in son_okumalar.columns:
            yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
            st.metric("🚨 Yüksek Riskli Tesisat", f"{yuksek_riskli}")
        else:
            st.metric("🚨 Risk Analizi", "Mevcut Değil")

# Tab Menü
tab1, tab2, tab3, tab4 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz", "📊 Kayıp Kaçak Simülasyonu"])

with tab1:
    if 'son_okumalar' in locals() and son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns:
                fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                                  title='Günlük Tüketim Dağılımı',
                                  labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'},
                                  color_discrete_sequence=['#3498DB'])
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if 'AKTIF_m3' in son_okumalar.columns and 'TOPLAM_TUTAR' in son_okumalar.columns and 'RISK_SEVIYESI' in son_okumalar.columns:
                fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                                color='RISK_SEVIYESI',
                                title='Tüketim-Tutar İlişkisi (Risk Seviyeli)',
                                labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                                color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
                st.plotly_chart(fig2, use_container_width=True)

with tab2:
    if 'zone_analizi' in locals() and zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'GIRN_SU_M3' in zone_analizi.columns and 'ZONE_ADI' in zone_analizi.columns:
                fig4 = px.pie(zone_analizi, values='GIRN_SU_M3', names='ZONE_ADI',
                            title='Zone Bazlı Su Giriş Dağılımı')
                st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            if 'TAHAKKUK_M3' in zone_analizi.columns and 'ZONE_ADI' in zone_analizi.columns:
                fig5 = px.bar(zone_analizi, x='ZONE_ADI', y='TAHAKKUK_M3',
                            title='Zone Bazlı Tahakkuk Miktarı',
                            labels={'ZONE_ADI': 'Zone', 'TAHAKKUK_M3': 'Tahakkuk (m³)'},
                            color_discrete_sequence=['#E74C3C'])
                st.plotly_chart(fig5, use_container_width=True)
        
        # Zone Karşılaştırma Tablosu
        st.subheader("Zone Karşılaştırma Tablosu")
        display_zone = zone_analizi.copy()
        if 'GIRN_SU_M3' in display_zone.columns and 'TAHAKKUK_M3' in display_zone.columns:
            display_zone['TOPLAM_KACAK_M3'] = display_zone['GIRN_SU_M3'] - display_zone['TAHAKKUK_M3']
            display_zone['KAYIP_ORANI'] = (display_zone['TOPLAM_KACAK_M3'] / display_zone['GIRN_SU_M3']) * 100
            display_zone['KAYIP_ORANI'] = display_zone['KAYIP_ORANI'].round(2)
        
        st.dataframe(display_zone, use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    if 'son_okumalar' in locals() and son_okumalar is not None:
        st.subheader("Tesisat Tablosu")
        
        # Filtreleme
        columns_to_show = ['TESISAT_NO']
        if 'AKTIF_m3' in son_okumalar.columns:
            columns_to_show.append('AKTIF_m3')
        if 'TOPLAM_TUTAR' in son_okumalar.columns:
            columns_to_show.append('TOPLAM_TUTAR')
        if 'GUNLUK_ORT_TUKETIM_m3' in son_okumalar.columns:
            columns_to_show.append('GUNLUK_ORT_TUKETIM_m3')
        if 'RISK_SEVIYESI' in son_okumalar.columns:
            columns_to_show.append('RISK_SEVIYESI')
        if 'DAVRANIS_YORUMU' in son_okumalar.columns:
            columns_to_show.append('DAVRANIS_YORUMU')
        
        st.dataframe(
            son_okumalar[columns_to_show].round(3),
            use_container_width=True
        )

# YENİ TAB: Kayıp Kaçak Simülasyonu - MEVCUT VERİLERLE
with tab4:
    st.header("💧 Yavuzeli Su Kayıp Kaçak Analizi Simülasyonu")
    st.markdown("### Literatür Destekli Risk Analizi ve Eylem Planı Önceliklendirmesi")
    st.markdown("---")
    
    # Mevcut verileri kullan
    if 'df_zone' in st.session_state:
        df_zone = st.session_state.df_zone
        
        st.success(f"✅ Zone verileri simülasyon için hazır: {len(df_zone)} kayıt")
        
        # RİSK PARAMETRELERİ
        st.header("1️⃣ Risk Parametrelerini Ayarlayın")
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
            boru_yasi = st.slider("1. Boru Yaşı Endeksi", min_value=1, max_value=5, value=5, step=1, 
                                help="1: Yeni (0-5 yıl), 5: Eski (20+ yıl)", key="boru_yasi_sim")
            
            malzeme_secimi = st.selectbox("2. Baskın Boru Malzemesi Kalitesi", 
                                        options=list(boru_malzemesi_options.keys()), index=4,
                                        help="Malzeme tipine göre risk puanı", key="malzeme_sim")
            malzeme_kalitesi = boru_malzemesi_options[malzeme_secimi]

        with col2:
            st.subheader("II. Çevresel ve Operasyonel Parametreler")
            sicaklik_stresi = st.slider("3. Zemin Hareketi/Sıcaklık Stresi", min_value=1, max_value=5, value=4, step=1,
                                      help="1: Stabil zemin, 5: Yüksek hareketli zemin", key="sicaklik_sim")
            
            basin_profili = st.slider("4. Basınç Profili", min_value=1, max_value=5, value=5, step=1,
                                    help="1: Düşük basınç, 5: Yüksek/değişken basınç", key="basin_sim")

        # Hesaplama ve Sonuçlar
        real_loss_percent_decimal = calculate_real_loss_percentage(boru_yasi, malzeme_kalitesi, sicaklik_stresi, basin_profili)
        real_loss_percent_display = round(real_loss_percent_decimal * 100, 1)

        df_results = calculate_losses(df_zone, real_loss_percent_decimal)

        st.header("2️⃣ Simülasyon Sonuçları ve Kayıp Dağılımı")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Toplam Kayıp Risk Puanı (Max 20)",
                value=f"{boru_yasi + malzeme_kalitesi + sicaklik_stresi + basin_profili}",
                help="4 ayrı parametrenin puanlarının toplamıdır."
            )

        with col2:
            st.metric(
                label="Tahmini Boru Kaybı (Gerçek Kayıp) Oranı",
                value=f"%{real_loss_percent_display}",
                delta=f"Kalan %{100 - real_loss_percent_display:.1f} Sayaç/İdari Kayıptır."
            )

        with col3:
            total_real_loss = df_results['TAHMINI_BORU_KAYBI_M3'].sum()
            total_apparent_loss = df_results['TAHMINI_SAYAC_KAYBI_M3'].sum()
            st.metric(
                label="Şebekeden Kaybolan Su Hacmi Tahmini",
                value=f"{total_real_loss:,} m³",
                delta="Boru Kaçağı (Fiziksel)"
            )

        st.subheader("Bölge (Zone) Bazında Tahmini Kayıp Hacmi ($m^3$)")
        
        display_cols = ['ZONE_ADI', 'GIRN_SU_M3', 'TOPLAM_KACAK_M3', 'TOPLAM_KACAK_ORANI',
                        'TAHMINI_BORU_KAYBI_M3', 'TAHMINI_SAYAC_KAYBI_M3']
        display_df = df_results[display_cols].copy()
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
        
    else:
        st.warning("⚠️ Simülasyon için Zone verisi bulunamadı.")
        st.info("Lütfen başlangıçta 'Zone Excel dosyasını' yükleyin (yavuzeli merkez ekim.xlsx)")

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim ve Kaçak Tahmin Analiz Sistemi | Streamlit Dashboard")
