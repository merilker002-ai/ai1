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
import warnings
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
# 📊 VERİ İŞLEME FONKSİYONLARI (İKİ DOSYA OKUYAN)
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
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    
    # Tesisat numarası olan kayıtları filtrele
    df = df[df['TESISAT_NO'].notnull()]
    
    # Zone veri dosyasını oku
    kullanici_zone_verileri = {}
    if zone_file is not None:
        try:
            zone_excel_df = pd.read_excel(zone_file)
            st.success(f"✅ Zone veri dosyası başarıyla yüklendi: {len(zone_excel_df)} kayıt")
            
            # Zone verilerini işle
            for idx, row in zone_excel_df.iterrows():
                # Karne no ve adını ayır
                if 'KARNE NO VE ADI' in row:
                    karne_adi = str(row['KARNE NO VE ADI']).strip()
                    
                    # Karne numarasını çıkar (ilk 4 rakam)
                    karne_no_match = re.search(r'(\d{4})', karne_adi)
                    if karne_no_match:
                        karne_no = karne_no_match.group(1)
                        
                        # Zone bilgilerini topla
                        zone_bilgisi = {
                            'ad': karne_adi,
                            'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                            'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                            'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                        }
                        
                        kullanici_zone_verileri[karne_no] = zone_bilgisi
        except Exception as e:
            st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

    # Davranış analizi fonksiyonları
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Davranış analizi fonksiyonu (şüpheli tesisat tespiti)
    def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')

        if len(tesisat_verisi) < 3:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta"

        tuketimler = tesisat_verisi['AKTIF_m3'].values
        tarihler_series = tesisat_verisi['OKUMA_TARIHI']

        # Sıfır tüketim analizi
        sifir_sayisi = sum(tuketimler == 0)

        # Varyasyon analizi
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0

        # Trend analizi (son 3 dönem)
        if len(tuketimler) >= 3:
            son_uc = tuketimler[-3:]
            trend = "artış" if son_uc[2] > son_uc[0] * 1.2 else "azalış" if son_uc[2] < son_uc[0] * 0.8 else "stabil"
        else:
            trend = "belirsiz"

        # Şüpheli durum tespiti ve risk seviyesi
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
                    for idx in sifir_indisler:
                        tarih_obj = pd.Timestamp(tarihler_series.iloc[idx])
                        suphe_donemleri.append(tarih_obj.strftime('%m/%Y'))

        # 2. Ani tüketim değişiklikleri
        if varyasyon_katsayisi > 1.5 and mean_tuketim > 5:
            suphe_aciklamasi += "Tüketimde yüksek dalgalanma. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi

        # 3. Trend analizi
        if trend == "artış" and mean_tuketim > 20:
            suphe_aciklamasi += "Yükselen tüketim trendi. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi

        # 4. Son dönem sıfır tüketim
        if tuketimler[-1] == 0 and len(tuketimler) > 1:
            suphe_aciklamasi += "Son dönem sıfır tüketim. "
            risk_seviyesi = "Yüksek" if sifir_sayisi >= 2 else "Orta"

        # Şüpheli dönemler varsa risk en az Orta olmalı
        if suphe_donemleri and risk_seviyesi == "Düşük":
            risk_seviyesi = "Orta"

        # Yorum kütüphanesi
        yorumlar_normal = ["Normal tüketim paterni", "Stabil tüketim alışkanlığı"]
        yorumlar_supheli = [
            "Tüketim alışkanlıklarında değişiklik gözlemleniyor",
            "Düzensiz tüketim paterni dikkat çekici",
            "Tüketim davranışında tutarsızlık mevcut",
            "Değişken tüketim alışkanlıkları",
            "Tüketim paterninde olağandışı dalgalanma"
        ]

        if not suphe_aciklamasi:
            davranis_yorumu = np.random.choice(yorumlar_normal)
        else:
            davranis_yorumu = np.random.choice(yorumlar_supheli)

        return davranis_yorumu, ", ".join(suphe_donemleri) if suphe_donemleri else "Yok", risk_seviyesi

    # Tüm tesisatlar için davranış analizi yap
    davranis_sonuclari = []
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

    # Zone analizi
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        ekim_2024_df = df[(df['OKUMA_TARIHI'].dt.month == 10) & (df['OKUMA_TARIHI'].dt.year == 2024)]
        if len(ekim_2024_df) == 0:
            ekim_2024_df = df.copy()
        
        zone_analizi = ekim_2024_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        # Zone risk analizi
        ekim_2024_risk = ekim_2024_df.merge(son_okumalar[['TESISAT_NO', 'RISK_SEVIYESI']], on='TESISAT_NO', how='left')
        zone_risk_analizi = ekim_2024_risk.groupby('KARNE_NO')['RISK_SEVIYESI'].apply(
            lambda x: (x == 'Yüksek').sum() if 'Yüksek' in x.values else 0
        ).reset_index(name='YUKSEK_RISKLI_TESISAT')

        zone_analizi = zone_analizi.merge(zone_risk_analizi, on='KARNE_NO', how='left')
        zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100

        # Kullanıcı zone verilerini birleştir
        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

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
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
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

def create_sample_features():
    """Örnek özellikler oluştur"""
    np.random.seed(42)
    
    sample_data = []
    for i in range(100):
        sample_data.append({
            'BASINÇ': np.random.normal(50, 10),
            'DEBİ': np.random.normal(100, 20),
            'SICAKLIK': np.random.normal(20, 5),
            'TİTREŞİM': np.random.normal(5, 2),
            'BASINÇ_DEĞİŞİM': np.random.exponential(1),
            'DEBİ_VARYASYON': np.random.exponential(2),
            'KAÇAK_RİSK_SKORU': np.random.uniform(0, 1)
        })
    
    df = pd.DataFrame(sample_data)
    feature_columns = ['BASINÇ', 'DEBİ', 'SICAKLIK', 'TİTREŞİM']
    
    st.info("Örnek veri kullanılıyor. Gerçek veriler yüklendiğinde otomatik olarak güncellenecektir.")
    return df, feature_columns

# ======================================================================
# 📥 RAPOR İNDİRME FONKSİYONLARI
# ======================================================================

def create_comprehensive_report(son_okumalar, zone_analizi, ml_results=None):
    """Kapsamlı Excel raporu oluştur"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Tüm Tesisatlar
        son_okumalar.to_excel(writer, sheet_name='Tüm_Tesisatlar', index=False)
        
        # 2. Yüksek Riskli Tesisatlar
        yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
        yuksek_riskli.to_excel(writer, sheet_name='Yüksek_Riskli_Tesisatlar', index=False)
        
        # 3. Orta Riskli Tesisatlar
        orta_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta']
        orta_riskli.to_excel(writer, sheet_name='Orta_Riskli_Tesisatlar', index=False)
        
        # 4. Zone Analizi
        if zone_analizi is not None:
            zone_analizi.to_excel(writer, sheet_name='Zone_Analizi', index=False)
        
        # 5. Risk Özeti
        risk_ozeti = pd.DataFrame({
            'Risk_Seviyesi': ['Yüksek', 'Orta', 'Düşük'],
            'Tesisat_Sayısı': [
                len(yuksek_riskli),
                len(orta_riskli),
                len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Düşük'])
            ],
            'Toplam_Tüketim_m3': [
                yuksek_riskli['AKTIF_m3'].sum(),
                orta_riskli['AKTIF_m3'].sum(),
                son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Düşük']['AKTIF_m3'].sum()
            ]
        })
        risk_ozeti.to_excel(writer, sheet_name='Risk_Özeti', index=False)
        
        # 6. Makine Öğrenmesi Sonuçları
        if ml_results is not None:
            ml_results['feature_importance'].to_excel(writer, sheet_name='ML_Özellik_Önemliliği', index=False)
            
            ml_performance = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'R2_Score'],
                'Value': [
                    mean_absolute_error(ml_results['y_test'], ml_results['y_pred']),
                    mean_squared_error(ml_results['y_test'], ml_results['y_pred']),
                    r2_score(ml_results['y_test'], ml_results['y_pred'])
                ]
            })
            ml_performance.to_excel(writer, sheet_name='ML_Performans', index=False)
    
    return output.getvalue()

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

# Makine Öğrenmesi Ayarları
st.sidebar.header("🤖 ML Ayarları")
ml_analysis = st.sidebar.checkbox("Makine Öğrenmesi Analizi Yap", value=True)

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    # Demo verisi oluştur
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
    # Örnek veri oluştur
    demo_data = []
    for i in range(1000):
        tesisat_no = f"TS{1000 + i}"
        aktif_m3 = np.random.gamma(2, 10)
        toplam_tutar = aktif_m3 * 15 + np.random.normal(0, 10)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0.1),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
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
    
    # Örnek zone verileri
    kullanici_zone_verileri = {
        'ZONE1': {'ad': 'ÖLÇÜM NOKTASI-1 (KIRMIZI)', 'verilen_su': 20078.00, 'tahakkuk_m3': 7010.00, 'kayip_oran': 65.09},
        'ZONE2': {'ad': 'ÖLÇÜM NOKTASI-2 (MAVİ)', 'verilen_su': 3968.00, 'tahakkuk_m3': 1813.00, 'kayip_oran': 54.31},
        'ZONE3': {'ad': 'ÖLÇÜM NOKTASI-3 (ALT BÖLGE) (YEŞİL)', 'verilen_su': 19623.00, 'tahakkuk_m3': 7375.00, 'kayip_oran': 62.42},
        'ZONE4': {'ad': 'ÖLÇÜM NOKTASI-5 (ÜST BÖLGE) (MOR)', 'verilen_su': 18666.00, 'tahakkuk_m3': 7654.00, 'kayip_oran': 58.99},
        'ZONE5': {'ad': 'HASTANE BÖLGESİ (SARI)', 'verilen_su': 17775.00, 'tahakkuk_m3': 2134.00, 'kayip_oran': 87.99}
    }
    
    st.success("✅ Demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Makine Öğrenmesi Analizi
ml_results = None
if ml_analysis and son_okumalar is not None:
    st.sidebar.subheader("ML Parametreleri")
    n_estimators = st.sidebar.slider("Ağaç Sayısı", 50, 500, 100)
    
    # Özellik mühendisliği
    try:
        # Sayısal sütunları seç
        numeric_columns = son_okumalar.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 3:
            feature_columns = numeric_columns[:3]
            
            # Yeni özellikler oluştur
            son_okumalar['TUKETIM_VARYASYON'] = son_okumalar[feature_columns[0]].rolling(window=3, min_periods=1).std()
            son_okumalar['TUTAR_TUKETIM_ORAN'] = son_okumalar[feature_columns[1]] / (son_okumalar[feature_columns[0]] + 1)
            
            # Kaçak risk skoru
            risk_factors = []
            for col in feature_columns[:2]:
                normalized = (son_okumalar[col] - son_okumalar[col].mean()) / son_okumalar[col].std()
                risk_factors.append(normalized.abs())
            
            son_okumalar['KAÇAK_RİSK_SKORU'] = sum(risk_factors) / len(risk_factors)
            
            # Model eğit
            predictor = LeakagePredictor()
            with st.spinner("Makine öğrenmesi modeli eğitiliyor..."):
                ml_results = predictor.train_model(son_okumalar, feature_columns)
                
            if ml_results:
                st.sidebar.success("✅ ML modeli eğitildi!")
        else:
            st.sidebar.warning("Yeterli sayısal sütun bulunamadı")
    except Exception as e:
        st.sidebar.error(f"ML analizi hatası: {e}")

# Rapor İndirme Butonları
st.sidebar.header("📥 Rapor İndirme")

if son_okumalar is not None:
    # Kapsamlı rapor
    comprehensive_report = create_comprehensive_report(son_okumalar, zone_analizi, ml_results)
    st.sidebar.download_button(
        label="📊 Tüm Raporu İndir (Excel)",
        data=comprehensive_report,
        file_name="tum_analiz_raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Bireysel raporlar
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        # Yüksek riskli tesisatlar
        yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
        if len(yuksek_riskli) > 0:
            csv_yuksek = yuksek_riskli.to_csv(index=False)
            st.download_button(
                label="🚨 Yüksek Riskli",
                data=csv_yuksek,
                file_name="yuksek_riskli_tesisatlar.csv",
                mime="text/csv"
            )
    
    with col2:
        # Orta riskli tesisatlar
        orta_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta']
        if len(orta_riskli) > 0:
            csv_orta = orta_riskli.to_csv(index=False)
            st.download_button(
                label="🟡 Orta Riskli",
                data=csv_orta,
                file_name="orta_riskli_tesisatlar.csv",
                mime="text/csv"
            )
    
    with col3:
        # Tüm tesisatlar
        csv_tum = son_okumalar.to_csv(index=False)
        st.download_button(
            label="📋 Tüm Tesisatlar",
            data=csv_tum,
            file_name="tum_tesisatlar.csv",
            mime="text/csv"
        )

# Genel Metrikler
if son_okumalar is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Toplam Tesisat",
            value=f"{len(son_okumalar):,}"
        )
    
    with col2:
        st.metric(
            label="💧 Toplam Tüketim",
            value=f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³"
        )
    
    with col3:
        st.metric(
            label="💰 Toplam Gelir",
            value=f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL"
        )
    
    with col4:
        # Risk dağılımı
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric(
            label="🚨 Yüksek Riskli Tesisat",
            value=f"{yuksek_riskli}"
        )

# Tab Menü - GÜNCELLENMİŞ
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Genel Görünüm", 
    "🗺️ Zone Analizi", 
    "🔍 Detaylı Analiz", 
    "📊 İleri Analiz",
    "🔥 Ateş Böceği Görünümü",
    "🤖 ML Kaçak Tahmini"
])

with tab1:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tüketim Dağılım Grafiği
            fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                              title='Günlük Tüketim Dağılımı',
                              labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'},
                              color_discrete_sequence=['#3498DB'])
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Tüketim-Tutar İlişkisi (Risk Renkli)
            fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            color='RISK_SEVIYESI',
                            title='Tüketim-Tutar İlişkisi (Risk Seviyeli)',
                            labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                            color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig2, use_container_width=True)

with tab6:
    st.header("🤖 Makine Öğrenmesi ile Kaçak Tahmini")
    
    if ml_results:
        col1, col2 = st.columns(2)
        
        with col1:
            # Model performansı
            mae = mean_absolute_error(ml_results['y_test'], ml_results['y_pred'])
            r2 = r2_score(ml_results['y_test'], ml_results['y_pred'])
            
            st.metric("Ortalama Mutlak Hata (MAE)", f"{mae:.4f}")
            st.metric("R² Skoru", f"{r2:.4f}")
            
            # Özellik önemliliği
            st.subheader("🎯 Özellik Önemliliği")
            fig_importance = px.bar(ml_results['feature_importance'], 
                                  x='importance', y='feature',
                                  title='Özellik Önemliliği')
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Tahmin vs Gerçek
            fig_pred = px.scatter(x=ml_results['y_test'], y=ml_results['y_pred'],
                                labels={'x': 'Gerçek Değerler', 'y': 'Tahmin Edilen Değerler'},
                                title='Gerçek vs Tahmin Değerleri')
            fig_pred.add_shape(type="line", line=dict(dash="dash"),
                             x0=min(ml_results['y_test']), y0=min(ml_results['y_test']),
                             x1=max(ml_results['y_test']), y1=max(ml_results['y_test']))
            st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Makine öğrenmesi analizi yapılmadı veya yeterli veri bulunamadı")

# Diğer tab'lar orijinal kodla aynı şekilde devam eder...
# [Orijinal tab2, tab3, tab4, tab5 kodları buraya gelecek]

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim ve Kaçak Tahmin Analiz Sistemi | Streamlit Dashboard | 🔥 Ateş Böceği Görünümü")
