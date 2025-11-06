# ======================================================================
# 🎨 STREAMLIT ARAYÜZ
# ======================================================================

# Başlık
st.title("💧 Su Tüketim ve Kaçak Tahmin Analiz Dashboard")

# Dosya yükleme bölümü
st.sidebar.header("📁 İki Dosya Yükle")
uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin",
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Demo butonu
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
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    st.success("✅ Demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Rapor İndirme Butonları
st.sidebar.header("📥 Rapor İndirme")

if son_okumalar is not None:
    # Kapsamlı rapor
    comprehensive_report = create_comprehensive_report(son_okumalar, zone_analizi)
    st.sidebar.download_button(
        label="📊 Tüm Raporu İndir (Excel)",
        data=comprehensive_report,
        file_name="tum_analiz_raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Bireysel raporlar
    if 'RISK_SEVIYESI' in son_okumalar.columns:
        col1, col2 = st.sidebar.columns(2)
        
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

# Genel Metrikler
if son_okumalar is not None:
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

# Tab Menü - YENİ TAB EKLENDİ
tab1, tab2, tab3, tab4 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz", "📊 Kayıp Kaçak Simülasyonu"])

with tab1:
    if son_okumalar is not None:
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
    if zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'TOPLAM_TUKETIM' in zone_analizi.columns and 'KARNE_NO' in zone_analizi.columns:
                fig4 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                            title='Zone Bazlı Tüketim Dağılımı')
                st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            if 'TESISAT_SAYISI' in zone_analizi.columns and 'KARNE_NO' in zone_analizi.columns:
                fig5 = px.bar(zone_analizi, x='KARNE_NO', y='TESISAT_SAYISI',
                            title='Zone Bazlı Tesisat Sayısı',
                            labels={'KARNE_NO': 'Zone', 'TESISAT_SAYISI': 'Tesisat Sayısı'},
                            color_discrete_sequence=['#E74C3C'])
                st.plotly_chart(fig5, use_container_width=True)
        
        # Zone Karşılaştırma Tablosu
        st.subheader("Zone Karşılaştırma Tablosu")
        st.dataframe(zone_analizi, use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    if son_okumalar is not None:
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

# YENİ TAB: Kayıp Kaçak Simülasyonu
with tab4:
    st.header("💧 Yavuzeli Su Kayıp Kaçak Analizi Simülasyonu")
    st.markdown("### Literatür Destekli Risk Analizi ve Eylem Planı Önceliklendirmesi")
    st.markdown("---")
    
    # Yardımcı Fonksiyonlar
    @st.cache_data
    def load_simulation_data(uploaded_file, header_row=8):
        """Yüklenen dosyayı okur (CSV veya Excel)."""
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=header_row-1)
        else:
            df = pd.read_excel(uploaded_file, header=header_row)
        return df

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

    def calculate_losses(df, real_loss_percentage):
        """Verilen yüzdeye göre kayıp hacimlerini hesaplar."""
        df_calc = df.copy()
        
        df_calc['TAHMINI_GERCEK_KAYIP_YUZDESI'] = real_loss_percentage * 100
        df_calc['TAHMINI_GORUNUR_KAYIP_YUZDESI'] = (1 - real_loss_percentage) * 100
        
        df_calc['TAHMINI_BORU_KAYBI_M3'] = df_calc['TOPLAM_KACAK_M3'] * real_loss_percentage
        df_calc['TAHMINI_SAYAC_KAYBI_M3'] = df_calc['TOPLAM_KACAK_M3'] * (1 - real_loss_percentage)

        cols_to_round = ['GIRN_SU_M3', 'TAHAKKUK_M3', 'TOPLAM_KACAK_M3', 'TAHMINI_BORU_KAYBI_M3', 'TAHMINI_SAYAC_KAYBI_M3']
        for col in cols_to_round:
            df_calc[col] = df_calc[col].round(0).astype(int)

        return df_calc

    # Dosya Yükleme Bölümü
    st.header("1️⃣ Veri Girişi")
    col_files1, col_files2 = st.columns(2)

    with col_files1:
        uploaded_file_zone = st.file_uploader(
            "**'YAVUZELİ MERKEZ EKİM.xlsx'** Dosyasını Yükleyin (Giriş/Tahakkuk Verileri)", 
            type=['csv', 'xlsx'],
            key="simulation_uploader"
        )

    with col_files2:
        st.info("Bu analizde sadece Giriş/Tahakkuk verileri kullanılacaktır. Tesisat detay (yavuz.xlsx) dosyası şu anki analiz için gerekli değildir.")

    if uploaded_file_zone is not None:
        try:
            # Veriyi yükle ve sütun adlarını sabitle
            df_zone_raw = load_simulation_data(uploaded_file_zone, header_row=8)
            
            # Sütunları temizle ve yeniden adlandır
            df_zone_raw.columns = df_zone_raw.columns.str.strip().str.replace('\n', ' ', regex=False)
            
            # Gerekli sütunları seç
            df_zone = df_zone_raw[['KARNE NO VE ADI', 'VERİLEN SU MİKTARI M3', 'TAHAKKUK M3']].copy()
            df_zone.columns = ['ZONE_ADI', 'GIRN_SU_M3', 'TAHAKKUK_M3']
            
            # NaN satırları temizle ve sayısal dönüşüm yap
            df_zone.dropna(subset=['ZONE_ADI'], inplace=True)
            df_zone = df_zone[~df_zone['ZONE_ADI'].str.contains('TOPLAM', na=False)] # TOPLAM satırını çıkar
            
            # Sayısal dönüşüm (virgül yerine nokta kullanılması gerekebilir)
            df_zone['GIRN_SU_M3'] = df_zone['GIRN_SU_M3'].astype(str).str.replace(',', '.', regex=False).astype(float)
            df_zone['TAHAKKUK_M3'] = df_zone['TAHAKKUK_M3'].astype(str).str.replace(',', '.', regex=False).astype(float)
            
            # Toplam Kaçak Hesaplama
            df_zone['TOPLAM_KACAK_M3'] = df_zone['GIRN_SU_M3'] - df_zone['TAHAKKUK_M3']
            df_zone['TOPLAM_KACAK_ORANI'] = (df_zone['TOPLAM_KACAK_M3'] / df_zone['GIRN_SU_M3']) * 100

            st.success("Veriler başarıyla yüklendi ve işlendi. Şimdi risk parametrelerini ayarlayabilirsiniz.")
            
            # Sidebar Risk Parametreleri
            st.sidebar.header("🔧 Kayıp Kaçak Simülasyonu - Risk Parametreleri")
            st.sidebar.markdown("**Talimat:** Zone'unuzun genel durumunu yansıtan risk puanlarını (1: Düşük Risk, 5: Yüksek Risk) seçin.")
            
            boru_malzemesi_options = {
                "Polietilen (PE/HDPE)": 1,
                "Beton/Betonarme (Çimento)": 3,
                "Sfero Döküm Demir": 3,
                "Gri Döküm (Font) Demir": 4,
                "Asbestli Çimento (AC)": 5
            }
            
            st.sidebar.subheader("I. Altyapı Parametreleri")

            boru_yasi = st.sidebar.slider("1. Boru Yaşı Endeksi", min_value=1, max_value=5, value=5, step=1, key="boru_yasi")
            
            malzeme_secimi = st.sidebar.selectbox("2. Baskın Boru Malzemesi Kalitesi", 
                                                options=list(boru_malzemesi_options.keys()), index=4, key="malzeme_secimi")
            malzeme_kalitesi = boru_malzemesi_options[malzeme_secimi]

            st.sidebar.subheader("II. Çevresel ve Operasyonel Parametreler")

            sicaklik_stresi = st.sidebar.slider("3. Zemin Hareketi/Sıcaklık Stresi", min_value=1, max_value=5, value=4, step=1, key="sicaklik_stresi")
            
            basin_profili = st.sidebar.slider("4. Basınç Profili", min_value=1, max_value=5, value=5, step=1, key="basin_profili")

            # Hesaplama ve Sonuçlar
            real_loss_percent_decimal = calculate_real_loss_percentage(boru_yasi, malzeme_kalitesi, sicaklik_stresi, basin_profili)
            real_loss_percent_display = round(real_loss_percent_decimal * 100, 1)

            df_results = calculate_losses(df_zone, real_loss_percent_decimal)

            st.header("3️⃣ Simülasyon Sonuçları ve Kayıp Dağılımı")

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
            st.error(f"Veri işlenirken bir hata oluştu. Lütfen yüklediğiniz dosyanın formatını ve sütun adlarını kontrol edin. Hata: {e}")

    else:
        st.info("Lütfen analize başlamak için 'YAVUZELİ MERKEZ EKİM.xlsx' dosyasını yukarıdaki yükleme kutusuna sürükleyip bırakın.")

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim ve Kaçak Tahmin Analiz Sistemi | Streamlit Dashboard")
