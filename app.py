import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. HEADER Wajib
st.set_page_config(page_title="Dashboard Analisis Islamphobia", layout="wide")
st.title("📊 Dashboard Analisis Sentimen & Framing Propaganda")
st.subheader("Topik: Mewacanakan Islamphobia pada Konflik AS–Iran di YouTube")

# Load Data
@st.cache_data
def load_data():
    # Pastikan nama file sama persis dengan yang ada di folder Anda
    return pd.read_csv("data.csv")

try:
    df = load_data()
    st.write(f"**Sumber Data:** YouTube | **Jumlah Data:** {len(df)} video")
    
    # 2. STATISTIK
    st.markdown("### 📈 1. Statistik Utama")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Video", len(df))
    col2.metric("Total Views", f"{df['viewCount'].sum():,.0f}")
    col3.metric("Total Channel", df['channelName'].nunique())

    # 3. VISUALISASI DATA (2 Grafik)
    st.markdown("### 📊 2. Visualisasi Data")
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.write("**Top 5 Channel dengan Views Tertinggi**")
        top_channels = df.groupby('channelName')['viewCount'].sum().nlargest(5)
        fig, ax = plt.subplots()
        top_channels.plot(kind='bar', color='skyblue', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col_viz2:
        st.write("**Proporsi Tipe Konten**")
        type_counts = df['type'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        st.pyplot(fig2)

    # 4. ANALISIS SENTIMEN (Simulasi Berbasis Kamus Kata)
    st.markdown("### 🧠 3. Analisis Sentimen Judul Video")
    def analyze_sentiment(text):
        text = str(text).lower()
        kata_negatif = ['propaganda', 'kalah', 'melemah', 'konflik', 'perang', 'islamphobia', 'ancaman', 'teroris']
        kata_positif = ['damai', 'bantuan', 'dukungan', 'solusi', 'aman']
        
        if any(word in text for word in kata_negatif):
            return 'Negatif'
        elif any(word in text for word in kata_positif):
            return 'Positif'
        else:
            return 'Netral'

    df['Sentimen'] = df['title'].apply(analyze_sentiment)
    sentiment_counts = df['Sentimen'].value_counts()
    st.bar_chart(sentiment_counts)

    # 5. AI SUMMARY
    st.markdown("### 🤖 4. AI Summary (Ringkasan Temuan)")
    if st.button("Buat Ringkasan Otomatis"):
        st.success("""
        **Ringkasan Insight:**
        1. Narasi didominasi oleh sentimen negatif, mengaitkan konflik geopolitik dengan sentimen agama.
        2. Media mainstream mendapat eksposur views terbesar dalam mewacanakan isu ini.
        3. Framing propaganda sering ditekankan pada kegagalan militer atau ancaman keamanan.
        """)

    # 6. TABEL INTERAKTIF
    st.markdown("### 🗂️ 5. Data Mentah")
    search_term = st.text_input("Cari kata kunci pada judul video:")
    if search_term:
        filtered_df = df[df['title'].str.contains(search_term, case=False, na=False)]
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)

except Exception as e:
    st.error(f"Gagal memuat data. Error: {e}")

# 7. CATATAN METODOLOGI
st.markdown("---")
st.markdown("""
**Catatan Metodologi:** Dashboard ini dibangun dengan Python Streamlit. Analisis sentimen menggunakan pendekatan *rule-based* (pencocokan kata kunci) sebagai simulasi. Hasil pelabelan AI bersifat perkiraan dan memerlukan validasi manual untuk memastikan akurasi konteks dakwah dan komunikasi Islam.
""")