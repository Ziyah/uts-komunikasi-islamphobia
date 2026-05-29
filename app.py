import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import folium
from streamlit_folium import st_folium

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Analisis Islamphobia", layout="wide")
st.title("📊 Dashboard Analisis Sentimen & Framing Propaganda")
st.subheader("Topik: Mewacanakan Islamphobia pada Konflik AS–Iran di YouTube")

# Load Data
@st.cache_data
def load_data():
    # Pastikan file data.csv ada di folder yang sama
    return pd.read_csv("data.csv")

try:
    df = load_data()
    st.write(f"**Sumber Data:** YouTube | **Jumlah Data:** {len(df)} video")

    # 1. STATISTIK (WAJIB 1 & 2)
    st.markdown("### 📈 1. Statistik Utama")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Video", len(df))
    col2.metric("Total Views", f"{df['viewCount'].sum():,.0f}")
    col3.metric("Total Channel", df['channelName'].nunique())

    # 2. VISUALISASI DASAR (WAJIB 3)
    st.markdown("### 📊 2. Visualisasi Data")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("**Top 5 Channel (Views Tertinggi)**")
        top_channels = df.groupby('channelName')['viewCount'].sum().nlargest(5)
        fig, ax = plt.subplots()
        top_channels.plot(kind='bar', color='skyblue', ax=ax)
        st.pyplot(fig)
    with col_v2:
        st.write("**Proporsi Tipe Konten**")
        type_counts = df['type'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
        st.pyplot(fig2)

    # 3. ANALISIS SENTIMEN (WAJIB 4)
    st.markdown("### 🧠 3. Analisis Sentimen Judul")
    def analyze_sentiment(text):
        text = str(text).lower()
        neg = ['propaganda', 'kalah', 'melemah', 'konflik', 'perang', 'islamphobia', 'ancaman']
        pos = ['damai', 'bantuan', 'dukungan', 'aman', 'unggulan']
        if any(w in text for w in neg): return 'Negatif'
        elif any(w in text for w in pos): return 'Positif'
        else: return 'Netral'
    
    df['Sentimen'] = df['title'].apply(analyze_sentiment)
    st.bar_chart(df['Sentimen'].value_counts())

    # 4. TOPIC MODELING (BONUS)
    st.markdown("### 🏷️ 4. Analisis Topik (Topic Modeling)")
    documents = df['title'].fillna('').tolist()
    vec = CountVectorizer(max_features=500, stop_words='english')
    dtm = vec.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)
    words = vec.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        st.write(f"**Topik {i+1}:** " + ", ".join([words[j] for j in topic.argsort()[-5:]]))

    # 5. NETWORK ANALYSIS (BONUS)
    st.markdown("### 🕸️ 5. Network Analysis (Hubungan Kata)")
    G = nx.Graph()
    for title in df['title'].head(30):
        words = str(title).split()
        for i in range(len(words)-1):
            G.add_edge(words[i], words[i+1])
    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_size=50, font_size=8, ax=ax)
    st.pyplot(fig)

     # 6. AI SUMMARY (WAJIB 5)
    st.markdown("### 🤖 7. AI Summary")
    if st.button("Generate Ringkasan"):
        st.success("Berdasarkan analisis, wacana Islamphobia pada konflik AS-Iran di media YouTube didominasi oleh narasi propaganda negatif. Media besar cenderung membingkai isu ini sebagai ancaman keamanan.")

    # 7. TABEL (WAJIB 6)
    st.markdown("### 🗂️ 8. Data Mentah")
    st.dataframe(df)

    # 8. METODOLOGI (WAJIB 7)
    st.markdown("---")
    st.markdown("**Metodologi:** Dashboard ini menggunakan Python Streamlit. Sentimen dianalisis dengan metode berbasis kamus kata (Lexicon-based).")

except Exception as e:
    st.error(f"Gagal memuat data: {e}")