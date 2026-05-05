import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from processor import (
    load_and_normalize_data, 
    get_stopwords, 
    clean_text, 
    load_embedding_model, 
    perform_embedding,
    classify_issue,
    get_quadrant_data,
    generate_wordcloud,
    build_faiss_index,
    generate_sankey_plot,
    search_semantic,
    plot_sankey_3layer,
    plot_sankey_semantic,
    plot_sankey_topn
)

# Konfigurasi Halaman
st.set_page_config(page_title="Tax Issue at DJPForum Analyzer", layout="wide", page_icon="📊")

# --- SIDEBAR: Navigasi & Upload ---
with st.sidebar:
    st.title("🚀 Navigasi Sidebar:")
    page = st.radio("Pilih Menu:", ["🏠 Home & Info Proyek", "📊 Dashboard Analisis"])
    
    st.divider()
    if page == "📊 Dashboard Analisis":
        st.header("📂 Data Input")
        uploaded_files = st.file_uploader(
            "Upload semua file JSON hasil scraping", 
            type="json", 
            accept_multiple_files=True
            #label="Upload semua file JSON hasil scraping"
        )
        
        st.header("⚙️ Rentang Waktu Analisis")
        start_dt = st.date_input("Tanggal Mulai", pd.to_datetime("2026-01-01"))
        end_dt = st.date_input("Tanggal Akhir", pd.to_datetime("2026-04-30"))
    else:
        st.info("Silakan ke halaman Dashboard Analisis untuk mengolah data.")

# --- HALAMAN 1: HOME & INFO PROYEK (Statis) ---
if page == "🏠 Home & Info Proyek":
    st.title("Pemanfaatan Isu Terkait Coretax untuk Perbaikan Proses Bisnis DJP")
    
    # Deskripsi Ringkas Proyek
    st.header("📝 Penjelasan Proyek")
    col_info, col_img = st.columns([2, 1])
    
    with col_info:
        st.markdown(f"""
        **Latar Belakang & Data:**
        Proyek ini menganalisis data dari aplikasi **DJP Forum** periode Januari - April 2026. 
        Data yang diolah mencakup **19.552 baris data** yang diekstraksi untuk mendapatkan pola pertanyaan pengguna yang relevan dengan topik perpajakan saat ini.
        
        **Metodologi Utama:**
        *   **Preprocessing:** Pembersihan teks menggunakan library **Sastrawi** (menghapus stopwords, slang, dan noise).
        *   **NLP Engine:** Menggunakan teknik **Bigram & Trigram** serta **Sentence-BERT** (`paraphrase-multilingual-MiniLM-L12-v2`) untuk memahami konteks kalimat penuh.
        *   **Pencarian Semantik:** Menggunakan **FAISS** untuk pencarian cepat berbasis makna, bukan sekadar kata kunci.
        """)
        
        st.success("**Insight Utama:** Isu terkait **SPT** ditemukan sebagai topik dominan (58,5%), diikuti oleh **Billing** (15,4%).")

    with col_img:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", caption="NLP Data Processing Flow")

    st.divider()
    
    # Daftar Anggota Tim (Kelompok 4)
    st.header("👥 Anggota Tim Proyek (Kelompok 4)")
    team_members = [
        "Mas Firman Wardhana Mulya Praja", "Ima Marlinda Saragih", 
        "Arif Anggodo", "Hana Kurniati", 
        "Patriot Sonli Manumpak", "Pinurba Anandita", 
        "Devenni Putri Fau", "Rohmat Kunto Wijoyo"
    ]
    
    # Tampilan Grid untuk Tim
    rows = st.columns(4)
    for i, name in enumerate(team_members):
        with rows[i % 4]:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
            st.markdown(f"**{name}**")
            st.caption(f"Anggota Kelompok 4")
            st.write("")

# --- HALAMAN 2: DASHBOARD ANALISIS ---
elif page == "📊 Dashboard Analisis":
    if uploaded_files:
        # 1. Load & Normalize
        df = load_and_normalize_data(uploaded_files)
        
        # Filter Tanggal
        if not df.empty and "question_date" in df.columns:
            df = df[(df["question_date"].dt.date >= start_dt) & (df["question_date"].dt.date <= end_dt)]

        if not df.empty:
            # 2. Preprocessing
            with st.spinner("Membersihkan teks & melakukan embedding..."):
                stopwords_id = get_stopwords()
                df["combined_text"] = df["question_title"].fillna("") + " " + df["question"].fillna("")
                df["cleaned_text"] = df["combined_text"].apply(lambda x: clean_text(x, stopwords_id))
                df["issue_category"] = df["cleaned_text"].apply(classify_issue)

            # --- DASHBOARD TABS ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Statistik Deskriptif", 
                "🔍 Analisis Isu & Klaster", 
                "⏱️ Resolusi Waktu",
                "🔍 Pencarian Tiket (FAISS)",
                "🌊 Semantic Flow (Sankey)"
            ])

            with tab1:
                st.header("Statistik Deskriptif")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Tiket", len(df))
                col2.metric("Kategori Terbanyak", df["parent1"].mode()[0] if not df["parent1"].empty else "-")
                col3.metric("Isu Utama", df["issue_category"].mode()[0] if not df["issue_category"].empty else "-")

                # Chart Volume Harian
                daily_vol = df.groupby(df["question_date"].dt.date).size().reset_index(name='counts')
                fig_vol = px.line(daily_vol, x='question_date', y='counts', title="Tren Volume Pertanyaan Harian")
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Preview Data Mentah
                with st.expander("Lihat Data Mentah"):
                     st.write(df.head(100))

            with tab2:
                st.header("Analisis Isu & Semantik")
                c1, c2 = st.columns(2)
                with c1:
                    issue_counts = df["issue_category"].value_counts().reset_index()
                    fig_issue = px.pie(issue_counts, values='count', names='issue_category', title="Distribusi Kategori Isu")
                    st.plotly_chart(fig_issue)
                with c2:
                    parent_counts = df["parent1"].value_counts().head(10).reset_index()
                    fig_parent = px.bar(parent_counts, x='count', y='parent1', orientation='h', title="Top 10 Kategori Forum")
                    st.plotly_chart(fig_parent)

                st.subheader("Awan Kata (WordCloud) Isu Populer")
                fig_wc = generate_wordcloud(df, "cleaned_text")
                if fig_wc: st.pyplot(fig_wc)
                else:
                    st.warning("Teks tidak cukup untuk membuat WordCloud.")

            with tab3:
                st.header("Analisis Resolusi Waktu (4 Kuadran)")
                quad_data, med_res, med_count = get_quadrant_data(df)
                if quad_data is not None:
                    fig_quad = px.scatter(
                        quad_data, x="avg_resolution", y="count", text="parent1",
                        title="Kuadran Waktu Resolusi vs Volume",
                        labels={"avg_resolution": "Rata-rata Resolusi (Jam)", "count": "Jumlah Pertanyaan"}
                    )
                    # Tambah garis median untuk membentuk kuadran
                    fig_quad.add_vline(x=med_res, line_dash="dash", line_color="red")
                    fig_quad.add_hline(y=med_count, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_quad, use_container_width=True)
                    st.caption("Garis merah menunjukkan median. Posisi Kanan-Atas berarti: Isu Kritis (Volume Tinggi, Resolusi Lama).")
                else:
                    st.warning("Data tanggal tidak lengkap untuk menghitung resolusi waktu.")

            with tab4:
                st.header("🔍 Pencarian Tiket Serupa (AI Powered)")
                user_query = st.text_input("Ketik masalah (misal: 'gagal login', 'kendala sertel', atau 'error billing')")
                if user_query:
                    model = load_embedding_model()
                    embeddings = perform_embedding(df["cleaned_text"].tolist(), model)
                    index = build_faiss_index(embeddings)
                    search_results = search_semantic(user_query, model, index, df)
                    st.write(f"Menampilkan {len(search_results)} hasil yang paling relevan secara makna/konteks:")
                    for i, row in search_results.iterrows():
                        with st.expander(f"Hasil {i+1}: {row['question_title']} (Skor: {row['similarity_score']:.2f})"):
                            st.write(row['question'])
                            st.caption(f"Kategori: {row['parent1']}")

            with tab5:
                st.header("🌊 Advanced Semantic Flow Analysis")
                st.info("Visualisasi ini memetakan aliran kata kunci dominan ke contoh-contoh teks spesifik menggunakan algoritma clustering.")
                
                # Selector untuk jenis fitur
                sankey_type = st.radio(
                    "Pilih Visualisasi Sankey:",
                    [
                        "1. Sankey MVP (Semantic Topic Flow)",
                        "2. Sankey 3-Layer (Cluster → Mediator → Text)",
                        "3. Sankey MVP (Semantic Topic → Keyword → Text)",
                        "4. Sankey MVP (Clean + Top-N Right Layer)"
                    ],
                    horizontal=True
                )
                st.divider()

                with st.spinner("Sedang memproses algoritma clustering dan membangun visualisasi..."):
                    try:
                        if "1." in sankey_type:
                            fig= generate_sankey_plot(df)
                        elif "2." in sankey_type:
                            fig = plot_sankey_3layer(df)
                        elif "3." in sankey_type:
                            fig = plot_sankey_semantic(df)
                        else:
                            fig = plot_sankey_topn(df)
                            
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Penjelasan dinamis berdasarkan pilihan
                        if "3." in sankey_type:
                            st.info("💡 Mode ini hanya menampilkan 2 teks paling relevan (skor TF-IDF tertinggi) per klaster untuk menjaga kebersihan visual.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat merender plot: {e}")

        else:
            st.warning("Data tidak ditemukan untuk rentang tanggal tersebut.")
    else:
        st.info("👋 Selamat Datang! Silakan unggah file JSON di sidebar untuk memulai analisis.")