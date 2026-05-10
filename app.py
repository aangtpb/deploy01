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
from streamlit_option_menu import option_menu

# Konfigurasi Halaman
st.set_page_config(page_title="Tax Issue @DJPForum Analyzer", layout="wide", page_icon="📊")

with st.sidebar:
    st.title("Navigasi")
    
    # Menggantikan st.radio dengan option_menu
    page = option_menu(
        menu_title=None, # Tidak perlu judul menu tambahan
        options=["Home", "Analisis"], 
        icons=["house", "bar-chart"], # Nama ikon dari Bootstrap Icons
        menu_icon="cast", 
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "#161414"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )    
    st.divider()
    if page == "Analisis":
            st.header("📂 Data Input")
            uploaded_files = st.file_uploader(
                "Upload semua file JSON hasil scraping", 
                type="json", 
                accept_multiple_files=True
            )
            
            # Indikator Visual Upload Status
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file berhasil diunggah.")
            else:
                st.info("Belum ada file yang diunggah.")

            st.header("⚙️ Filter Analisis")
            # Menggunakan form agar tidak rerun otomatis setiap kali tanggal diubah
            with st.form("filter_form"):
                start_dt = st.date_input("Tanggal Mulai", pd.to_datetime("2026-01-01"))
                end_dt = st.date_input("Tanggal Akhir", pd.to_datetime("2026-04-30"))
                
                # Tombol "Terapkan Filter"
                submit_button = st.form_submit_button(label='Terapkan Filter & Analisis')
                
            if not submit_button:
                st.warning("Silakan klik 'Terapkan Filter & Analisis' untuk memproses data.")
                
    else:
        st.info("Silakan ke halaman Analisis untuk mengolah data.")

# --- HALAMAN 1: HOME & INFO PROYEK (Statis) ---
if page == "Home":
    st.title("Pemanfaatan Isu DJPForum untuk Perbaikan Proses Bisnis DJP")
    
    # Deskripsi Ringkas Proyek
    st.header("📝 Penjelasan Proyek")
    col_info, col_img = st.columns([2, 1])
    
    with col_info:
        st.markdown(f"""
        **Latar Belakang & Data:**
        Pada masa implementasi Coretax, Direktorat Jenderal Pajak (DJP) menghadapi dinamika yang cukup kompleks, ditandai dengan munculnya berbagai isu dari ruang publik. Isu-isu tersebut banyak bersumber dari pemberitaan media, forum diskusi, serta kanal interaksi digital seperti DJPForum, yang mencerminkan pengalaman langsung pengguna terhadap sistem. Sebagian besar isu bersifat tidak terstruktur (unstructured), mencakup keluhan teknis (error system), kendala usability, hingga kekhawatiran terkait keamanan data. Volume data yang besar dan tersebar di berbagai sumber menyebabkan kesulitan dalam mengidentifikasi pola permasalahan secara sistematis.
        
        **DJP Forum** merupakan platform diskusi internal antarpegawai yang berfungsi sebagai media tanya jawab, berbagi pengalaman, dan pengelolaan pengetahuan (knowledge management) lintas proses bisnis DJP, seperti pelayanan, penyuluhan, pengawasan, pemeriksaan, dan penagihan. Fitur utama DJP Forum meliputi: 
            *   Pengguna dapat mengajukan pertanyaan maupun memberikan jawaban.
            *   Tersedia mekanisme upvote/downvote untuk menilai kualitas pertanyaan dan jawaban. 
            *   Adanya Subject Matter Expert (SME) pada tiap kategori untuk memvalidasi atau memilih jawaban terbaik.
                    
        Untuk memastikan fokus dan keterlaksanaan proyek, ruang lingkup ditetapkan sebagai berikut:
        *   **Sumber Data:** Data pertanyaan dan jawaban aplikasi DJP Forum.
        *   **Jenis Data:** Judul pertanyaan, isi pertanyaan, jawaban, tanggal posting, kategori/ topik (jika tersedia), metadata dasar.
        *   **Metode Akuisisi:** Scraping/ ekstraksi data internal
        *   **Periode:** Sesuai filter
        *   **Unit Analisis:** Pertanyaan pengguna (isu utama) dan jawaban (pendukung konteks).
        """)            


        with st.expander("📄 Panduan Struktur & Format File JSON"):
            st.markdown("""
            Aplikasi ini dirancang khusus untuk memproses file JSON yang dihasilkan oleh Scrapper **DJPForum Monitoring Downloader**. Agar fitur analisis (Statistik, NLP, dan Resolusi Waktu) berjalan optimal, pastikan file Anda mengikuti struktur berikut:

            ### 1. Kolom Wajib (Mandatori)
            Berdasarkan skrip pengunduh, atribut berikut harus tersedia:
            * **`postid`**: ID unik pertanyaan (digunakan untuk menghitung total tiket).
            * **`category`**: Kategori forum (akan dipetakan menjadi 'Kategori Utama' di dashboard).
            * **`question_title`**: Judul pertanyaan (digunakan sebagai input utama proses NLP).
            * **`question`**: pertanyaan (digunakan untuk Pencarian Tiket Semantik (AI) dan Semantic Flow).
            * **`question_date`**: Waktu saat pertanyaan dibuat (wajib untuk tren volume harian).
            * **`answered_date`**: Waktu respon pertama (wajib untuk analisis resolusi waktu).
            * **`userid`**: Identitas pengirim (dipetakan sebagai 'handle' pengguna).

            ### 2. Format Data
            Aplikasi mendukung format **DataTables AJAX Response** (format standar hasil scraping skrip PowerShell):
            * Data harus dibungkus dalam key `"data"` (contoh: `{"data": [...]}`).
            * Jika data berupa baris JSON tunggal per baris (JSONL), aplikasi akan otomatis menormalisasinya.

            ### 3. Contoh Cuplikan JSON yang Valid:
            ```json
            {
            "data": [
                {
                "postid": "13579",
                "category": "SPT Tahunan PPh Orang Pribadi",
                "question_title": "Gagal kirim CSV di menu e-Reporting",
                "question": "Selamat siang,...mohon bantuan...",
                "question_date": "2026-02-10 08:00:00",
                "answered_date": "2026-02-10 10:30:00",
                "userid": "nama.pegawai",
                "url": "[https://djpforum.intranet.pajak.go.id/](https://djpforum.intranet.pajak.go.id/)..."
                }
            ]
            }
            ```

            ### ⚠️ Catatan Penting:
            Untuk fitur **Pencarian Tiket Semantik (AI)** dan **Semantic Flow**, aplikasi akan mencoba mencari field tambahan bernama `question` (isi detail pertanyaan). Jika tidak ditemukan, sistem secara otomatis akan menggunakan field `question_title` sebagai pengganti data teks.
            """)


        # 3. Prioritas Solusi Tahap Awal (MVP)
        st.info("""                    
        Untuk implementasi awal yang cepat dan berdampak tinggi, **prioritas solusi tahap Awal (MVP):**
        *   **Topic Modelling:** Identifikasi top 10 topik dominan.
        *   **Clustering:** Pengelompokan pertanyaan serupa.
        *   **Visualisasi:** Dashboard analisis isu DJP Forum.

        **Metodologi Utama:**
        *   **Preprocessing:** Pembersihan teks menggunakan library **Sastrawi** (menghapus stopwords, slang, dan noise).
        *   **NLP Engine:** Menggunakan teknik **Bigram & Trigram** serta **Sentence-BERT** (`paraphrase-multilingual-MiniLM-L12-v2`) untuk memahami konteks kalimat penuh.
        *   **Pencarian Semantik:** Menggunakan **FAISS** untuk pencarian cepat berbasis makna, bukan sekadar kata kunci.
        """)
        
        #st.success("**Insight Utama:** Isu terkait **SPT** ditemukan sebagai topik dominan (58,5%), diikuti oleh **Billing** (15,4%).")

    with col_img:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", caption="NLP Data Processing Flow")

    st.divider()
    
    # Daftar Anggota Tim (Kelompok 4)
    st.header("👥 Tim Proyek (Kelompok 4-Dit.TPB)")
    team_data = [
        {"name": "Mas Firman Wardhana M. P.", "role": "Business Leader"},
        {"name": "Patriot Sonli Manumpak", "role": "Business Analyst & Data Analyst"},
        {"name": "Ima Marlinda Saragih", "role": "Business Analyst & Data Analyst"},
        {"name": "Pinurba Anandita", "role": "Business Analyst & Data Analyst"},
        {"name": "Arif Anggodo", "role": "Data Analyst & ML Engineer / Deployment Engineer"},
        {"name": "Devenni Putri Fau", "role": "Data Engineer & Data Scientist"},
        {"name": "Hana Kurniati", "role": "Business Analyst & Decision Support"},
        {"name": "Rohmat Kunto Wijoyo", "role": "Data Engineer & Data Scientist"}
    ]
    
    # Tampilan Tim
    rows = st.columns(4)
    for i, member in enumerate(team_data):
        with rows[i % 4]:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
            st.markdown(f"**{member['name']}**")
            st.caption(f"{member['role']}") # Menampilkan peran di bawah nama
            st.write("")

# --- HALAMAN 2: DASHBOARD ANALISIS ---
elif page == "Analisis":
    # Inisialisasi session state jika belum ada
    if "analyzed_df" not in st.session_state:
        st.session_state.analyzed_df = None

    if uploaded_files:
        # Proses data hanya jika tombol diklik
        if submit_button:
            with st.spinner("Sedang memproses data ..."):
                df_raw = load_and_normalize_data(uploaded_files)
                
                if not df_raw.empty:
                    # Filter berdasarkan tanggal dari form
                    df_filtered = df_raw[(df_raw["question_date"].dt.date >= start_dt) & 
                                         (df_raw["question_date"].dt.date <= end_dt)]
                    
                    # Jalankan preprocessing berat sekali saja
                    stopwords_id = get_stopwords()
                    df_filtered["combined_text"] = df_filtered["question_title"].fillna("") + " " + df_filtered["question"].fillna("")
                    df_filtered["cleaned_text"] = df_filtered["combined_text"].apply(lambda x: clean_text(x, stopwords_id))
                    df_filtered["issue_category"] = df_filtered["cleaned_text"].apply(classify_issue)
                    
                    # Simpan hasil ke session state
                    st.session_state.analyzed_df = df_filtered
        
        # Tampilkan dashboard jika data sudah tersedia di memori (session state)
        if st.session_state.analyzed_df is not None:
            df = st.session_state.analyzed_df

            tab_titles = [
                "📊 Statistik Deskriptif", 
                "🔍 Analisis Isu & Klaster", 
                "⏱️ Resolusi Waktu",
                "🔍 Pencarian Tiket (FAISS)",
                "🌊 Semantic Flow (Sankey)"
            ]
            
            # Gunakan st.tabs dengan parameter 'key' yang statis
            # Ini akan membuat Streamlit mengingat tab mana yang terakhir dibuka berdasarkan 'key' ini
            tabs = st.tabs(tab_titles)

            with tabs[0]:
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

            with tabs[1]:
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

            with tabs[2]:
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

            with tabs[3]:
                st.header("🔍 Pencarian Tiket Serupa (AI Powered)")
                user_query = st.text_input("Ketik masalah (misal: 'gagal login', 'kendala sertel', atau 'error billing', lalu tekan ENTER pada keyboard)")
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
                
                # --- BAGIAN DISCLAIMER & LOGIKA ---
                with st.expander("ℹ️ Detail Logika & Ketentuan Pencarian Semantik"):
                    st.markdown("""
                    Sistem ini menggunakan pendekatan *Natural Language Processing* (NLP) canggih untuk menemukan solusi berdasarkan konteks makna, bukan sekadar kesamaan kata kunci. Berikut adalah rincian teknisnya:

                    **1. Pembersihan Data (Preprocessing)**
                    * **Penghapusan Noise:** Teks dibersihkan dari tag HTML, emoji, tanda baca, dan karakter khusus menggunakan *BeautifulSoup* dan *regex*.
                    * **Stopword Removal:** Menggunakan library **Sastrawi** untuk menghapus kata-kata umum yang tidak memiliki makna kontekstual (seperti 'yang', 'di', 'ke').

                    **2. Model Kecerdasan Artifisial (Embedding)**
                    * **Engine:** Menggunakan model **Sentence-Transformer** `paraphrase-multilingual-MiniLM-L12-v2`.
                    * **Karakteristik:** Model ini mampu memetakan kalimat ke dalam ruang vektor 384 dimensi. Hal ini memungkinkan sistem memahami bahwa "gagal login" memiliki kedekatan makna dengan "kendala masuk aplikasi".

                    **3. Algoritma Pencarian (FAISS Indexing)**
                    * **Teknologi:** Menggunakan **FAISS (Facebook AI Similarity Search)** dengan tipe `IndexFlatL2`.
                    * **Metode:** Sistem menghitung jarak *Euclidean* antara vektor pertanyaan Anda dengan ribuan vektor data historis secara instan.
                    * **Output:** Hasil yang ditampilkan adalah **10 tiket teratas** dengan skor kemiripan (*similarity score*) tertinggi. Semakin kecil skor/jarak, semakin relevan hasilnya.

                    **Catatan:** Akurasi sangat bergantung pada kelengkapan teks input. Gunakan kalimat yang spesifik untuk hasil yang lebih presisi.
                    """)

            with tabs[4]:
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
                    horizontal=False,
                    key="sankey_radio_selection" # Memberikan key unik
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
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")
                        
        else:
            st.warning("Silakan klik 'Terapkan Filter & Analisis'.")
    else:
        st.info("👋 Selamat Datang! Silakan unggah file JSON, pilih filter tanggal, dan klik 'Terapkan Filter & Analisis' di sidebar untuk memulai analisis.")