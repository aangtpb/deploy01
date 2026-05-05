import pandas as pd
import numpy as np
import json
import io
import re
import html
import time
from collections import Counter

# NLP & ML Libraries
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from bs4 import BeautifulSoup
from gensim.models.phrases import Phrases, Phraser
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# 1. DATA LOADING & NORMALIZATION (Cell 3 & 17)
# =========================================================
@st.cache_data
def load_and_normalize_data(uploaded_files):
    data_list = []
    for file in uploaded_files:
        try:
            # Handle BOM & JSONL format
            content = file.getvalue().decode("utf-8-sig", errors="ignore")
            for line in content.splitlines():
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
        except Exception as e:
            st.error(f"Error parsing {file.name}: {e}")

    df = pd.DataFrame(data_list)
    
    # Flattening logic (Cell 3)
    if "data" in df.columns:
        df["data"] = df["data"].apply(lambda x: x if isinstance(x, list) else [x])
        df = df.explode("data", ignore_index=True)
        df_normalized = pd.json_normalize(df["data"])
        df = df.drop(columns=["data"]).join(df_normalized)

    # Required Columns & Date Parsing (Cell 3 & 18)
    required_cols = ["postid", "handle", "question_title", "question", 
                     "question_date", "answered_date", "selected_date", "parent1"]
    for col in required_cols:
        if col not in df.columns: df[col] = None
            
    for col in ["question_date", "answered_date", "selected_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Time Resolution Logic (Cell 18)
    if "answered_date" in df.columns and "question_date" in df.columns:
        df["response_time_h"] = (df["answered_date"] - df["question_date"]).dt.total_seconds() / 3600
    if "selected_date" in df.columns and "question_date" in df.columns:
        df["resolution_time_h"] = (df["selected_date"] - df["question_date"]).dt.total_seconds() / 3600

    return df

# =========================================================
# 2. TEXT CLEANING (Cell 4 & 5)
# =========================================================
def get_stopwords():
    factory = StopWordRemoverFactory()
    stop_words = set(factory.get_stop_words())
    custom_stops = {
    "atas","bawah","mohon","terkait","tentang","untuk",
    "dengan","dan","atau","yang","di","ke","dari",
    "saya","aku","kami","kita","ini","itu","adalah",
    "pada","dalam","sebagai","agar","bisa","akan",
    "selamat","pagi","siang","sore","namun",
    "tolong","silakan","please","help","problem","issue",
    "pajak","wajib","wp","coretax","bantuannya","melati",
    "lanjut","request","case","data","masa",
    "tersebut","nomor","kasus","muncul","terima","kasih",
    "bagaimana","apakah","nya"
    }
    return stop_words.union(custom_stops)

def clean_text(text, stopwords_id):
    # HTML Cleaning
    text = html.unescape(str(text).lower())
    text = re.sub(r"<(script|style).*?>.*?</\1>", " ", text, flags=re.DOTALL)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")
    
    # Regex Cleaning
    text = re.sub(r"http\S+|www\S+", " ", text) # URL
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Non-Alphanum
    text = re.sub(r"\b\d+\b", " ", text) # Standalone Numbers
    text = re.sub(r"\s+", " ", text).strip()
    
    # Slang & Stopwords
    slang_dict = {"gk": "tidak","ga": "tidak","nggak": "tidak",
    "bgt": "banget","yg": "yang","dgn": "dengan",
    "dr": "dari","tdk": "tidak","eror": "error",
    "errorr": "error","instal": "install","installl": "install",
    "piip": "pip"}
    words = [slang_dict.get(w, w) for w in text.split()]
    words = [w for w in words if w not in stopwords_id and len(w) > 1]
    
    return " ".join(words)

# =========================================================
# 3. N-GRAM & SEMANTIC LOGIC (Cell 6 - 10)
# =========================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def perform_embedding(texts, _model):
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index

# =========================================================
# 4. ISSUE DETECTION (Cell 15)
# =========================================================
def classify_issue(text):
    issue_keywords = {
        "error": ["error", "gagal", "tidak bisa", "bug", "exception"],
        "billing": ["billing", "kode", "pembayaran", "deposit"],
        "spt": ["spt", "pelaporan", "pph", "ppn"],
        "validasi": ["validasi", "tidak valid", "reject"],
        "pengembalian": ["pengembalian", "refund", "restitusi"],
        "akses": ["login", "akses", "authorize", "aksesnya"]
    }
    text = text.lower()
    for label, keywords in issue_keywords.items():
        if any(kw in text for kw in keywords):
            return label
    return "other"

# =========================================================
# 5. QUADRANT LOGIC (Fitur Opsional)
# =========================================================
def get_quadrant_data(df):
    if "resolution_time_h" not in df.columns:
        return None
    
    summary = df.groupby("parent1").agg(
        avg_resolution=("resolution_time_h", "mean"),
        count=("parent1", "count")
    ).reset_index()
    
    # Tentukan threshold kuadran (median)
    med_res = summary["avg_resolution"].median()
    med_count = summary["count"].median()
    
    return summary, med_res, med_count

# =========================================================
# 6. TAMBAHAN UNTUK WORD CLOUD
# =========================================================
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(df, column_name):
    # Gabungkan semua baris teks menjadi satu string besar
    text = " ".join(df[column_name].astype(str))
    
    if not text.strip():
        return None

    # Inisialisasi WordCloud
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    # Buat plot menggunakan Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


# =========================================================
# 7. TAMBAHAN UNTUK TAB 4 FITUR PENCARIAN TIKET SERUPA
# =========================================================

# Di dalam processor.py
def build_faiss_index(embeddings):
    import faiss
    import numpy as np
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Menggunakan Inner Product untuk Cosine Similarity
    index.add(embeddings.astype('float32'))
    return index

def search_semantic(query, _model, index, original_df, top_k=10):
    # 1. Ubah input user menjadi vektor (embedding)
    query_vector = _model.encode([query], normalize_embeddings=True)
    
    # 2. Cari di index FAISS untuk menemukan 'tetangga terdekat'
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    
    # 3. Ambil data asli berdasarkan index yang ditemukan
    results = original_df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]
    return results



# ... (kode existing Anda) ...

def generate_sankey_plot(df, text_column="cleaned_text", top_clusters=15, threshold=0.70):
    # 1. Penyiapan Data (Limit untuk performa agar tidak lag di Streamlit)
    sample_df = df.head(1000) # Batasi 1000 baris pertama untuk kestabilan plot
    sample_texts = sample_df[text_column].tolist()
    
    # Load model & hitung embedding
    model = load_embedding_model()
    sample_emb = perform_embedding(sample_texts, model)

    # 2. Simple Clustering Logic (dari file sankey1.txt)
    def simple_cluster(texts, embeddings, threshold):
        clusters = []
        used = set()
        for i in range(len(texts)):
            if i in used: continue
            group = [i]
            used.add(i)
            for j in range(len(texts)):
                if j not in used:
                    sim = np.dot(embeddings[i], embeddings[j])
                    if sim >= threshold:
                        group.append(j)
                        used.add(j)
            clusters.append(group)
        return clusters

    clusters = simple_cluster(sample_texts, sample_emb, threshold)
    clusters = [c for c in clusters if len(c) > 2] # Filter noise
    clusters = sorted(clusters, key=len, reverse=True)[:top_clusters] # Top clusters

    # 3. TF-IDF untuk Label
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(sample_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    def get_cluster_label(texts_cluster):
        if not texts_cluster: return "unknown"
        tfidf = vectorizer.transform(texts_cluster)
        mean_scores = tfidf.mean(axis=0).A1
        top_idx = mean_scores.argsort()[-1]
        return feature_names[top_idx]

    # 4. Build Sankey Structure
    labels = []
    label_map = {}
    source, target, value = [], [], []

    for i, cluster in enumerate(clusters):
        cluster_texts = [sample_texts[idx] for idx in cluster]
        cluster_label = f"Topic: {get_cluster_label(cluster_texts)}"

        if cluster_label not in label_map:
            label_map[cluster_label] = len(labels)
            labels.append(cluster_label)
        
        cluster_id = label_map[cluster_label]

        for idx in cluster[:3]: # Ambil 3 contoh per klaster agar tidak terlalu penuh
            text_label = (sample_texts[idx][:30] + '...') if len(sample_texts[idx]) > 30 else sample_texts[idx]
            if text_label not in label_map:
                label_map[text_label] = len(labels)
                labels.append(text_label)
            
            source.append(cluster_id)
            target.append(label_map[text_label])
            value.append(1)

    # 5. Create Figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=source, target=target, value=value)
    )])
    
    fig.update_layout(title_text="Sankey MVP (Semantic Topic Flow)", font_size=12)
    return fig


# Fungsi Helper untuk Clustering (Shared logic)
def get_clusters_and_vectorizer(df, text_column, threshold=0.70):
    sample_texts = df[text_column].head(2000).tolist() # Limit 2000
    model = load_embedding_model()
    sample_emb = perform_embedding(sample_texts, model)
    
    def simple_cluster(texts, embeddings, threshold):
        clusters, used = [], set()
        for i in range(len(texts)):
            if i in used: continue
            group = [i]; used.add(i)
            for j in range(len(texts)):
                if j not in used:
                    sim = np.dot(embeddings[i], embeddings[j])
                    if sim >= threshold:
                        group.append(j); used.add(j)
            clusters.append(group)
        return clusters

    clusters = simple_cluster(sample_texts, sample_emb, threshold)
    clusters = sorted([c for c in clusters if len(c) > 2], key=len, reverse=True)[:15]
    
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(sample_texts)
    
    return sample_texts, clusters, vectorizer

# 1. Sankey 3-Layer (Cluster -> Mediator -> Text)
def plot_sankey_3layer(df, text_column="cleaned_text"):
    texts, clusters, vec = get_clusters_and_vectorizer(df, text_column)
    feature_names = np.array(vec.get_feature_names_out())
    labels, label_map, source, target, value = [], {}, [], [], []

    for i, cluster in enumerate(clusters):
        cluster_name = f"Cluster_{i}"
        if cluster_name not in label_map:
            label_map[cluster_name] = len(labels); labels.append(cluster_name)
        
        # Mediator Keywords
        tfidf = vec.transform([texts[idx] for idx in cluster])
        keywords = feature_names[tfidf.mean(axis=0).A1.argsort()[-2:][::-1]]
        
        for kw in keywords:
            if kw not in label_map: label_map[kw] = len(labels); labels.append(kw)
            source.append(label_map[cluster_name]); target.append(label_map[kw]); value.append(len(cluster))
            
            for idx in cluster[:3]:
                txt = texts[idx][:40]
                if txt not in label_map: label_map[txt] = len(labels); labels.append(txt)
                source.append(label_map[kw]); target.append(label_map[txt]); value.append(1)

    fig = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15, thickness=18), 
                                  link=dict(source=source, target=target, value=value))])
    fig.update_layout(title="Sankey 3-Layer (Cluster → Mediator → Text)", font_size=10)
    return fig

# 2. Sankey MVP (Semantic Topic -> Keyword -> Text)
def plot_sankey_semantic(df, text_column="cleaned_text"):
    texts, clusters, vec = get_clusters_and_vectorizer(df, text_column)
    feature_names = np.array(vec.get_feature_names_out())
    labels, label_map, source, target, value = [], {}, [], [], []

    for cluster in clusters:
        cluster_texts = [texts[i] for i in cluster]
        tfidf_res = vec.transform(cluster_texts).mean(axis=0).A1
        topic_label = feature_names[tfidf_res.argmax()] # Semantic Topic[cite: 5]

        if topic_label not in label_map:
            label_map[topic_label] = len(labels); labels.append(topic_label)
        
        keywords = feature_names[tfidf_res.argsort()[-2:][::-1]]
        for kw in keywords:
            if kw not in label_map: label_map[kw] = len(labels); labels.append(kw)
            source.append(label_map[topic_label]); target.append(label_map[kw]); value.append(len(cluster))
            
            for idx in cluster[:3]:
                txt = texts[idx][:40]
                if txt not in label_map: label_map[txt] = len(labels); labels.append(txt)
                source.append(label_map[kw]); target.append(label_map[txt]); value.append(1)

    fig = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15, thickness=18), 
                                  link=dict(source=source, target=target, value=value))])
    fig.update_layout(title="Sankey MVP (Semantic Topic → Keyword → Text)", font_size=10)
    return fig

# 3. Sankey MVP (Clean + Top-N Right Layer)
def plot_sankey_topn(df, text_column="cleaned_text"):
    texts, clusters, vec = get_clusters_and_vectorizer(df, text_column)
    feature_names = np.array(vec.get_feature_names_out())
    labels, label_map, source, target, value = [], {}, [], [], []

    for cluster in clusters:
        cluster_texts = [texts[i] for i in cluster]
        tfidf_matrix = vec.transform(cluster_texts)
        tfidf_res = tfidf_matrix.mean(axis=0).A1
        topic_label = feature_names[tfidf_res.argmax()]
        
        if topic_label not in label_map:
            label_map[topic_label] = len(labels); labels.append(topic_label)
        
        keywords = feature_names[tfidf_res.argsort()[-2:][::-1]]
        for kw in keywords:
            if kw not in label_map: label_map[kw] = len(labels); labels.append(kw)
            source.append(label_map[topic_label]); target.append(label_map[kw]); value.append(len(cluster))
            
            # TOP-N Right Layer logic[cite: 6]
            row_sums = tfidf_matrix.toarray().sum(axis=1)
            top_idx = np.argsort(row_sums)[-2:] # TOP_K_TEXT = 2
            for i in top_idx:
                txt = cluster_texts[i][:40]
                if txt not in label_map: label_map[txt] = len(labels); labels.append(txt)
                source.append(label_map[kw]); target.append(label_map[txt]); value.append(1)

    fig = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15, thickness=18), 
                                  link=dict(source=source, target=target, value=value))])
    fig.update_layout(title="Sankey MVP (Clean + Top-N Right Layer)", font_size=10)
    return fig