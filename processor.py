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
        "atas","mohon","terkait","tentang","pajak","wajib","wp","coretax",
        "bantuannya","terima","kasih","bagaimana","apakah","bisa","akan","tersebut","nya"
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
    slang_dict = {"gk": "tidak", "ga": "tidak", "yg": "yang", "dgn": "dengan"}
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
import faiss

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