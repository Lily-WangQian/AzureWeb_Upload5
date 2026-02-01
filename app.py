import os
import time
import torch
import traceback
from flask import Flask, request, render_template, jsonify

import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


# =========================================================
# 1. Flask App
# =========================================================
app = Flask(__name__)


# =========================================================
# 2. Environment Variables (Azure)
# =========================================================
USE_CHUNKING = os.getenv("USE_CHUNKING", "true").lower() == "true"

EMB_CHUNK_CHARS = int(os.getenv("EMB_CHUNK_CHARS", 3500))
EMB_CHUNK_OVERLAP = int(os.getenv("EMB_CHUNK_OVERLAP", 300))

KB_CHUNK_CHARS = int(os.getenv("KB_CHUNK_CHARS", 6000))
KB_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", 400))

TOP_TFIDF_N = int(os.getenv("TOP_TFIDF_N", 5))
TOP_CTX_N = int(os.getenv("TOP_CTX_N", 5))

PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", 8000))

MAX_PDF_CHARS = int(os.getenv("MAX_PDF_CHARS", 0))   # 0 = NO LIMIT
MAX_KB_CHARS = int(os.getenv("MAX_KB_CHARS", 40000))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 0))         # 0 = NO LIMIT


# =========================================================
# 3. FORCE Alibaba GTE Large Model (NO FALLBACK)
# =========================================================
MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
TRUST_REMOTE_CODE = True

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading embedding model: {MODEL_NAME} on {device}")

EMBED_MODEL = SentenceTransformer(
    MODEL_NAME,
    device=device,
    trust_remote_code=TRUST_REMOTE_CODE
)

KW_MODEL = KeyBERT(EMBED_MODEL)

print("[INFO] GTE-Large model loaded successfully")


# =========================================================
# 4. Utility Functions
# =========================================================
def read_pdf(file_storage):
    reader = PdfReader(file_storage)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    full_text = "\n".join(text)

    if MAX_PDF_CHARS > 0:
        full_text = full_text[:MAX_PDF_CHARS]

    return full_text


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

        if MAX_CHUNKS > 0 and len(chunks) >= MAX_CHUNKS:
            break

    return chunks


def embed_chunks(chunks):
    return EMBED_MODEL.encode(
        chunks,
        show_progress_bar=False,
        normalize_embeddings=True
    )


# =========================================================
# 5. Routes
# =========================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        start_time = time.time()

        bank_pdf = request.files.get("bank_pdf")
        std_pdf = request.files.get("std_pdf")

        if not bank_pdf or not std_pdf:
            return jsonify({"error": "Both PDFs are required"}), 400

        # ------------------ Read PDFs ------------------
        bank_text = read_pdf(bank_pdf)
        std_text = read_pdf(std_pdf)

        preview_text = bank_text[:PREVIEW_CHARS]

        # ------------------ TF-IDF ------------------
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform([bank_text, std_text])

        tfidf_scores = tfidf_matrix.toarray()
        tfidf_similarity = cosine_similarity(
            tfidf_scores[0:1],
            tfidf_scores[1:2]
        )[0][0]

        feature_names = tfidf.get_feature_names_out()
        tfidf_weights = tfidf_scores[0]
        top_tfidf_idx = tfidf_weights.argsort()[-TOP_TFIDF_N:][::-1]
        tfidf_keywords = [feature_names[i] for i in top_tfidf_idx]

        # ------------------ Contextual Embeddings ------------------
        if USE_CHUNKING:
            bank_chunks = chunk_text(bank_text, EMB_CHUNK_CHARS, EMB_CHUNK_OVERLAP)
            std_chunks = chunk_text(std_text, EMB_CHUNK_CHARS, EMB_CHUNK_OVERLAP)
        else:
            bank_chunks = [bank_text]
            std_chunks = [std_text]

        bank_emb = embed_chunks(bank_chunks)
        std_emb = embed_chunks(std_chunks)

        emb_similarity = cosine_similarity(bank_emb, std_emb).mean()

        # ------------------ KeyBERT ------------------
        kb_chunks = chunk_text(bank_text, KB_CHUNK_CHARS, KB_CHUNK_OVERLAP)
        kb_chunks = kb_chunks[:5]

        keywords = []
        for chunk in kb_chunks:
            kws = KW_MODEL.extract_keywords(
                chunk,
                top_n=TOP_CTX_N,
                stop_words="english"
            )
            keywords.extend([k[0] for k in kws])

        keywords = list(dict.fromkeys(keywords))

        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "tfidf_similarity": round(float(tfidf_similarity), 4),
            "embedding_similarity": round(float(emb_similarity), 4),
            "tfidf_keywords": tfidf_keywords,
            "contextual_keywords": keywords,
            "preview_text": preview_text,
            "elapsed_seconds": elapsed,
            "model_used": MODEL_NAME
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# =========================================================
# 6. Azure Entry Point
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
