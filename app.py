from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import string
import ast
import logging
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CSV_PATH = "standards_keywords.csv"
ALLOWED_EXT = {".pdf"}

# ---------------- Azure-friendly environment variables ----------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Alibaba-NLP/gte-large-en-v1.5")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "false").lower() == "true"

# IMPORTANT: set MAX_PDF_CHARS=0 on Azure => process ENTIRE PDF (no truncation)
MAX_PDF_CHARS = int(os.environ.get("MAX_PDF_CHARS", "0"))  # 0 = unlimited
PREVIEW_CHARS = int(os.environ.get("PREVIEW_CHARS", "8000"))

USE_CHUNKING = os.environ.get("USE_CHUNKING", "true").lower() == "true"

# Embedding chunking (full-text but done in pieces)
EMB_CHUNK_CHARS = int(os.environ.get("EMB_CHUNK_CHARS", "3500"))
EMB_CHUNK_OVERLAP = int(os.environ.get("EMB_CHUNK_OVERLAP", "300"))

# KeyBERT chunking (keyword extraction in pieces)
KB_CHUNK_CHARS = int(os.environ.get("KB_CHUNK_CHARS", "6000"))
KB_CHUNK_OVERLAP = int(os.environ.get("KB_CHUNK_OVERLAP", "400"))
KB_TOPN_PER_CHUNK = int(os.environ.get("KB_TOPN_PER_CHUNK", "20"))

# 0 = unlimited chunks; set to e.g. 80 if you ever need a safety cap
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "0"))

# KeyBERT input length guard (still ok because we chunk)
MAX_KB_CHARS = int(os.environ.get("MAX_KB_CHARS", "40000"))

# Top keywords to show
TOP_TFIDF_N = int(os.environ.get("TOP_TFIDF_N", "5"))
TOP_CTX_N = int(os.environ.get("TOP_CTX_N", "5"))

# ---------------- Load standards CSV ----------------
standards_df = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8")
standards_df.columns = standards_df.columns.str.strip()

required_cols = ["Standards", "Body", "Publication Date", "No Stopwords",
                 "TFIDF Keywords", "Contextual Keywords", "Combined Keywords"]
for col in required_cols:
    if col not in standards_df.columns:
        standards_df[col] = ""

standards_df_copy = standards_df[required_cols].copy()

standards_list = (
    standards_df_copy["Standards"]
    .dropna()
    .astype(str)
    .str.strip()
    .sort_values()
    .unique()
    .tolist()
)

def parse_keywords(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, (list, tuple)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        return [t.strip() for t in txt.split(",") if t.strip()]
    return []

standards_df_copy["TFIDF Keywords List"] = standards_df_copy["TFIDF Keywords"].apply(parse_keywords)
standards_df_copy["Contextual Keywords List"] = standards_df_copy["Contextual Keywords"].apply(parse_keywords)
standards_df_copy["Combined Keywords List"] = standards_df_copy["Combined Keywords"].apply(parse_keywords)

# Keep comma-separated strings because HTML uses split(',')
standards_df_copy["TFIDF Keywords Display"] = standards_df_copy["TFIDF Keywords List"].apply(lambda lst: ", ".join(lst))
standards_df_copy["Contextual Keywords Display"] = standards_df_copy["Contextual Keywords List"].apply(lambda lst: ", ".join(lst))
standards_df_copy["Combined Keywords Display"] = standards_df_copy["Combined Keywords List"].apply(lambda lst: ", ".join(lst))

# ---------------- Stopwords + TFIDF ----------------
custom_stopwords = set([
    'shall','among','best','would','like','see','needs','•','their','to','requires','within','may',
    'lot','etc','b','with','without','pdfs','shows','tells','e','g','also','always','however','go','–',
    'by','for','that','and','or','0c','meet','includes','could','example','examples','chapter','an','a',
    'on','in','as','box','additionally','particularly','thereafter','please','the','there','has','have',
    'this','welcome','website','appendix','we','re',"we’re",'we re','should','be','com','rbc','at','from',
    'ceo','appendices','endnotes','is','ii','of','our'
])

def remove_stopwords(text: str):
    if not text:
        return ""
    sentence = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = sentence.split()
    return ' '.join([w.lower() for w in words if w.lower() not in custom_stopwords and not w.isdigit()])

# Use bigrams like your current approach
vectorizer = TfidfVectorizer(ngram_range=(2, 2))

def chunk_text(text: str, chunk_chars: int, overlap: int, max_chunks: int = 0):
    """
    Chunk by characters with overlap.
    max_chunks=0 => unlimited
    """
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks

def extract_tfidf_keywords_fulltext(text_no_stopwords: str, top_n: int):
    """
    Full-text TFIDF without truncation:
    Fit TFIDF over chunks and sum weights across chunks.
    """
    if not text_no_stopwords or not text_no_stopwords.strip():
        return []

    chunks = chunk_text(text_no_stopwords, chunk_chars=8000, overlap=300, max_chunks=MAX_CHUNKS)
    if not chunks:
        chunks = [text_no_stopwords]

    X = vectorizer.fit_transform(chunks)  # shape: (n_chunks, n_features)
    # Sum weights across chunks -> overall importance
    weights = X.sum(axis=0)               # shape: (1, n_features)
    weights = weights.A1                  # numpy array

    feature_names = vectorizer.get_feature_names_out()
    top_idx = weights.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_idx if weights[i] > 0]

# ---------------- Model cache ----------------
EMBED_MODEL = None
KEYEXTRACTOR = None
STANDARD_EMBEDDINGS = {}  # std_name -> tensor(1, dim)

def get_models():
    global EMBED_MODEL, KEYEXTRACTOR
    if EMBED_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading model: {MODEL_NAME} on {device}")
        EMBED_MODEL = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=TRUST_REMOTE_CODE)
        KEYEXTRACTOR = KeyBERT(EMBED_MODEL)
        logging.info("Model loaded.")
    return EMBED_MODEL, KEYEXTRACTOR

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=1).item())

def encode_fulltext_with_chunking(model, text: str):
    """
    Encode entire PDF text by chunking + mean pooling.
    Avoids passing a massive string into the embedding model.
    """
    if not text or not str(text).strip():
        return None

    if not USE_CHUNKING:
        emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return emb.unsqueeze(0)

    chunks = chunk_text(text, EMB_CHUNK_CHARS, EMB_CHUNK_OVERLAP, MAX_CHUNKS)
    if not chunks:
        return None

    # Batch encode chunks
    chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
    if chunk_embs is None or len(chunk_embs) == 0:
        return None

    # Mean pool then normalize
    pooled = chunk_embs.mean(dim=0, keepdim=True)
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled

def build_standard_embeddings_if_needed():
    global STANDARD_EMBEDDINGS
    if STANDARD_EMBEDDINGS:
        return
    model, _ = get_models()
    tmp = {}
    for _, row in standards_df_copy.iterrows():
        name = str(row["Standards"]).strip()
        combined = str(row["Combined Keywords"]).strip()
        if name and combined:
            # Standards strings are small, no need to chunk
            emb = model.encode(combined, convert_to_tensor=True, normalize_embeddings=True)
            tmp[name] = emb.unsqueeze(0)
    STANDARD_EMBEDDINGS = tmp
    logging.info(f"Cached standard embeddings: {len(STANDARD_EMBEDDINGS)}")

def extract_contextual_keywords_fulltext(text: str, top_n: int):
    """
    Extract contextual keywords from ENTIRE PDF by chunking and merging scores.
    """
    if not text or not text.strip():
        return []

    _, keyextractor = get_models()

    if not USE_CHUNKING:
        short_text = text[:MAX_KB_CHARS] if MAX_KB_CHARS > 0 else text
        results = keyextractor.extract_keywords(
            short_text,
            keyphrase_ngram_range=(2, 2),
            top_n=top_n,
            stop_words="english"
        )
        return [x[0] for x in results]

    chunks = chunk_text(text, KB_CHUNK_CHARS, KB_CHUNK_OVERLAP, MAX_CHUNKS)
    if not chunks:
        return []

    best_score = defaultdict(float)

    for ch in chunks:
        # Extra guard (rarely needed because we already chunk)
        ch2 = ch[:MAX_KB_CHARS] if MAX_KB_CHARS > 0 else ch
        results = keyextractor.extract_keywords(
            ch2,
            keyphrase_ngram_range=(2, 2),
            top_n=KB_TOPN_PER_CHUNK,
            stop_words="english"
        )
        for phrase, score in results:
            phrase = (phrase or "").strip()
            if phrase:
                best_score[phrase] = max(best_score[phrase], float(score))

    ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:top_n]]

# ---------------- PDF reading (NO truncation when MAX_PDF_CHARS=0) ----------------
def read_pdf_text(path: str, num_header=6):
    try:
        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return "", "This PDF is encrypted. Please upload an unencrypted PDF."

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            words = text.split()
            pages.append(" ".join(words[num_header:]))

        full = " ".join(pages)

        # MAX_PDF_CHARS=0 => unlimited
        if MAX_PDF_CHARS > 0:
            full = full[:MAX_PDF_CHARS]

        return full, None
    except Exception as e:
        logging.exception("PDF read failed")
        return "", f"Cannot read this PDF on the server. Error: {type(e).__name__}"

MONTHS = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)
def detect_publication_date(text: str):
    if not text:
        return ""
    m = re.search(fr"{MONTHS}\s+20\d{{2}}", text, re.IGNORECASE)
    if m:
        return m.group(0)
    m = re.search(r"\b20\d{2}\b", text)
    return m.group(0) if m else ""

def lookup_standard(std_name: str):
    row = standards_df_copy.loc[standards_df_copy["Standards"].astype(str).str.strip() == std_name]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "standard": r["Standards"],
        "pub_date": r["Publication Date"],
        "tfidf": r["TFIDF Keywords Display"],
        "contextual": r["Contextual Keywords Display"],
        "combined": r["Combined Keywords Display"],
    }

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        standards=standards_list,
        selected=None,
        result=None,
        std_info=None,
        error=None,
        message=None
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    std = (request.form.get("standard") or "").strip()
    pdf_file = request.files.get("bank_pdf")

    if not std:
        return render_template("index.html", standards=standards_list, selected=None,
                               result=None, std_info=None, error="Please select a standard.", message=None)

    if not pdf_file or pdf_file.filename == "":
        return render_template("index.html", standards=standards_list, selected=std,
                               result=None, std_info=None, error="Please upload a Bank ESG PDF.", message=None)

    ext = os.path.splitext(pdf_file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("index.html", standards=standards_list, selected=std,
                               result=None, std_info=None, error="Only .pdf is accepted.", message=None)

    fname = secure_filename(pdf_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    try:
        pdf_file.save(fpath)

        full_text, pdf_err = read_pdf_text(fpath, num_header=6)
        if pdf_err:
            return render_template("index.html", standards=standards_list, selected=std,
                                   result=None, std_info=None, error=pdf_err, message=None)

        preview = (full_text or "").strip()[:PREVIEW_CHARS] if PREVIEW_CHARS > 0 else ""

        build_standard_embeddings_if_needed()
        model, _ = get_models()

        bank_pub_date = detect_publication_date(full_text)

        # ---- FULL TEXT processing (chunked) ----
        ns_text = remove_stopwords(full_text)
        tfidf_list = extract_tfidf_keywords_fulltext(ns_text, top_n=TOP_TFIDF_N)
        contextual_list = extract_contextual_keywords_fulltext(full_text, top_n=TOP_CTX_N)

        combined_list = [k.strip() for k in (contextual_list + tfidf_list) if k and k.strip()]
        combined_str = ", ".join(combined_list)

        # bank embedding uses ENTIRE PDF via chunking+pooling
        bank_emb = encode_fulltext_with_chunking(model, full_text)

        # standard embedding uses cached combined keywords
        std_emb = STANDARD_EMBEDDINGS.get(std)

        similarity = cosine_sim(bank_emb, std_emb)

        std_info = lookup_standard(std)
        result = {
            "filename": fname,
            "standard": std,
            "bank_pub_date": bank_pub_date,
            "bank_tfidf": ", ".join(tfidf_list),
            "bank_contextual": ", ".join(contextual_list),
            "bank_combined": combined_str,
            "similarity_score": similarity,
            "preview": preview
        }

        return render_template(
            "index.html",
            standards=standards_list,
            selected=std,
            result=result,
            std_info=std_info,
            error=None,
            message=None
        )

    except Exception as e:
        logging.exception("Analyze failed")
        return render_template(
            "index.html",
            standards=standards_list,
            selected=std,
            result=None,
            std_info=None,
            error=f"Server error: {type(e).__name__}. Please check Log stream.",
            message=None
        )

    finally:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("WEBSITES_PORT") or os.environ.get("PORT") or 8000)
    app.run(host="0.0.0.0", port=port, debug=False)
