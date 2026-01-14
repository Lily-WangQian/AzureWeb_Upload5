from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import ast
import logging

import pandas as pd
from pypdf import PdfReader

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CSV_PATH = "standards_keywords.csv"
ALLOWED_EXT = {".pdf"}

# ---- Model config (switchable in Azure App Settings) ----
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "false").lower() == "true"

# ---- Whole-document embedding settings ----
# We do NOT truncate the PDF. We embed in chunks and pool them.
EMB_CHUNK_CHARS = int(os.environ.get("EMB_CHUNK_CHARS", "3500"))     # chunk size
EMB_CHUNK_OVERLAP = int(os.environ.get("EMB_CHUNK_OVERLAP", "300"))  # overlap
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "0"))                  # 0 = no limit (use ALL chunks)

PREVIEW_CHARS = int(os.environ.get("PREVIEW_CHARS", "8000"))

# ---- Load standards CSV ----
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

# ---- Model cache ----
EMBED_MODEL = None

def get_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading model: {MODEL_NAME} on {device}")
        EMBED_MODEL = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=TRUST_REMOTE_CODE)
        logging.info("Model loaded.")
    return EMBED_MODEL

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=0).item())

# ---- FULL PDF reading (no truncation) ----
def read_pdf_text_full(path: str, num_header=0):
    try:
        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return "", "This PDF is encrypted. Please upload an unencrypted PDF."

        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            words = text.split()
            parts.append(" ".join(words[num_header:]))

        full = "\n".join(parts)
        return full, None
    except Exception as e:
        logging.exception("PDF read failed")
        return "", f"Cannot read this PDF on the server. Error: {type(e).__name__}"

def chunk_text(text: str, chunk_size: int, overlap: int):
    if not text:
        return
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        yield text[start:end]
        if end == n:
            break
        start = max(0, end - overlap)

def embed_whole_document(model, full_text: str):
    """
    ✅ Whole-document embedding:
    - split into chunks
    - encode each chunk
    - mean-pool into ONE document vector
    """
    chunks = []
    for i, ch in enumerate(chunk_text(full_text, EMB_CHUNK_CHARS, EMB_CHUNK_OVERLAP)):
        ch = (ch or "").strip()
        if ch:
            chunks.append(ch)
        if MAX_CHUNKS and len(chunks) >= MAX_CHUNKS:
            break

    if not chunks:
        return None

    # encode in small batches to avoid memory spikes
    # normalize_embeddings=True gives stable cosine similarity
    embs = model.encode(
        chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=8
    )

    # mean pool -> single vector
    doc_emb = embs.mean(dim=0)
    doc_emb = F.normalize(doc_emb, dim=0)
    return doc_emb

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

def lookup_standard_full(std_name: str):
    row = standards_df_copy.loc[standards_df_copy["Standards"].astype(str).str.strip() == std_name]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "standard": str(r["Standards"]).strip(),
        "pub_date": str(r["Publication Date"]).strip(),
        "body": str(r["Body"] or "").strip(),  # ✅ use full standard body for whole-doc similarity
    }

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

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

        # ✅ Full PDF text (no truncation)
        bank_text, pdf_err = read_pdf_text_full(fpath, num_header=0)
        if pdf_err:
            return render_template("index.html", standards=standards_list, selected=std,
                                   result=None, std_info=None, error=pdf_err, message=None)

        # ✅ Full standard text
        std_info = lookup_standard_full(std)
        if not std_info:
            return render_template("index.html", standards=standards_list, selected=std,
                                   result=None, std_info=None, error="Selected standard not found in CSV.", message=None)

        std_text = std_info["body"]
        if not std_text:
            return render_template("index.html", standards=standards_list, selected=std,
                                   result=None, std_info=None, error="Selected standard has empty Body text in CSV.", message=None)

        preview = (bank_text or "").strip()[:PREVIEW_CHARS]

        model = get_model()

        # ✅ Whole-document embeddings (chunked + pooled)
        bank_emb = embed_whole_document(model, bank_text)
        std_emb = embed_whole_document(model, std_text)

        similarity = cosine_sim(bank_emb, std_emb)

        bank_pub_date = detect_publication_date(bank_text)

        result = {
            "filename": fname,
            "standard": std,
            "bank_pub_date": bank_pub_date,
            # keep these fields for your existing index.html (can be blank)
            "bank_tfidf": "",
            "bank_contextual": "",
            "bank_combined": "",
            "similarity_score": similarity,
            "preview": preview
        }

        return render_template(
            "index.html",
            standards=standards_list,
            selected=std,
            result=result,
            std_info={
                "standard": std_info["standard"],
                "pub_date": std_info["pub_date"],
                "tfidf": "",        # not used in this whole-doc version
                "contextual": "",
                "combined": ""
            },
            error=None,
            message=f"Whole-document similarity computed using full PDF + full standard text (chunked embeddings)."
        )

    except Exception:
        logging.exception("Analyze failed")
        return render_template(
            "index.html",
            standards=standards_list,
            selected=std,
            result=None,
            std_info=None,
            error="Server error. Please check Log stream.",
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
