from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import string
import ast
import time
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

# ---------------- Config ----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CSV_PATH = "standards_keywords.csv"

# Original method: top 5 + top 5
TOP_TFIDF_N = int(os.environ.get("TOP_TFIDF_N", "5"))
TOP_CTX_N = int(os.environ.get("TOP_CTX_N", "5"))

# Model (switch in Azure App Settings)
MODEL_NAME = os.environ.get("MODEL_NAME", "Alibaba-NLP/gte-large-en-v1.5")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "true").lower() == "true"

# KeyBERT chunking (keeps method, avoids timeouts)
KB_CHUNK_CHARS = int(os.environ.get("KB_CHUNK_CHARS", "6000"))
KB_CHUNK_OVERLAP = int(os.environ.get("KB_CHUNK_OVERLAP", "400"))
KB_TOPN_PER_CHUNK = int(os.environ.get("KB_TOPN_PER_CHUNK", "20"))
KB_NGRAM_MIN = int(os.environ.get("KB_NGRAM_MIN", "2"))
KB_NGRAM_MAX = int(os.environ.get("KB_NGRAM_MAX", "2"))

# Preview
PREVIEW_CHARS = int(os.environ.get("PREVIEW_CHARS", "3000"))

# ---------------- Load standards CSV ----------------
standards_df = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8")
standards_df.columns = standards_df.columns.str.strip()

required_cols = [
    "Standards",
    "Body",
    "Publication Date",
    "No Stopwords",
    "TFIDF Keywords",
    "Contextual Keywords",
    "Combined Keywords",
]
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

# ---------------- Keyword parsing for DISPLAY ----------------
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

standards_df_copy["TFIDF Keywords Display"] = standards_df_copy["TFIDF Keywords List"].apply(lambda lst: ", ".join(lst[:TOP_TFIDF_N]))
standards_df_copy["Contextual Keywords Display"] = standards_df_copy["Contextual Keywords List"].apply(lambda lst: ", ".join(lst[:TOP_CTX_N]))
# combined display: top 5 contextual + top 5 tfidf from the CSV if available
standards_df_copy["Combined Keywords Display"] = standards_df_copy["Combined Keywords List"].apply(
    lambda lst: ", ".join([x for x in lst if str(x).strip()][: (TOP_TFIDF_N + TOP_CTX_N)])
)

# ---------------- Stopwords + TFIDF ----------------
custom_stopwords = set([
    'shall','among','best','would','like','see','needs','•','their','to','requires','within','may',
    'lot','etc','b','with','without','pdfs','shows','tells','e','g','also','always','however','go','–',
    'by','for','that','and','or','0c','meet','includes','could','example','examples','chapter','an','a',
    'on','in','as','box','additionally','particularly','thereafter','please','the','there','has','have',
    'this','welcome','website','appendix','we','re',"we’re",'we re','should','be','com','rbc','at','from',
    'ceo','appendices','endnotes','is','ii','of','our'
])

def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    sentence = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = sentence.split()
    return ' '.join([w.lower() for w in words if w.lower() not in custom_stopwords and not w.isdigit()])

vectorizer = TfidfVectorizer(ngram_range=(2, 2))

def extract_tfidf_keywords(text: str, top_n=TOP_TFIDF_N):
    if not text or not text.strip():
        return []
    x = vectorizer.fit_transform([text])
    df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out()).transpose()
    return df.sort_values(by=0, ascending=False).head(top_n).index.tolist()

# ---------------- Model cache (LAZY) ----------------
EMBED_MODEL = None
KEYEXTRACTOR = None

# cache only the standards user selects (fast + safe)
STANDARD_EMBED_CACHE = {}  # std_name -> tensor(1, dim)

def get_models():
    global EMBED_MODEL, KEYEXTRACTOR
    if EMBED_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"[model] loading {MODEL_NAME} on {device}")
        t0 = time.time()
        EMBED_MODEL = SentenceTransformer(
            MODEL_NAME,
            device=device,
            trust_remote_code=TRUST_REMOTE_CODE
        )
        KEYEXTRACTOR = KeyBERT(EMBED_MODEL)
        logging.info(f"[model] loaded in {time.time() - t0:.1f}s")
    return EMBED_MODEL, KEYEXTRACTOR

def generate_embeddings(model, text: str):
    if not text or not str(text).strip():
        return None
    # fast + safe + normalized
    emb = model.encode(str(text), convert_to_tensor=True, normalize_embeddings=True)
    return emb.unsqueeze(0)  # (1, dim)

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=1).item())

# ---------------- KeyBERT chunked extraction ----------------
def chunk_text(text: str, chunk_size: int, overlap: int):
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        yield text[start:end]
        if end == n:
            break
        start = max(0, end - overlap)

def extract_contextual_keywords_chunked(full_text: str, top_n=TOP_CTX_N):
    """
    Same KeyBERT method, but chunked over the full PDF to avoid timeouts.
    """
    if not full_text or not full_text.strip():
        return []

    _, keyextractor = get_models()

    scores = defaultdict(float)
    for chunk in chunk_text(full_text, KB_CHUNK_CHARS, KB_CHUNK_OVERLAP):
        chunk = chunk.strip()
        if not chunk:
            continue

        results = keyextractor.extract_keywords(
            chunk,
            keyphrase_ngram_range=(KB_NGRAM_MIN, KB_NGRAM_MAX),
            top_n=KB_TOPN_PER_CHUNK,
            stop_words="english",
        )
        for phrase, score in results:
            phrase = (phrase or "").strip()
            if phrase:
                # keep best score seen
                scores[phrase] = max(scores[phrase], float(score))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:top_n]]

# ---------------- Combine keywords ----------------
def combine_keywords(ctx_list, tfidf_list):
    # keep order: contextual then tfidf
    combined = []
    for x in (ctx_list + tfidf_list):
        x = (x or "").strip()
        if x and x not in combined:
            combined.append(x)
    return combined

# ---------------- PDF reading ----------------
def read_pdf_full(path: str, num_header=6):
    """
    Reads the entire PDF (no truncation). This step is usually OK.
    The expensive part is KeyBERT + model load, which we optimized.
    """
    try:
        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return "", "This PDF is encrypted. Please upload an unencrypted PDF."

        cleaned_pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            words = text.split()
            cleaned_pages.append(" ".join(words[num_header:]))

        return " ".join(cleaned_pages), None
    except Exception:
        logging.exception("PDF read failed")
        return "", "Cannot read this PDF on the server."

# ---------------- Publication date ----------------
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

# ---------------- Lookup standard ----------------
def lookup_standard(std_name: str):
    row = standards_df_copy.loc[standards_df_copy["Standards"].astype(str).str.strip() == std_name]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "standard": str(r["Standards"]).strip(),
        "pub_date": str(r["Publication Date"]).strip(),
        # show top 5 from CSV
        "tfidf": str(r["TFIDF Keywords Display"]).strip(),
        "contextual": str(r["Contextual Keywords Display"]).strip(),
        "combined": str(r["Combined Keywords Display"]).strip(),
        # for embedding similarity we use the full Combined Keywords column (not just display)
        "combined_raw": str(r["Combined Keywords"]).strip(),
    }

def get_standard_embedding(std_name: str, combined_raw: str):
    """
    Compute ONLY the selected standard embedding, then cache it.
    (Avoids precomputing 100+ standards at startup.)
    """
    if std_name in STANDARD_EMBED_CACHE:
        return STANDARD_EMBED_CACHE[std_name]

    model, _ = get_models()
    emb = generate_embeddings(model, combined_raw)
    STANDARD_EMBED_CACHE[std_name] = emb
    return emb

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
        standards=standards_list,
        selected=None,
        result=None,
        std_info=None,
        error=None,
        message=None
    )

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    t0 = time.time()

    std = (request.form.get("standard") or "").strip()
    pdf_file = request.files.get("bank_pdf")

    if not std:
        return render_template("index.html", standards=standards_list, selected=None,
                               result=None, std_info=None, error="Please select a standard.", message=None)

    if not pdf_file or pdf_file.filename == "":
        return render_template("index.html", standards=standards_list, selected=std,
                               result=None, std_info=None, error="Please upload a Bank ESG report (PDF).", message=None)

    if os.path.splitext(pdf_file.filename)[1].lower() not in ALLOWED_EXT:
        return render_template("index.html", standards=standards_list, selected=std,
                               result=None, std_info=None, error="The uploaded file should be a PDF.", message=None)

    std_info = lookup_standard(std)
    if not std_info:
        return render_template("index.html", standards=standards_list, selected=std,
                               result=None, std_info=None, error="Standard not found in CSV.", message=None)

    fname = secure_filename(pdf_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    try:
        pdf_file.save(fpath)

        full_text, pdf_err = read_pdf_full(fpath, num_header=6)
        if pdf_err:
            return render_template("index.html", standards=standards_list, selected=std,
                                   result=None, std_info=None, error=pdf_err, message=None)

        preview = full_text[:PREVIEW_CHARS] if full_text else "(no text extracted)"
        bank_pub_date = detect_publication_date(full_text)

        # ---- ORIGINAL METHOD: TFIDF + KeyBERT -> combined -> embed -> cosine ----
        ns_text = remove_stopwords(full_text)
        tfidf_list = extract_tfidf_keywords(ns_text, top_n=TOP_TFIDF_N)

        # chunked KeyBERT over full text (same method, safer)
        contextual_list = extract_contextual_keywords_chunked(full_text, top_n=TOP_CTX_N)

        combined_list = combine_keywords(contextual_list, tfidf_list)
        combined_str = ", ".join(combined_list)

        model, _ = get_models()

        bank_embedding = generate_embeddings(model, combined_str)

        std_embedding = get_standard_embedding(std, std_info["combined_raw"])

        similarity_score = cosine_similarity(bank_embedding, std_embedding)

        result = {
            "filename": fname,
            "standard": std,
            "preview": preview,
            "bank_pub_date": bank_pub_date,
            "bank_tfidf": ", ".join(tfidf_list),
            "bank_contextual": ", ".join(contextual_list),
            "bank_combined": ", ".join(combined_list),
            "similarity_score": similarity_score,
        }

        msg = f"Done in {time.time() - t0:.1f}s (Azure-safe: lazy model + chunked KeyBERT + per-standard embedding cache)."
        return render_template("index.html",
            standards=standards_list,
            selected=std,
            result=result,
            std_info={
                "standard": std_info["standard"],
                "pub_date": std_info["pub_date"],
                "tfidf": std_info["tfidf"],
                "contextual": std_info["contextual"],
                "combined": std_info["combined"],
            },
            error=None,
            message=msg
        )

    except Exception:
        logging.exception("Analyze failed")
        return render_template("index.html",
            standards=standards_list,
            selected=std,
            result=None,
            std_info=None,
            error="Server error. Please check Log stream.",
            message=None
        )

    finally:
        # always delete uploaded file on Azure
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("WEBSITES_PORT") or os.environ.get("PORT") or 8000)
    app.run(host="0.0.0.0", port=port, debug=False)
