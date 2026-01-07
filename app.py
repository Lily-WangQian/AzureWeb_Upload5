from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import string
import ast
import logging

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

# ---- Azure-safe defaults (override in Azure App Settings later) ----
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "false").lower() == "true"

MAX_PDF_CHARS = int(os.environ.get("MAX_PDF_CHARS", "120000"))   # total text to process
MAX_KB_CHARS  = int(os.environ.get("MAX_KB_CHARS", "40000"))     # KeyBERT input length
PREVIEW_CHARS = int(os.environ.get("PREVIEW_CHARS", "8000"))     # show preview on page

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

# IMPORTANT: keep comma-separated strings because your HTML uses split(',')
standards_df_copy["TFIDF Keywords Display"] = standards_df_copy["TFIDF Keywords List"].apply(lambda lst: ", ".join(lst))
standards_df_copy["Contextual Keywords Display"] = standards_df_copy["Contextual Keywords List"].apply(lambda lst: ", ".join(lst))
standards_df_copy["Combined Keywords Display"] = standards_df_copy["Combined Keywords List"].apply(lambda lst: ", ".join(lst))

# ---- Stopwords + TFIDF ----
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

vectorizer = TfidfVectorizer(ngram_range=(2, 2))

def extract_tfidf_keywords(text: str, top_n=5):
    if not text or not text.strip():
        return []
    x = vectorizer.fit_transform([text])
    df_kw = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out()).transpose()
    return df_kw.sort_values(by=0, ascending=False).head(top_n).index.tolist()

# ---- Model cache ----
EMBED_MODEL = None
KEYEXTRACTOR = None
STANDARD_EMBEDDINGS = {}   # std_name -> tensor(1, dim)

def get_models():
    global EMBED_MODEL, KEYEXTRACTOR
    if EMBED_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading model: {MODEL_NAME} on {device}")
        EMBED_MODEL = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=TRUST_REMOTE_CODE)
        KEYEXTRACTOR = KeyBERT(EMBED_MODEL)
        logging.info("Model loaded.")
    return EMBED_MODEL, KEYEXTRACTOR

def encode_text(model, text: str):
    if not text or not str(text).strip():
        return None
    emb = model.encode(str(text), convert_to_tensor=True, normalize_embeddings=True)
    return emb.unsqueeze(0)

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
            emb = encode_text(model, combined)
            if emb is not None:
                tmp[name] = emb
    STANDARD_EMBEDDINGS = tmp
    logging.info(f"Cached standard embeddings: {len(STANDARD_EMBEDDINGS)}")

def extract_contextual_keywords(text: str, top_n=5):
    if not text or not text.strip():
        return []
    _, keyextractor = get_models()
    short_text = text[:MAX_KB_CHARS]  # keep fast on Azure
    results = keyextractor.extract_keywords(
        short_text,
        keyphrase_ngram_range=(2, 2),
        top_n=top_n,
        stop_words="english"
    )
    return [x[0] for x in results]

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=1).item())

# ---- PDF reading ----
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

        # preview MUST exist because your HTML uses result.preview
        preview = (full_text or "").strip()[:PREVIEW_CHARS]

        # cache standards + load models
        build_standard_embeddings_if_needed()
        model, _ = get_models()

        bank_pub_date = detect_publication_date(full_text)

        # keywords (comma-separated strings for HTML split(','))
        ns_text = remove_stopwords(full_text)
        tfidf_list = extract_tfidf_keywords(ns_text, top_n=5)
        contextual_list = extract_contextual_keywords(full_text, top_n=5)

        combined_list = contextual_list + tfidf_list
        combined_list = [k.strip() for k in combined_list if k and k.strip()]
        combined_str = ", ".join(combined_list)

        bank_emb = encode_text(model, combined_str)
        std_emb = STANDARD_EMBEDDINGS.get(std)

        similarity = cosine_sim(bank_emb, std_emb)

        std_info = lookup_standard(std)
        result = {
            "filename": fname,
            "standard": std,
            "bank_pub_date": bank_pub_date,
            "bank_tfidf": ", ".join(tfidf_list),
            "bank_contextual": ", ".join(contextual_list),
            "bank_combined": ", ".join(combined_list),
            "similarity_score": similarity,   # your HTML uses result.similarity_score
            "preview": preview                # your HTML uses result.preview
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
