from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import re
import string
import ast
import logging
from collections import defaultdict
from io import BytesIO
import uuid
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CSV_PATH = "standards_keywords.csv"
ALLOWED_EXT = {".pdf"}

MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "false").lower() == "true"

MAX_PDF_CHARS = int(os.environ.get("MAX_PDF_CHARS", "0"))
PREVIEW_CHARS = int(os.environ.get("PREVIEW_CHARS", "8000"))

USE_CHUNKING = os.environ.get("USE_CHUNKING", "true").lower() == "true"
EMB_CHUNK_CHARS = int(os.environ.get("EMB_CHUNK_CHARS", "3500"))
EMB_CHUNK_OVERLAP = int(os.environ.get("EMB_CHUNK_OVERLAP", "300"))
KB_CHUNK_CHARS = int(os.environ.get("KB_CHUNK_CHARS", "6000"))
KB_CHUNK_OVERLAP = int(os.environ.get("KB_CHUNK_OVERLAP", "400"))
KB_TOPN_PER_CHUNK = int(os.environ.get("KB_TOPN_PER_CHUNK", "20"))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "0"))
MAX_KB_CHARS = int(os.environ.get("MAX_KB_CHARS", "40000"))
TOP_TFIDF_N = int(os.environ.get("TOP_TFIDF_N", "5"))
TOP_CTX_N = int(os.environ.get("TOP_CTX_N", "5"))

REPORT_STORE = {}
REPORT_TTL_SECONDS = 60 * 30


def store_report(payload: dict) -> str:
    report_id = uuid.uuid4().hex
    REPORT_STORE[report_id] = {"ts": time.time(), "payload": payload}
    return report_id


def get_report(report_id: str):
    obj = REPORT_STORE.get(report_id)
    if not obj:
        return None
    if time.time() - obj["ts"] > REPORT_TTL_SECONDS:
        REPORT_STORE.pop(report_id, None)
        return None
    return obj["payload"]


def cleanup_reports():
    now = time.time()
    to_del = [k for k, v in REPORT_STORE.items() if now - v["ts"] > REPORT_TTL_SECONDS]
    for k in to_del:
        REPORT_STORE.pop(k, None)


standards_df = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8")
standards_df.columns = standards_df.columns.str.strip()

required_cols = [
    "Standards",
    "Body",
    "Publication Date",
    "No Stopwords",
    "TFIDF Keywords",
    "Contextual Keywords",
    "Combined Keywords"
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

standards_df_copy["TFIDF Keywords Display"] = standards_df_copy["TFIDF Keywords List"].apply(lambda lst: ", ".join(lst))
standards_df_copy["Contextual Keywords Display"] = standards_df_copy["Contextual Keywords List"].apply(lambda lst: ", ".join(lst))
standards_df_copy["Combined Keywords Display"] = standards_df_copy["Combined Keywords List"].apply(lambda lst: ", ".join(lst))

custom_stopwords = set([
    'shall', 'among', 'best', 'would', 'like', 'see', 'needs', '•', 'their', 'to', 'requires',
    'within', 'may', 'lot', 'etc', 'b', 'with', 'without', 'pdfs', 'shows', 'tells', 'e', 'g',
    'also', 'always', 'however', 'go', '–', 'by', 'for', 'that', 'and', 'or', '0c', 'meet',
    'includes', 'could', 'example', 'examples', 'chapter', 'an', 'a', 'on', 'in', 'as', 'box',
    'additionally', 'particularly', 'thereafter', 'please', 'the', 'there', 'has', 'have',
    'this', 'welcome', 'website', 'appendix', 'we', 're', "we’re", 'we re', 'should', 'be',
    'com', 'rbc', 'at', 'from', 'ceo', 'appendices', 'endnotes', 'is', 'ii', 'of', 'our'
])


def remove_stopwords(text: str):
    if not text:
        return ""
    sentence = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = sentence.split()
    return ' '.join([w.lower() for w in words if w.lower() not in custom_stopwords and not w.isdigit()])


vectorizer = TfidfVectorizer(ngram_range=(2, 2))

EMBED_MODEL = None
KEYEXTRACTOR = None
STANDARD_EMBEDDINGS = {}


def chunk_text(text: str, chunk_chars: int, overlap: int, max_chunks: int = 0):
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
    if not text_no_stopwords or not text_no_stopwords.strip():
        return []

    chunks = chunk_text(text_no_stopwords, chunk_chars=8000, overlap=300, max_chunks=MAX_CHUNKS) or [text_no_stopwords]
    X = vectorizer.fit_transform(chunks)
    weights = X.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    top_idx = weights.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_idx if weights[i] > 0]


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
    if not text or not str(text).strip():
        return None

    if not USE_CHUNKING:
        emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return emb.unsqueeze(0)

    chunks = chunk_text(text, EMB_CHUNK_CHARS, EMB_CHUNK_OVERLAP, MAX_CHUNKS)
    if not chunks:
        return None

    chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
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
            emb = model.encode(combined, convert_to_tensor=True, normalize_embeddings=True)
            tmp[name] = emb.unsqueeze(0)

    STANDARD_EMBEDDINGS = tmp


def extract_contextual_keywords_fulltext(text: str, top_n: int):
    if not text or not text.strip():
        return []

    _, keyextractor = get_models()
    chunks = chunk_text(text, KB_CHUNK_CHARS, KB_CHUNK_OVERLAP, MAX_CHUNKS)
    if not chunks:
        return []

    best_score = defaultdict(float)
    for ch in chunks:
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


def normalize_phrase(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_keywords(keywords):
    seen = set()
    output = []
    for kw in keywords:
        clean = (kw or "").strip()
        norm = normalize_phrase(clean)
        if clean and norm not in seen:
            seen.add(norm)
            output.append(clean)
    return output


def get_top_similarity_keywords(model, bank_keywords, standard_keywords, top_n=10):
    bank_keywords = unique_keywords(bank_keywords)
    standard_keywords = unique_keywords(standard_keywords)

    if not bank_keywords or not standard_keywords:
        return []

    bank_embs = model.encode(bank_keywords, convert_to_tensor=True, normalize_embeddings=True)
    std_embs = model.encode(standard_keywords, convert_to_tensor=True, normalize_embeddings=True)

    sim_matrix = torch.matmul(bank_embs, std_embs.T)

    pairs = []
    for i, bank_kw in enumerate(bank_keywords):
        for j, std_kw in enumerate(standard_keywords):
            score = float(sim_matrix[i, j].item())
            pairs.append((score, bank_kw, std_kw))

    pairs.sort(key=lambda x: x[0], reverse=True)

    used_bank = set()
    used_std = set()
    output = []

    for score, bank_kw, std_kw in pairs:
        bank_norm = normalize_phrase(bank_kw)
        std_norm = normalize_phrase(std_kw)

        if bank_norm in used_bank or std_norm in used_std:
            continue

        used_bank.add(bank_norm)
        used_std.add(std_norm)

        output.append({
            "bank_keyword": bank_kw,
            "standard_keyword": std_kw,
            "score_pct": round(score * 100, 2),
            "same": bank_norm == std_norm
        })

        if len(output) >= top_n:
            break

    return output


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
        return "", f"Cannot read this PDF. Error: {type(e).__name__}"


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


def similarity_category(sim: float) -> str:
    if sim is None:
        return "Very low"
    if sim < 0.20:
        return "Very low"
    if sim < 0.40:
        return "Low"
    if sim <= 0.70:
        return "Moderate"
    return "High"


def similarity_ranges_rows(active_cat: str):
    rows = [
        ("Very low", "< 0.20"),
        ("Low", "0.20 – 0.40"),
        ("Moderate", "0.40 – 0.70"),
        ("High", "> 0.70"),
    ]
    return [{"cat": c, "range": r, "active": (c == active_cat)} for c, r in rows]


def build_pdf_report(payload: dict) -> bytes:
    styles = getSampleStyleSheet()
    story = []

    result = payload.get("result", {}) or {}
    category = payload.get("category", "Very low")
    top_similarity_keywords = result.get("top_similarity_keywords", []) or []

    filename = result.get("filename", "Report")
    standard = result.get("standard", "Standard")
    sim = float(result.get("similarity_score", 0.0) or 0.0)
    sim_pct = round(sim * 100.0, 2)

    story.append(Paragraph("Sustainability Report Alignment (SRA)", styles["Title"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph(f"Uploaded file: {filename}", styles["BodyText"]))
    story.append(Paragraph(f"Selected standard: {standard}", styles["BodyText"]))
    story.append(Spacer(1, 18))

    story.append(Paragraph(f"<b>Embedding-based Similarity Score: {sim_pct}%</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Category: {category}", styles["BodyText"]))
    story.append(Spacer(1, 16))

    if top_similarity_keywords:
        story.append(Paragraph("Top 10 Similarity Keywords", styles["Heading2"]))
        story.append(Spacer(1, 8))

        table_data = [[
            "#",
            "Uploaded Report Keyword",
            "Selected Standard Keyword",
            "Match Score"
        ]]

        for idx, item in enumerate(top_similarity_keywords, start=1):
            table_data.append([
                str(idx),
                item.get("bank_keyword", ""),
                item.get("standard_keyword", ""),
                f"{item.get('score_pct', 0)}%"
            ])

        keyword_table = Table(
            table_data,
            colWidths=[35, 180, 180, 75],
            repeatRows=1
        )

        keyword_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9edf7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (-1, 1), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        story.append(keyword_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "These keyword pairs show the strongest semantic similarity between the uploaded report and the selected standard.",
            styles["BodyText"]
        ))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=48,
        bottomMargin=48
    )
    doc.build(story)
    return buf.getvalue()


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        active_page="sra",
        standards=standards_list,
        selected=None,
        result=None,
        std_info=None,
        error=None,
        message=None,
        sra_summary=None,
        range_rows=None,
        report_id=None
    )


@app.route("/tool", methods=["GET"])
def tool_page():
    return render_template("tool.html", active_page="tool")


@app.route("/standards", methods=["GET"])
def standards_page():
    return render_template("standards.html", standards=standards_list, active_page="standards")


@app.route("/method", methods=["GET"])
def method_page():
    return render_template("method.html", active_page="method")


@app.route("/how", methods=["GET"])
def how_page():
    return render_template("how.html", active_page="how")


@app.route("/faq", methods=["GET"])
def faq():
    return render_template("faq.html", active_page="faq")


@app.route("/download_report/<report_id>", methods=["GET"])
def download_report(report_id):
    payload = get_report(report_id)
    if not payload:
        return "No report found. Please run Calculate first.", 400

    pdf_bytes = build_pdf_report(payload)
    fname = payload.get("result", {}).get("filename", "sra_report.pdf")
    safe_name = os.path.splitext(fname)[0]
    out_name = f"SRA_Report_{safe_name}.pdf"

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=out_name
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    cleanup_reports()

    std = (request.form.get("standard") or "").strip()
    pdf_file = request.files.get("bank_pdf")

    if not std:
        return render_template(
            "index.html",
            active_page="sra",
            standards=standards_list,
            selected=None,
            result=None,
            std_info=None,
            error="Please select a standard.",
            message=None,
            sra_summary=None,
            range_rows=None,
            report_id=None
        )

    if not pdf_file or pdf_file.filename == "":
        return render_template(
            "index.html",
            active_page="sra",
            standards=standards_list,
            selected=std,
            result=None,
            std_info=None,
            error="Please upload a Bank ESG PDF.",
            message=None,
            sra_summary=None,
            range_rows=None,
            report_id=None
        )

    ext = os.path.splitext(pdf_file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template(
            "index.html",
            active_page="sra",
            standards=standards_list,
            selected=std,
            result=None,
            std_info=None,
            error="Only .pdf is accepted.",
            message=None,
            sra_summary=None,
            range_rows=None,
            report_id=None
        )

    fname = secure_filename(pdf_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    try:
        pdf_file.save(fpath)

        full_text, pdf_err = read_pdf_text(fpath, num_header=6)
        if pdf_err:
            return render_template(
                "index.html",
                active_page="sra",
                standards=standards_list,
                selected=std,
                result=None,
                std_info=None,
                error=pdf_err,
                message=None,
                sra_summary=None,
                range_rows=None,
                report_id=None
            )

        preview = (full_text or "").strip()[:PREVIEW_CHARS] if PREVIEW_CHARS > 0 else ""

        build_standard_embeddings_if_needed()
        model, _ = get_models()

        bank_pub_date = detect_publication_date(full_text)

        ns_text = remove_stopwords(full_text)
        tfidf_list = extract_tfidf_keywords_fulltext(ns_text, top_n=TOP_TFIDF_N)
        contextual_list = extract_contextual_keywords_fulltext(full_text, top_n=TOP_CTX_N)

        combined_list = [k.strip() for k in (contextual_list + tfidf_list) if k and k.strip()]
        combined_list = unique_keywords(combined_list)
        combined_str = ", ".join(combined_list)

        bank_emb = encode_fulltext_with_chunking(model, full_text)
        std_emb = STANDARD_EMBEDDINGS.get(std)
        similarity = cosine_sim(bank_emb, std_emb)

        std_info = lookup_standard(std)
        std_combined_list = parse_keywords(std_info.get("combined", "")) if std_info else []

        top_similarity_keywords = get_top_similarity_keywords(
            model=model,
            bank_keywords=combined_list,
            standard_keywords=std_combined_list,
            top_n=10
        )

        result = {
            "filename": fname,
            "standard": std,
            "bank_pub_date": bank_pub_date,
            "bank_tfidf": ", ".join(tfidf_list),
            "bank_contextual": ", ".join(contextual_list),
            "bank_combined": combined_str,
            "similarity_score": similarity,
            "preview": preview,
            "top_similarity_keywords": top_similarity_keywords
        }

        cat = similarity_category(similarity)
        range_rows = similarity_ranges_rows(cat)

        sra_summary = {
            "filename": fname,
            "standard": std,
            "similarity_pct": round(similarity * 100.0, 2),
            "category": cat
        }

        payload = {"result": result, "std_info": std_info, "category": cat}
        report_id = store_report(payload)

        return render_template(
            "index.html",
            active_page="sra",
            standards=standards_list,
            selected=std,
            result=result,
            std_info=std_info,
            error=None,
            message=None,
            sra_summary=sra_summary,
            range_rows=range_rows,
            report_id=report_id
        )

    except Exception as e:
        logging.exception("Analyze failed")
        return render_template(
            "index.html",
            active_page="sra",
            standards=standards_list,
            selected=std,
            result=None,
            std_info=None,
            error=f"Server error: {type(e).__name__}. Please check terminal logs.",
            message=None,
            sra_summary=None,
            range_rows=None,
            report_id=None
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