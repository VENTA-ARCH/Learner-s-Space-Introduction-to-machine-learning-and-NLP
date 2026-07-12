import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

URL_LINE_RE = re.compile(r"^URL:\s*(\S+)", re.MULTILINE)


def _extract_source_url(text: str) -> str:
    match = URL_LINE_RE.search(text)
    return match.group(1) if match else ""


def _load_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    title = path.stem
    for line in text.splitlines():
        if line.strip().startswith("#"):
            title = line.strip().lstrip("#").strip()
            break
    return {
        "doc_id": path.stem,
        "title": title,
        "text": text,
        "source_url": _extract_source_url(text),
    }


def _load_pdf(path: Path) -> dict:
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is not installed. Run `pip install pdfplumber` to ingest "
            f"PDF files, or convert {path.name} to markdown manually."
        )
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            page_text = re.sub(r"[ \t]+", " ", page_text)
            page_text = re.sub(r"\n{2,}", "\n\n", page_text)
            pages_text.append(page_text.strip())
    text = "\n\n".join(p for p in pages_text if p)

    title = path.stem
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 8:
            title = line
            break

    return {"doc_id": path.stem, "title": title, "text": text, "source_url": ""}


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(data_dir.glob("*.md")):
        docs.append(_load_markdown(path))
    for path in sorted(data_dir.glob("*.pdf")):
        docs.append(_load_pdf(path))
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n\n{para}" if overlap_text else para
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]

    if current:
        chunks.append(current)

    return chunks


def build_index():
    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents(DATA_DIR)
    print(f"  -> {len(docs)} documents loaded")

    all_chunks = []
    all_metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chunk_index": i,
                "source_url": doc.get("source_url", ""),
            })

    print(f"  -> {len(all_chunks)} chunks created (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks ...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  -> FAISS index built: {index.ntotal} vectors, dim={dim}")

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": all_chunks, "metadata": all_metadata, "embed_model": EMBED_MODEL_NAME},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved index + chunk metadata to {INDEX_DIR}/")
    print("Done. Run `streamlit run src/app.py` to launch the assistant.")


if __name__ == "__main__":
    build_index()
