import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from google import genai
from google.genai import types as genai_types
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"

TOP_K = 4
MIN_SCORE = 0.30
STRONG_SCORE = 0.55

LLM_MODEL = os.environ.get("INSTI_ASSIST_MODEL", "gemini-flash-latest")

SYSTEM_PROMPT = (
    "You are IITB Insti-Assist, an academic assistant for IIT Bombay. "
    "Answer the question using ONLY the information in the provided context. "
    "Do not use any outside knowledge. If the context does not contain the "
    "answer, say exactly: \"I don't know — that isn't covered in the course "
    "material I have access to.\" Keep answers concise and cite which source "
    "each fact came from using the bracketed source labels, e.g. [Source 1]."
)


class InstiAssist:
    def __init__(self):
        if not (INDEX_DIR / "faiss.index").exists():
            raise FileNotFoundError(
                "No index found. Run `python src/ingest.py` first to build the index."
            )

        with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        embed_model_name = data["embed_model"]

        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.embed_model = SentenceTransformer(embed_model_name)
        self.llm_client = genai.Client()

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        query_vec = self.embed_text(question)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "chunk_index": meta["chunk_index"],
                "source_url": meta.get("source_url", ""),
            })
        return results

    def build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            context = "(No relevant context was found in the knowledge base.)"
        else:
            context = "\n\n".join(
                f"[Source {i+1}: {c['title']}]\n{c['text']}"
                for i, c in enumerate(retrieved_chunks)
            )
        return f"Context:\n{context}\n\nQuestion: {question}"

    def _split_sentences(self, chunk_text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text.replace("\n", " "))
        return [s.strip() for s in sentences if s.strip()]

    def _best_sentences(self, question: str, chunk_text: str, max_sentences: int = 3) -> list[str]:
        sentences = self._split_sentences(chunk_text)
        if not sentences:
            return [chunk_text.strip()] if chunk_text.strip() else []

        q_words = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 2)
        scored = []
        for s in sentences:
            s_words = set(w.lower() for w in re.findall(r"\w+", s))
            overlap = len(q_words & s_words)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:max_sentences]]
        return [s for s in sentences if s in top]

    def _groundedness(self, retrieved_chunks: list[dict]) -> dict:
        if not retrieved_chunks:
            return {"label": "Not grounded", "top_score": 0.0}
        top_score = retrieved_chunks[0]["score"]
        if top_score < MIN_SCORE:
            label = "Not grounded"
        elif top_score < STRONG_SCORE:
            label = "Weakly grounded"
        else:
            label = "Strongly grounded"
        return {"label": label, "top_score": top_score}

    def generate_answer(self, question: str, retrieved_chunks: list[dict]) -> dict:
        groundedness = self._groundedness(retrieved_chunks)

        if groundedness["label"] == "Not grounded":
            return {
                "text": "I don't know — that isn't covered in the course material I have access to.",
                "used_sentences": {},
                "groundedness": groundedness,
            }

        used_sentences = {}
        for c in retrieved_chunks[:2]:
            used_sentences[c["chunk_index"], c["doc_id"]] = self._best_sentences(question, c["text"])

        prompt = self.build_prompt(question, retrieved_chunks)
        response = self.llm_client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=600,
            ),
        )
        answer_text = (response.text or "").strip()

        return {
            "text": answer_text,
            "used_sentences": used_sentences,
            "groundedness": groundedness,
        }

    def highlight_source(self, chunk: dict, used_sentences: dict) -> str:
        key = (chunk["chunk_index"], chunk["doc_id"])
        sentences_to_highlight = used_sentences.get(key, [])
        text = chunk["text"]
        for s in sentences_to_highlight:
            if s in text:
                text = text.replace(s, f"[[{s}]]")
        return text

    def rag_answer(self, question: str, k: int = TOP_K) -> dict:
        retrieved_chunks = self.retrieve(question, k=k)
        answer = self.generate_answer(question, retrieved_chunks)
        return {
            "question": question,
            "answer": answer["text"],
            "groundedness": answer["groundedness"],
            "used_sentences": answer["used_sentences"],
            "sources": retrieved_chunks,
        }


def run_chatbot():
    assistant = InstiAssist()
    print("IITB Insti-Assist — Academic Assistant (IIT Bombay Academic Office)")
    print("Ask a question about the academic calendar, grading, registration, medical rules, or medals. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = assistant.rag_answer(question)
        print(f"\nAssistant [{result['groundedness']['label']}]: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            highlighted = assistant.highlight_source(s, result["used_sentences"])
            print(f"  - [{s['score']:.3f}] {s['title']} (chunk {s['chunk_index']})")
            if s.get("source_url"):
                print(f"    Source: {s['source_url']}")
            print(f"    {highlighted[:300]}")
        print()


if __name__ == "__main__":
    run_chatbot()
