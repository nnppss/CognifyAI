import re
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    HYBRID_ALPHA,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_TOPK,
    NEIGHBOR_WINDOW,
)
from indexing import load_index_and_segments
from llm_client import call_llm

# Optional lexical scorer
try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_BM25 = False


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _tok(text: str):
    return re.findall(r"\w+", (text or "").lower())


class LectureQA:
    """
    Industry-style RAG engine for a single lecture:

    - loads embeddings and transcript segments
    - hybrid retrieval: cosine (semantic) + BM25 (lexical)
    - builds a rich prompt and calls Gemini through llm_client.call_llm()
    """

    def __init__(self, lecture_id: str):
        self.lecture_id = lecture_id
        self.emb, self.segments = load_index_and_segments(lecture_id)

        # L2-normalize embeddings (safety if not already normalized)
        self.emb = self.emb.astype("float32")
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-10
        self.emb = self.emb / norms

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        if _HAS_BM25:
            corpus_tokens = [_tok(s["text"]) for s in self.segments]
            self.bm25 = BM25Okapi(corpus_tokens)
        else:
            self.bm25 = None

    # -------- Retrieval --------

    def _hybrid_search(self, question: str, pool_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Hybrid score = HYBRID_ALPHA * cosine + (1 - HYBRID_ALPHA) * normalized BM25.
        Returns (indices, hybrid_scores, cos_scores, bm25_scores).
        """
        # Semantic
        qv = self.embedder.encode([question], convert_to_numpy=True)[0].astype("float32")
        qv = qv / (np.linalg.norm(qv) + 1e-10)
        cos_scores = self.emb @ qv
        cos_n = _normalize(cos_scores)

        # Lexical
        if self.bm25 is not None:
            bm_scores = np.asarray(self.bm25.get_scores(_tok(question)), dtype=np.float32)
            bm_n = _normalize(bm_scores)
        else:
            bm_scores = np.zeros_like(cos_scores)
            bm_n = np.zeros_like(cos_scores)

        hybrid = HYBRID_ALPHA * cos_n + (1.0 - HYBRID_ALPHA) * bm_n

        k = min(pool_k, hybrid.shape[0])
        idx = np.argpartition(-hybrid, k - 1)[:k]
        idx = idx[np.argsort(-hybrid[idx])]

        return idx, hybrid[idx], cos_scores[idx], bm_scores[idx]

    def retrieve_segments(self, question: str, top_k: int = RETRIEVAL_TOPK, pool_k: int = RETRIEVAL_CANDIDATES) -> List[Dict]:
        """
        Retrieve top_k segments using hybrid search.
        """
        idx, hybrid, cos_vals, bm_vals = self._hybrid_search(question, pool_k=pool_k)
        out: List[Dict] = []
        for rank, (i, h, c, b) in enumerate(zip(idx, hybrid, cos_vals, bm_vals), start=1):
            seg = self.segments[int(i)]
            out.append(
                {
                    "rank": rank,
                    "i": int(i),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg["text"],
                    "hybrid": float(h),
                    "cosine": float(c),
                    "bm25": float(b),
                }
            )
        return out[:top_k]

    def _expand_neighbors(self, core_segments: List[Dict]) -> List[Dict]:
        """
        Include neighboring chunks around each selected core segment so
        the LLM sees better context. NEIGHBOR_WINDOW controls how many.
        """
        picked = {}
        for c in core_segments:
            center = c["i"]
            for j in range(center - NEIGHBOR_WINDOW, center + NEIGHBOR_WINDOW + 1):
                if 0 <= j < len(self.segments):
                    seg = self.segments[j]
                    if j not in picked:
                        picked[j] = {
                            "i": j,
                            "start": float(seg["start"]),
                            "end": float(seg["end"]),
                            "text": seg["text"],
                        }
        ordered = [picked[k] for k in sorted(picked.keys(), key=lambda ix: picked[ix]["start"])]
        return ordered

    # -------- Prompt building --------

    def _build_context_block(self, segs: List[Dict]) -> str:
        lines = []
        for idx, s in enumerate(segs, start=1):
            lines.append(f"[{idx}] {s['start']:.1f}–{s['end']:.1f}s :: {s['text']}")
        return "\n".join(lines)

    def _build_prompts(self, question: str, support_segs: List[Dict]) -> (str, str):
        context = self._build_context_block(support_segs)

        system_prompt = (
            "You are an expert teaching assistant for B.Tech students. "
            "You help explain material from a lecture video. "
            "You are given relevant transcript segments with timestamps. "
            "Use those segments as the PRIMARY source of truth. "
            "You may also use your own knowledge of the subject to clarify concepts, "
            "give examples, and connect ideas, but do not contradict the transcript."
        )

        user_prompt = (
            f"Student question:\n{question}\n\n"
            f"Relevant transcript segments from the lecture:\n"
            f"{context}\n\n"
            "Instructions:\n"
            "- First, answer the question clearly in your own words.\n"
            "- Base your explanation primarily on the transcript text above.\n"
            "- You MAY add extra details from your general knowledge, but if you go beyond the video, "
            "make sure it is standard textbook knowledge.\n"
            "- Use simple language and, if helpful, bullet points.\n"
            "- At the end, add a short line: 'Recommended segments to rewatch: [list the segment numbers]'."
        )

        return system_prompt, user_prompt

    # -------- Public API --------

    def answer_question(self, question: str, top_k: int = RETRIEVAL_TOPK) -> Dict:
        retrieved_core = self.retrieve_segments(
            question, top_k=top_k, pool_k=RETRIEVAL_CANDIDATES
        )
        if not retrieved_core:
            return {
                "answer": "I couldn't find any relevant part in this lecture.",
                "score": 0.0,
                "timestamp": 0.0,
                "segments": [],
                "mode": "none",
            }

        windowed = self._expand_neighbors(retrieved_core)
        earliest_start = min(s["start"] for s in windowed)

        try:
            system_prompt, user_prompt = self._build_prompts(question, windowed)
            answer_text = call_llm(system_prompt, user_prompt).strip()
            mode = "gemini-rag"
        except Exception as e:
            # fallback: show best transcript chunk only
            answer_text = (
                f"(Gemini error: {e})\n\n"
                f"Best matching transcript snippet:\n\n{windowed[0]['text']}"
            )
            mode = "retrieval-only"

        # For UI segments, use the core retrieved ones (with scores)
        return {
            "answer": answer_text,
            "score": float(retrieved_core[0]["hybrid"]),
            "timestamp": float(earliest_start),
            "segments": retrieved_core,
            "mode": mode,
        }
