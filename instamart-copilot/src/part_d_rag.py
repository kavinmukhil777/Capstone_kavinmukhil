"""Part D (tasks 1-4): chunking, TF-IDF retrieval, grounded answers, retrieval/grounding evaluation."""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (mock_llm.py)
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for Instamart support staff. Answer ONLY from the supplied CONTEXT. "
                 "If the context does not contain the answer, say that you could not find the information "
                 "in the policy documents.")


def chunk_words(words, size=40, overlap=10):
    """Windows of `size` words, each starting size-overlap words after the previous; stop after the
    first window whose end reaches or passes the last word."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


class PolicyIndex:
    def __init__(self, size=40, overlap=10, policy_dir=POLICY_DIR):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(policy_dir, "*.txt"))):
            words = open(path, encoding="utf-8").read().split()
            for text in chunk_words(words, size, overlap):
                self.chunks.append({"source": os.path.basename(path), "text": text})
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]}
                for i in order]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [h["source"] for h in hits]


INDEX = PolicyIndex()


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


def answer(question):
    return INDEX.answer(question)


def evaluate(index, label):
    ev = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"), keep_default_na=False)
    doc_ok = fact_ok = fallback_ok = n_ans = n_unans = 0
    failures = []
    for r in ev.itertuples():
        text, sources = index.answer(r.question)
        if r.expected_doc == "NONE":
            n_unans += 1
            fallback_ok += text.startswith("I could not find")
        else:
            n_ans += 1
            d, f = r.expected_doc in sources, r.expected_fact in text
            doc_ok += d
            fact_ok += f
            if not (d and f):
                failures.append((r.question, r.expected_doc, r.expected_fact, d, f, text))
    print(f"\n[{label}] chunks={len(index.chunks)}")
    print(f"  expected_doc in top-3: {doc_ok}/{n_ans} | answer contains expected_fact: {fact_ok}/{n_ans} "
          f"| fallback on unanswerable: {fallback_ok}/{n_unans}")
    for q, doc, fact, d, f, text in failures:
        print(f"  FAIL: {q!r} doc_ok={d} fact_ok={f}\n        expected {fact!r}; answer: {text!r}")
    return failures


if __name__ == "__main__":
    counts = pd.Series([c["source"] for c in INDEX.chunks]).value_counts().sort_index()
    print("total chunks:", len(INDEX.chunks))
    print(counts.to_string())
    top = retrieve("refund policy late delivery")
    print("\nquery 'refund policy late delivery' top-3:", [(h["source"], round(h["score"], 3)) for h in top])
    q = "What delivery time makes a completed order eligible for a full refund?"
    print("\nsample answer:", answer(q))
    evaluate(INDEX, "overlap 10")
    evaluate(PolicyIndex(40, 0), "overlap 0")
