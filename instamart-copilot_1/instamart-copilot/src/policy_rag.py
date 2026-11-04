"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering and evaluation."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")

def chunk_words(text, size=40, overlap=10):
    words = text.split()
    step, out, start = size - overlap, [], 0
    while True:
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):       # this chunk reaches/passes the last word
            return out
        start += step

def load_chunks(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        txt = open(path, encoding="utf-8").read()
        for c in chunk_words(txt, size, overlap):
            chunks.append({"source": os.path.basename(path), "text": c})
    return chunks

class PolicyIndex:
    def __init__(self, size=40, overlap=10):
        self.chunks = load_chunks(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        top = sims.argsort()[::-1][:k]          # stable enough: ties keep ascending index order after reverse
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]

_DEFAULT = None
def retrieve(query, k=3, index=None):
    global _DEFAULT
    if index is None:
        if _DEFAULT is None:
            _DEFAULT = PolicyIndex(40, 10)
        index = _DEFAULT
    return index.retrieve(query, k)

SYSTEM = ("You are a POLICY ASSISTANT for support staff. Answer only from the supplied context. "
          "If the context does not contain the answer, say that you could not find the information.")

def answer(question, index=None):
    hits = retrieve(question, 3, index)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

def evaluate(size=40, overlap=10, verbose=True):
    idx = PolicyIndex(size, overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = 0; ans_n = unans_n = 0; failures = []
    for r in ev.itertuples():
        text, sources = answer(r.question, idx)
        if r.expected_doc == "NONE":
            unans_n += 1; fallback += text.startswith("I could not find")
            if not text.startswith("I could not find"): failures.append((r.question, "should have fallen back", text))
        else:
            ans_n += 1
            d = r.expected_doc in sources; f = r.expected_fact in text
            doc_hit += d; fact_hit += f
            if not (d and f): failures.append((r.question, f"doc_in_top3={d} fact_in_answer={f}", text))
    print(f"chunks={len(idx.chunks)} (size={size}, overlap={overlap}); per-file: "
          f"{pd.Series([c['source'] for c in idx.chunks]).value_counts().sort_index().to_dict()}")
    print(f"answerable: expected_doc in top-3 {doc_hit}/{ans_n}; expected_fact in answer {fact_hit}/{ans_n}; unanswerable fallback {fallback}/{unans_n}")
    for q, why, t in failures:
        print(f"  FAIL: {q!r} -> {why}\n        answer: {t}")
    return idx

if __name__ == "__main__":
    print("== overlap 10 =="); idx = evaluate(40, 10)
    print("\nretrieve('refund policy late delivery'):")
    for h in idx.retrieve("refund policy late delivery"):
        print(f"  {h['source']} score={h['score']:.3f}")
    print("\n== overlap 0 =="); evaluate(40, 0)
