"""Part C: support-ticket triage with guardrails (redaction, retries, validation, budget, routing)."""
import json
import os
import random
import re
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (mock_llm.py)
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ('You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ('You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. '
            'Reply with JSON only in the form {"category": "...", "reason": "..."}.\n'
            'EXAMPLES:\n'
            'Ticket: Someone hacked my account and used my wallet, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}\n'
            'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The app freezes and shows an error at checkout. -> {"category": "technical", "reason": "app fault"}\n'
            'Ticket: The rider has not arrived and my order is late. -> {"category": "delivery", "reason": "late order"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT


def redact(text, counts=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counts is not None:
            counts[name] += n
    return text


def call_with_retry(messages, ticket_id, log, stats):
    for retry in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient"] += 1
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            log.append((ticket_id, retry, wait))
            print(f"retry log: ticket={ticket_id} retry={retry} wait={wait:.4f}s")
            time.sleep(wait)


def validate(content):
    try:
        cat = json.loads(content).get("category")
        if cat in CATEGORIES:
            return "classified_json", cat
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    found = [w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATEGORIES]
    if len(set(found)) == 1:
        return "classified_keyword", found[0]
    return "flagged_invalid_output", None


def run_pipeline(tickets, system_prompt, token_budget=None, verbose=False):
    mock_llm.reset()
    random.seed(7)
    log, stats = [], {"transient": 0}
    sent_prompts, total_tokens, rows = [], 0, []
    for t in tickets.itertuples():
        text = t.text
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": None,
               "category": "", "tokens_used": 0, "true_category": t.true_category}
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            safe = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": safe}]
            sent_prompts.append(safe)
            resp = call_with_retry(messages, t.ticket_id, log, stats)
            used = resp.usage.total_tokens
            total_tokens += used
            row["tokens_used"] = used
            row["status"], cat = validate(resp.choices[0].message.content)
            row["category"] = cat or ""
        rows.append(row)
    return pd.DataFrame(rows), {"total_tokens": total_tokens, "prompts": sent_prompts,
                                "retry_log": log, "transient": stats["transient"]}


def route(df, seg_path="outputs/customer_segments.csv"):
    seg = pd.read_csv(seg_path).set_index("customer_id")
    out = []
    for r in df.itertuples():
        if r.customer_id in seg.index:
            segname, prem = seg.loc[r.customer_id, "segment_name"], bool(seg.loc[r.customer_id, "is_premium"])
        else:
            segname, prem = "", False
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            rt, pr = "human_review", "NORMAL"
        elif r.category == "fraud":
            rt, pr = "security_team", "HIGH"
        elif prem:
            rt, pr = "premium_desk", "HIGH"
        else:
            rt, pr = "support_queue", "NORMAL"
        out.append({"ticket_id": r.ticket_id, "customer_id": r.customer_id, "status": r.status,
                    "category": r.category, "segment_name": segname, "priority": pr, "route": rt,
                    "tokens_used": r.tokens_used})
    return pd.DataFrame(out)


def summarize(name, df, info):
    cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (cls["category"] == cls["true_category"]).mean() * 100
    print(f"\n--- {name} ---")
    print("status counts:", dict(Counter(df["status"])))
    print(f"classified {len(cls)} | accuracy {acc:.1f}% ({int((cls['category']==cls['true_category']).sum())}/{len(cls)})"
          f" | total tokens {info['total_tokens']} | tokens/classified ticket {info['total_tokens']/len(cls):.1f}")
    print("transient errors:", info["transient"], "| retry log lines:", len(info["retry_log"]))
    return cls


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("tickets:", len(tickets))

    # guardrails
    empty = int((tickets["text"].str.strip() == "").sum())
    long_ = int(tickets["text"].map(est_tokens).gt(MAX_INPUT_TOKENS).sum())
    print("empty:", empty, "long:", long_, "long est tokens:", sorted(set(tickets["text"].map(est_tokens)))[-1])

    # redaction
    counts = Counter()
    red = tickets["text"].map(lambda s: redact(s, counts))
    print("substitutions:", dict(counts), "total", sum(counts.values()))
    left = sum(len(p.findall(s)) for s in red for _, p, _ in PATTERNS)
    print("patterns found in redacted texts:", left)
    assert left == 0

    for p in (ZERO_SHOT, FEW_SHOT):
        print("prompt est tokens:", est_tokens(p))

    # zero-shot and few-shot full runs
    zs, zi = run_pipeline(tickets, ZERO_SHOT)
    zcls = summarize("zero-shot", zs, zi)
    fs, fi = run_pipeline(tickets, FEW_SHOT)
    summarize("few-shot", fs, fi)
    for pr in zi["prompts"] + fi["prompts"]:        # privacy assertion on every prompt sent
        assert not any(p.search(pr) for _, p, _ in PATTERNS)
    waits = [(r, w) for _, r, w in zi["retry_log"] + fi["retry_log"]]
    assert all(BASE_DELAY * 2 ** r * 0.9 <= w <= BASE_DELAY * 2 ** r * 1.15 for r, w in waits)
    print("assertions passed: no PII in any prompt; every retry wait within jitter range:",
          f"{min(w for _, w in waits):.4f}-{max(w for _, w in waits):.4f}s")

    print("\nzero-shot confusion (rows=true, cols=predicted):")
    print(pd.crosstab(zcls["true_category"], zcls["category"]).to_string())

    # budget demo (zero-shot)
    budget = 6000
    bd, bi = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
    exhausted = int((bd["status"] == "budget_exhausted").sum())
    print(f"\nbudget demo: TOKEN_BUDGET={budget} -> budget_exhausted = {exhausted}, tokens used {bi['total_tokens']}")
    assert exhausted >= 1
    route(bd).to_csv("outputs/triage_budget_demo.csv", index=False)

    # final run
    fr, ri = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    res = route(fr)
    res.to_csv("outputs/triage_results.csv", index=False)
    print("\ntriage_results rows:", len(res))
    print(res.groupby(["route", "priority"]).size().to_string())
    print("status counts:", dict(Counter(res["status"])))
