"""Part C: support-ticket triage with guardrails, redaction, retries, output validation, budget and routing."""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS = 400
BASE_DELAY, MAX_RETRIES = 0.05, 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
RETRY_LOG = "outputs/retry_log.txt"

ZERO_SHOT = ('You are a support ticket router. CLASSIFY the ticket into exactly one category: billing, delivery, '
             'technical or fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ('You are a support ticket router. CLASSIFY the ticket into exactly one category: billing, delivery, '
            'technical or fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.\n'
            'EXAMPLES:\n'
            'Ticket: Someone used my card without my consent, this is an unauthorized transaction. -> {"category": "fraud", "reason": "unauthorized payment"}\n'
            'Ticket: I was charged twice and the bill amount is wrong. -> {"category": "billing", "reason": "wrong charge"}\n'
            'Ticket: The app freezes and shows an error at login. -> {"category": "technical", "reason": "app fault"}\n'
            'Ticket: My rider has not arrived and the order is late. -> {"category": "delivery", "reason": "late order"}')

def redact(text, counts=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counts is not None:
            counts[name] = counts.get(name, 0) + n
    return text

def assert_no_pii(messages):
    for m in messages:
        for _, pat, _ in PATTERNS:
            assert not pat.search(m["content"]), f"PII pattern leaked into prompt: {m['content'][:60]}"

def call_with_retry(messages, ticket_id, log):
    assert_no_pii(messages)
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            log.append((ticket_id, attempt, wait))
            print(f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s", file=log_file)
            time.sleep(wait)

def validate(text):
    """returns (status, category)"""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower())) & set(CATEGORIES)
        return ("classified_keyword", words.pop()) if len(words) == 1 else ("flagged_invalid_output", "")
    cat = obj.get("category") if isinstance(obj, dict) else None
    if isinstance(cat, str) and cat in CATEGORIES:
        return "classified_json", cat
    return "flagged_invalid_output", ""

def run_pipeline(system_prompt, token_budget=None, label=""):
    random.seed(7)  # only affects retry jitter
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    mock_llm.reset()
    print(f"=== run: {label} ===", file=log_file)
    log, total, rows = [], 0, []
    for t in tickets.itertuples():
        text = t.text
        row = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, true_category=t.true_category,
                   status=None, category="", tokens_used=0)
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            red = redact(text)
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red}]
            assert msgs[1]["content"] == red
            resp = call_with_retry(msgs, t.ticket_id, log)
            used = resp.usage.total_tokens
            total += used; row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), total, log

def add_routing(df):
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    df = df.copy()
    df["segment_name"] = df.customer_id.map(seg.segment_name).fillna("")
    prem = df.customer_id.map(seg.is_premium).fillna(False).astype(bool)
    human = df.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    route, prio = [], []
    for h, c, p in zip(human, df.category, prem):
        if h: route.append("human_review"); prio.append("NORMAL")
        elif c == "fraud": route.append("security_team"); prio.append("HIGH")
        elif p: route.append("premium_desk"); prio.append("HIGH")
        else: route.append("support_queue"); prio.append("NORMAL")
    df["route"], df["priority"] = route, prio
    return df

def summarise(df, total, name):
    classified = df[df.status.isin(["classified_json", "classified_keyword"])]
    acc = (classified.category == classified.true_category).mean()
    print(f"\n[{name}] status counts: {df.status.value_counts().to_dict()}")
    print(f"[{name}] classified={len(classified)} accuracy={acc*100:.1f}% ({int((classified.category==classified.true_category).sum())}/{len(classified)}) "
          f"total_tokens={total} tokens/classified={total/len(classified):.2f}")
    return classified

if __name__ == "__main__":
    log_file = open(RETRY_LOG, "w")
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    # ---- guardrail and redaction report
    empty = (tickets.text.str.strip() == "").sum()
    long_ = tickets.text.map(lambda s: est_tokens(s) > MAX_INPUT_TOKENS).sum()
    print(f"empty tickets={empty}; long tickets={long_}; long est_tokens={sorted(set(est_tokens(s) for s in tickets.text if est_tokens(s) > MAX_INPUT_TOKENS))}")
    counts = {}
    red = tickets.text.map(lambda s: redact(s, counts))
    print("redaction substitutions:", counts, "total", sum(counts.values()))
    left = sum(len(p.findall(s)) for s in red for _, p, _ in PATTERNS)
    print("PII patterns left after redaction:", left); assert left == 0
    print("zero-shot prompt tokens:", est_tokens(ZERO_SHOT), " few-shot prompt tokens:", est_tokens(FEW_SHOT))
    assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

    zs, zs_tok, zlog = run_pipeline(ZERO_SHOT, label="zero-shot full")
    print(f"transient errors recovered (zero-shot): {len(zlog)}")
    summarise(zs, zs_tok, "zero-shot")
    cl = zs[zs.status.isin(["classified_json", "classified_keyword"])]
    print("zero-shot confusion table (rows=true, cols=predicted):")
    print(pd.crosstab(cl.true_category, cl.category).to_string())
    fs, fs_tok, flog = run_pipeline(FEW_SHOT, label="few-shot full")
    print(f"transient errors recovered (few-shot): {len(flog)}")
    summarise(fs, fs_tok, "few-shot")
    waits = [w for _, _, w in zlog + flog]
    print(f"retry waits: n={len(waits)} min={min(waits):.4f} max={max(waits):.4f} retry numbers={sorted(set(r for _, r, _ in zlog + flog))}")
    assert all(0.045 <= w <= 0.0575 for w in waits)

    # ---- budget demo (zero-shot)
    BUDGET = 6000
    bd, bd_tok, _ = run_pipeline(ZERO_SHOT, token_budget=BUDGET, label=f"zero-shot budget {BUDGET}")
    add_routing(bd).drop(columns="true_category")[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"\nBudget demo TOKEN_BUDGET={BUDGET}: tokens used={bd_tok}, budget_exhausted tickets={(bd.status=='budget_exhausted').sum()}; status counts {bd.status.value_counts().to_dict()}")

    # ---- final results (few-shot, budget 100000)
    fin, fin_tok, _ = run_pipeline(FEW_SHOT, token_budget=100000, label="few-shot final")
    fin = add_routing(fin)
    fin[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv("outputs/triage_results.csv", index=False)
    print(f"\nFinal triage_results.csv rows={len(fin)} tokens={fin_tok}")
    print("route counts:", fin.route.value_counts().to_dict())
    print("priority counts:", fin.priority.value_counts().to_dict())
    print("route x status:"); print(pd.crosstab(fin.status, fin.route).to_string())
    log_file.close()
    print(f"retry log lines written: {sum(1 for l in open(RETRY_LOG) if l.startswith('retry'))} -> {RETRY_LOG}")
