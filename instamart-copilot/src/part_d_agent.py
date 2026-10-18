"""Part D (tasks 5-8): tools, ReAct loop, safe stopping, checkpoint/resume, scenario runs and demos."""
import glob
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (mock_llm.py)
import mock_llm
from part_d_rag import retrieve

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
RUNS = os.path.join("outputs", "agent_runs.jsonl")
DEMOS = os.path.join("outputs", "agent_demos.jsonl")
STATE_DIR = "state"
AGENT_PROMPT = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, check the "
                "policy, calculate the refund and then issue it or escalate it. Finish with one line that starts "
                "with REFUND, ESCALATE, DENIED or NOT_FOUND followed by a colon.")
_CURRENT_RUN = {"id": None}


# ------------------------------------------------------------------ tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]}
                        for h in retrieve(query, k=3)]}


def calculate_refund(order_amount, delivery_minutes):
    """Completed orders delivered in more than 45 minutes get 100%; otherwise 0. (Status is checked by the caller.)"""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr, "run_id": _CURRENT_RUN["id"]}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the most relevant chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ loop
class SimulatedCrash(Exception):
    pass


def _checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, default=str)


def _fallback(state, status, reason):
    done = ", ".join(state["tools_executed"]) or "none"
    state["final_answer"] = (f"{status}: {reason}. Handing scenario {state['scenario_id']} to a human agent. "
                             f"Tools already run: {done}.")
    state["status"] = status
    return state


def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    _CURRENT_RUN["id"] = state["run_id"]
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached")
        if state["llm_calls"] >= max_iterations:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit of {token_limit} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            state["final_answer"] = msg.content
            state["status"] = msg.content.split(":")[0].strip()
            return state
        tc = msg.tool_calls[0]
        args = dict(tc.arguments)
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        observation = TOOL_MAP[tc.name](**args)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(observation)})
        state["steps"].append({"tool": tc.name, "arguments": args, "observation": observation})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["tools_executed"])
        _checkpoint(state)
        if crash_after_tools is not None and state["step"] == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool execution #{state['step']} ({tc.name})")


def _result(state):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": state["status"],
            "final_answer": state["final_answer"], "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": state["tools_executed"]}


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}-run", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0,
             "status": None, "final_answer": None}
    state = _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)
    return _result(state)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    state = _loop(state, max_iterations, token_limit, time_limit_seconds)
    return _result(state)


def refund_amount(res):
    if res["status"] in ("REFUND", "ESCALATE"):
        return res["steps"][-1]["observation"]["refund_inr"]
    return 0.0


if __name__ == "__main__":
    for p in glob.glob(os.path.join(STATE_DIR, "*.json")) + [LEDGER, RUNS, DEMOS]:
        if os.path.exists(p):
            os.remove(p)
    os.makedirs("outputs", exist_ok=True)
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    for r in sc.itertuples():
        res = run_agent(r.message, r.scenario_id)
        amt = refund_amount(res)
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls",
                                                    "tokens_used", "steps")}) + "\n")
        exp = r.expected_outcome
        ok = (res["status"].startswith("FALLBACK") if exp == "FALLBACK" else res["status"] == exp) \
            and abs(amt - float(r.expected_refund_inr)) <= 0.01
        rows.append({"scenario": r.scenario_id, "status": res["status"], "refund_inr": f"{amt:.2f}",
                     "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "expected": exp,
                     "ok": ok})
    print(pd.DataFrame(rows).to_string(index=False))
    assert all(x["ok"] for x in rows)
    assert rows[0]["llm_calls"] == 5 and rows[6]["status"] == "FALLBACK_MAX_ITERATIONS" and rows[6]["llm_calls"] == 6

    s1 = sc.iloc[0]
    demos = []
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-demo")
    print("\nS1 token_limit=200 ->", d1["status"], "| tokens", d1["tokens_used"], "| tools run", d1["tools_executed"])
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    print("S1 time_limit_seconds=0 ->", d2["status"], "| llm_calls", d2["llm_calls"])
    demos += [d1, d2]

    try:
        run_agent(s1.message, "S1", run_id="S1-crash-demo", crash_after_tools=2)
        raise AssertionError("expected a simulated crash")
    except SimulatedCrash as e:
        print("\n", e)
    ck = json.load(open(os.path.join(STATE_DIR, "S1-crash-demo.json")))
    print("checkpoint keys:", sorted(ck), "| tools executed before crash:", ck["tools_executed"])
    d3 = resume_agent("S1-crash-demo")
    print("after resume ->", d3["status"], refund_amount(d3), "| tools executed across crash+resume:", d3["tools_executed"])
    assert d3["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert d3["status"] == "REFUND" and abs(refund_amount(d3) - 128.25) <= 0.01
    demos.append(d3)
    with open(DEMOS, "w", encoding="utf-8") as f:
        for d in demos:
            f.write(json.dumps({k: d[k] for k in ("run_id", "scenario_id", "status", "final_answer", "llm_calls",
                                                  "tokens_used", "tools_executed")}) + "\n")
    print("\nstate files:", sorted(os.listdir(STATE_DIR)))
    print("ledger lines:", sum(1 for _ in open(LEDGER)))
