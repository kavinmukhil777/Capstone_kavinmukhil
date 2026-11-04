"""Part D tasks 5-8: tools, ReAct loop, safe stopping, checkpoint/resume."""
import json, os, re, sqlite3, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import mock_llm
from policy_rag import retrieve

DB, LEDGER, STATE_DIR, RUNS = "data/instamart.db", "outputs/refund_ledger.jsonl", "state", "outputs/agent_runs.jsonl"
LATE_MINUTES, AUTO_LIMIT = 45, 1000

# ------------------------------------------------------------------ tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    try:
        r = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()
    if r is None:
        return {"error": "order not found"}
    return {"order_id": r[0], "status": r[1], "order_amount": r[2], "delivery_minutes": r[3]}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 3), "text": h["text"]} for h in retrieve(query, 3)]}

def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > LATE_MINUTES
    return {"refund_inr": round(float(order_amount), 2) if late else 0}

def _ledger(tool, order_id, refund_inr):
    """Append one JSON line. A second issue for the same order is recorded but flagged, never paid twice."""
    dup = False
    if tool == "issue_refund" and os.path.exists(LEDGER):
        with open(LEDGER) as f:
            dup = any((j := json.loads(l)).get("tool") == tool and j.get("order_id") == order_id and not j.get("duplicate_suppressed") for l in f if l.strip())
    line = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
            "order_id": order_id, "refund_inr": refund_inr}
    if dup:
        line["duplicate_suppressed"] = True
    with open(LEDGER, "a") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props, req):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.", {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three most relevant passages.", {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR: full order amount if delivered in more than 45 minutes, otherwise 0.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

# ------------------------------------------------------------------ agent loop
SYSTEM = ("You are a refund AGENT for a quick-commerce support desk. Use the tools to look up the order, check the policy, "
          "calculate the refund, then issue it or escalate it. Finish with STATUS: explanation.")

class SimulatedCrash(RuntimeError):
    pass

def _save(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w") as f:
        json.dump(state, f, indent=1)

def _fallback(state, status, reason):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    state["final_answer"] = (f"{status}: handing over to a human agent. Reason: {reason}. "
                             f"Scenario {state['scenario_id']}, steps completed: {done}. Request: {state['user_message']}")
    state["status"] = status
    return state

def _loop(state, crash_after_tools=None):
    cfg = state["config"]; start = time.monotonic()
    while state["llm_calls"] < cfg["max_iterations"]:
        if time.monotonic() - start >= cfg["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {cfg['time_limit_seconds']}s reached")
        resp = mock_llm.chat(state["messages"], TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > cfg["token_limit"]:       # stop before executing the requested tool
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit {cfg['token_limit']} exceeded ({state['tokens_used']})")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            state["final_answer"] = msg.content
            state["status"] = msg.content.split(":", 1)[0].strip()
            return state
        tc = msg.tool_calls[0]
        call = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [call]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        state["steps"].append({"step": len(state["steps"]) + 1, "tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        _save(state)                                         # checkpoint after every observation
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no final answer after {cfg['max_iterations']} model calls")

def _result(state):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": state["status"],
            "final_answer": state["final_answer"], "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]}

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    run_id = run_id or f"{scenario_id}-{int(time.time()*1000)}"
    state = {"run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [], "status": None, "final_answer": None,
             "config": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _result(_loop(state, crash_after_tools))

def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json")) as f:
        state = json.load(f)
    return _result(_loop(state))

# ------------------------------------------------------------------ batch + demos
def refund_amount(res):
    for s in res["steps"]:
        if s["tool"] == "calculate_refund":
            return s["observation"]["refund_inr"]
    return 0

def log_run(res, variant):
    rec = {"scenario_id": res["scenario_id"], "variant": variant, "status": res["status"], "final_answer": res["final_answer"],
           "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
           "steps": [{"tool": s["tool"], "arguments": s["arguments"], "observation": s["observation"]} for s in res["steps"]]}
    with open(RUNS, "a") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    for p in (RUNS, LEDGER):
        if os.path.exists(p): os.remove(p)
    if os.path.isdir(STATE_DIR):
        for fn in os.listdir(STATE_DIR):
            if fn.endswith(".json"): os.remove(os.path.join(STATE_DIR, fn))
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("Scenario | status | refund_inr | expected | model calls | tokens")
    for r in sc.itertuples():
        mock_llm.reset()
        res = run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}-default")
        log_run(res, "default")
        amt = refund_amount(res)
        exp_status = "FALLBACK_MAX_ITERATIONS" if r.expected_outcome == "FALLBACK" else r.expected_outcome
        assert res["status"] == exp_status, (r.scenario_id, res["status"])
        if r.expected_outcome in ("REFUND", "ESCALATE"):
            assert abs(amt - r.expected_refund_inr) < 0.01
        print(f"{r.scenario_id} | {res['status']} | {amt:.2f} | {r.expected_refund_inr:.2f} | {res['llm_calls']} | {res['tokens_used']}")

    s1 = sc.iloc[0]
    print("\n-- Demo: S1 with token_limit=200")
    d = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit"); log_run(d, "token_limit=200")
    print(f"status={d['status']} llm_calls={d['llm_calls']} tokens={d['tokens_used']} steps_completed={[s['tool'] for s in d['steps']]}")
    print("-- Demo: S1 with time_limit_seconds=0")
    d = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout"); log_run(d, "time_limit_seconds=0")
    print(f"status={d['status']} llm_calls={d['llm_calls']} tokens={d['tokens_used']}")

    print("-- Demo: S1 crash after the 2nd tool execution, then resume")
    mock_llm.reset()
    try:
        run_agent(s1.message, "S1", run_id="S1-crash-resume", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crash:", e)
    ck = json.load(open("state/S1-crash-resume.json"))
    print("checkpoint tools executed so far:", ck["tools_executed"])
    res = resume_agent("S1-crash-resume")
    final = json.load(open("state/S1-crash-resume.json"))
    print("tools executed across crash + resume:", final["tools_executed"])
    print(f"final status={res['status']} refund={refund_amount(res):.2f} llm_calls={res['llm_calls']} tokens={res['tokens_used']}")
    assert final["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(refund_amount(res) - 128.25) < 0.01
    log_run(res, "crash_and_resume")
    print("\nledger lines:"); print(open(LEDGER).read())
