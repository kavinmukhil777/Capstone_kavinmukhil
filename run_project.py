"""Convenience runner: executes every step of the README's Reproduce section (after `pip install -r requirements.txt`).
Usage: python run_project.py"""
import os, subprocess, sys

STEPS = ["generate_data.py", "src/part_a_clean.py", "src/validate.py", "src/part_a_analysis.py",
         "src/part_b_segments.py", "src/part_b_lapse.py", "src/part_c_triage.py",
         "src/policy_rag.py", "src/refund_agent.py"]
os.makedirs("outputs/reports", exist_ok=True); os.makedirs("state", exist_ok=True)
for s in STEPS:
    print(f"\n>>> python {s}", flush=True)
    out = subprocess.run([sys.executable, s], capture_output=True, text=True)
    sys.stdout.write(out.stdout); sys.stderr.write(out.stderr)
    if out.returncode:
        sys.exit(f"step failed: {s}")
    if s != "generate_data.py":
        open(f"outputs/reports/{os.path.basename(s)[:-3]}.txt", "w").write(out.stdout)
print("\nAll steps finished.")
