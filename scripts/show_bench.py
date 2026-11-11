"""Print a compact view of core benchmark reports: python scripts/show_bench.py runs/core-bench/*.json"""
import json
import sys

for path in [a for a in sys.argv[1:] if not a.startswith('-')]:
    d = json.load(open(path))
    print("=" * 8, d["model_dir"], d.get("architecture"))
    for k in ("cold_load_s", "params_total", "params_text_stack", "after_load", "speed", "suite_peak", "context_cost", "injection",
              "concurrent_gpu_load_at_start", "concurrent_gpu_load_at_end"):
        print(" ", k, d.get(k))
    q = d["quality"]
    print("  quality", {c: f"{v['ok']}/{v['n']}" for c, v in q["by_category"].items()}, f"total {q['total_ok']}/{q['total']}")
    if "-v" in sys.argv:
        continue
    for r in d["rows"]:
        if not r["ok"]:
            print("   FAIL", r["id"], repr(r["reply"][:120]))
