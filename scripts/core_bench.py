"""Benchmark one candidate core on this machine.

Same prompts, system prompt, greedy decoding, token limits and precision (bf16)
for every candidate. Records cold load, warm time to first token, throughput,
memory peaks, context cost, and whether the model supports the gradient and
embedding-injection path the modular architecture needs.

Usage: python scripts/core_bench.py --model models/Qwen3.5-4B --out runs/core-bench/qwen3.5-4b.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpnstudio.evals import core_suite  # noqa: E402


def mem_available_gib() -> float:
    return psutil.virtual_memory().available / 2**30


def proc_rss_gib() -> float:
    return psutil.Process().memory_info().rss / 2**30


def gpu_proc_mib() -> int | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == os.getpid():
                return int(p.usedGpuMemory / 2**20) if p.usedGpuMemory else None
    except Exception:
        return None
    return None


def other_gpu_load() -> list[dict]:
    out = []
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid != os.getpid():
                name = psutil.Process(p.pid).name() if psutil.pid_exists(p.pid) else "?"
                out.append({"name": name, "gpu_mib": int((p.usedGpuMemory or 0) / 2**20)})
        out.append({"gpu_util_percent": pynvml.nvmlDeviceGetUtilizationRates(h).gpu})
    except Exception as e:
        out.append({"error": str(e)})
    return out


def load(model_dir: str):
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    arch = (cfg.architectures or ["?"])[0]
    kw = dict(dtype=torch.bfloat16, device_map={"": "cuda"}, low_cpu_mem_usage=True)
    if "ConditionalGeneration" in arch or cfg.model_type in ("qwen3_5", "gemma4", "gemma3n", "gemma3"):
        from transformers import AutoModelForImageTextToText as Auto
        try:
            model = Auto.from_pretrained(model_dir, **kw)
        except Exception:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    model.eval()
    return cfg, tok, model


def chat_ids(tok, messages, tools=None):
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if tools:
        kw["tools"] = tools
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


def build_messages(task):
    return [{"role": "system", "content": core_suite.SYSTEM}, {"role": "user", "content": task["prompt"]}]


@torch.inference_mode()
def generate(model, tok, enc, max_new_tokens):
    from transformers import TextIteratorStreamer
    enc = {k: v.to(model.device) for k, v in enc.items()}
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    t0 = time.perf_counter()
    result = {}

    def run():
        result["out"] = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer,
                                       temperature=None, top_p=None, top_k=None)
    th = threading.Thread(target=run)
    th.start()
    first = None
    pieces = []
    for piece in streamer:
        if first is None and piece:
            first = time.perf_counter() - t0
        pieces.append(piece)
    th.join()
    total = time.perf_counter() - t0
    n_in = enc["input_ids"].shape[1]
    n_out = result["out"].shape[1] - n_in
    raw = tok.decode(result["out"][0, n_in:], skip_special_tokens=False)
    clean = tok.decode(result["out"][0, n_in:], skip_special_tokens=True)
    return {"raw": raw, "text": clean, "n_in": n_in, "n_out": int(n_out), "ttft_s": first, "total_s": total}


def injection_test(model, tok):
    """Does the core accept inputs_embeds, and does a gradient reach an injected soft prompt while the core is frozen?"""
    res = {}
    try:
        emb = model.get_input_embeddings()
        d = emb.embedding_dim
        res["hidden_size"] = d
        enc = chat_ids(tok, [{"role": "user", "content": "Name the colour of the sky."}])
        ids = enc["input_ids"].to(model.device)
        for p in model.parameters():
            p.requires_grad_(False)
        soft = torch.nn.Parameter(torch.randn(1, 4, d, device=model.device, dtype=torch.float32) * 0.02)
        tgt = tok(" Blue.", return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.cat([soft.to(torch.bfloat16), emb(ids), emb(tgt)], dim=1)
            labels = torch.cat([torch.full((1, 4 + ids.shape[1]), -100, device=model.device), tgt], dim=1)
            out = model(inputs_embeds=x, labels=labels)
        out.loss.backward()
        g = soft.grad
        res.update(inputs_embeds_forward=True, loss=float(out.loss), grad_norm=float(g.norm()),
                   grad_finite=bool(torch.isfinite(g).all()), grad_reaches_soft_prompt=bool(g.abs().sum() > 0))
        # generate() from embeddings only (the runtime path)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x2 = torch.cat([soft.detach().to(torch.bfloat16), emb(ids)], dim=1)
            gen = model.generate(inputs_embeds=x2, attention_mask=torch.ones(x2.shape[:2], device=model.device, dtype=torch.long),
                                 max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
        res["generate_from_embeds"] = True
        res["generate_sample"] = tok.decode(gen[0], skip_special_tokens=True)[:80]
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:400]
    finally:
        model.zero_grad(set_to_none=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    report = {"model_dir": args.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "host": platform.node(), "torch": torch.__version__, "precision": "bf16",
              "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "thinking": "disabled where the template supports it"},
              "concurrent_gpu_load_at_start": other_gpu_load(),
              "mem_available_before_gib": round(mem_available_gib(), 2)}
    import transformers
    report["transformers"] = transformers.__version__

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cfg, tok, model = load(args.model)
    torch.cuda.synchronize()
    report["cold_load_s"] = round(time.perf_counter() - t0, 2)
    report["architecture"] = (cfg.architectures or ["?"])[0]
    report["params_total"] = sum(p.numel() for p in model.parameters())
    report["params_text_stack"] = None
    lm = getattr(model, "language_model", None) or getattr(getattr(model, "model", None), "language_model", None)
    if lm is not None:
        report["params_text_stack"] = sum(p.numel() for p in lm.parameters())
    report["after_load"] = {"torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                            "process_rss_gib": round(proc_rss_gib(), 2), "nvml_process_mib": gpu_proc_mib(),
                            "mem_available_gib": round(mem_available_gib(), 2)}

    # Warm-up, then the suite.
    generate(model, tok, chat_ids(tok, [{"role": "user", "content": "Hello."}]), 8)
    tasks = core_suite.TASKS[: args.limit] if args.limit else core_suite.TASKS
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        enc = chat_ids(tok, build_messages(task), core_suite.TOOLS if task.get("tools") else None)
        g = generate(model, tok, enc, args.max_new_tokens)
        reply = g["raw"] if task.get("tools") else g["text"]
        ok = bool(core_suite.score(task, reply))
        rows.append({"id": task["id"], "cat": task["cat"], "ok": ok, "n_in": g["n_in"], "n_out": g["n_out"],
                     "ttft_s": round(g["ttft_s"] or 0, 4), "total_s": round(g["total_s"], 4), "reply": reply[:600]})
        print(f"{task['id']:16s} {'ok ' if ok else 'FAIL'} in={g['n_in']:4d} out={g['n_out']:3d} ttft={g['ttft_s'] or 0:.3f}s  {reply[:70]!r}", flush=True)

    cats = {}
    for r in rows:
        c = cats.setdefault(r["cat"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += r["ok"]
    gen_rows = [r for r in rows if r["n_out"] >= 16 and r["total_s"] > r["ttft_s"]]
    tps = [(r["n_out"] - 1) / (r["total_s"] - r["ttft_s"]) for r in gen_rows]
    ttfts = sorted(r["ttft_s"] for r in rows)
    report["quality"] = {"by_category": cats, "total_ok": sum(r["ok"] for r in rows), "total": len(rows)}
    report["speed"] = {"warm_ttft_median_s": ttfts[len(ttfts) // 2], "warm_ttft_p95_s": ttfts[int(len(ttfts) * 0.95) - 1],
                       "tokens_per_s_median": round(sorted(tps)[len(tps) // 2], 1) if tps else None,
                       "n_throughput_samples": len(tps)}
    report["suite_peak"] = {"torch_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                            "process_rss_gib": round(proc_rss_gib(), 2), "nvml_process_mib": gpu_proc_mib()}

    # Context cost: memory and latency for a long prompt.
    filler = "The node network stores knowledge in cells. " * 700
    enc = chat_ids(tok, [{"role": "user", "content": filler + "\nReply with the single word: done"}])
    n_ctx = enc["input_ids"].shape[1]
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    g = generate(model, tok, enc, 4)
    report["context_cost"] = {"prompt_tokens": int(n_ctx), "ttft_s": round(g["ttft_s"] or 0, 3),
                              "extra_peak_gib": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3),
                              "prompt_tokens_per_s": round(n_ctx / (g["ttft_s"] or 1), 0)}
    report["injection"] = injection_test(model, tok)
    report["concurrent_gpu_load_at_end"] = other_gpu_load()
    report["rows"] = rows
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("cold_load_s", "params_total", "after_load", "quality", "speed", "suite_peak", "context_cost", "injection")}, indent=1))


if __name__ == "__main__":
    main()
