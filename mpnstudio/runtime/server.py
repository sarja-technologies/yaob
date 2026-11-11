"""Runtime process: holds the core, the encoder and the selectively loaded MPN state.

Listens on loopback only and is never exposed to the network. Its clients are local: the API server (which applies
the access rules) and, through the OpenAI-compatible `/v1` routes, other services on the same machine such as a chat app.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from ..store.db import Store
from ..train import jobs
from ..train.gpulock import GpuLock
from .engine import Engine

log = logging.getLogger("mpnstudio.runtime")


V1_MODELS = {"yaob-n10k": "Yaob N10K: learned routing over 10,000 nodes, node text and procedures, on Qwen3-4B-Instruct-2507 "
                          "(specialist adapters off by default; budget max_active_adapters)",
             "yaob-n10k-retrieval": "Yaob N10K nodes through ordinary retrieval (no learned routing or adapters), same core"}
V1_MODE = {"yaob-n10k": "yaob", "yaob-n10k-retrieval": "retrieval"}


def text_of(content) -> str:
    """OpenAI message content is a string or a list of parts; only text parts are used (the core reads text only)."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return content if isinstance(content, str) else ""


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def request_log(cid: str, model_id: str, mode: str, ext: dict, done: dict) -> None:
    """One line per answered chat request: resolved limits, measured tokens, what was routed and applied. No message text."""
    route = ext.get("route") or {}
    tok = (route.get("budget") or {}).get("tokens") or {}
    log.info("chat %s", json.dumps({
        "id": cid, "model": model_id, "mode": mode, "rev": ext.get("rev"),
        "context": tok.get("context"), "output": tok.get("output"), "output_requested": tok.get("output_requested"),
        "input_tokens": (tok.get("input") or {}).get("total"), "completion_tokens": (done.get("usage") or {}).get("completion_tokens"),
        "adapters_applied": (done.get("usage") or {}).get("adapters_applied"),
        "mpns": [m.get("name") for m in route.get("mpns", []) if m.get("active")],
        "nodes_text": [n["id"] for n in route.get("nodes", []) if "text" in n.get("read_as", [])],
        "nodes_selected": len(route.get("nodes", [])), "reductions": len((route.get("budget") or {}).get("reductions") or []),
        "total_ms": (done.get("timing") or {}).get("total_ms"), "torch_peak_mib": (done.get("memory") or {}).get("torch_peak_mib")}))


def create_app() -> FastAPI:
    store = Store(jobs.db_path())
    root = jobs.data_root()
    models = Path(os.environ.get("MPN_MODELS", Path.home() / "mpn-studio" / "models"))
    engine = Engine(store, root, models)
    lock = GpuLock()
    app = FastAPI(title="Yaob runtime", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.engine = engine
    state = {"loading": False, "error": None}

    def load_all():
        state["loading"] = True
        try:
            lock.acquire()          # waits behind an exclusive media run; then shares the GPU
            lock.release()
            engine.load_core()
            try:
                engine.activate()
            except Exception as e:
                state["error"] = f"checkpoint not activated: {e}"
            try:
                while engine.reindex(2000).get("pending"):
                    pass
            except Exception as e:
                state["error"] = f"indexing failed: {e}"
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"
        finally:
            state["loading"] = False

    if os.environ.get("MPN_RUNTIME_AUTOLOAD", "1") == "1":
        threading.Thread(target=load_all, daemon=True).start()

    def background_index():
        while True:
            time.sleep(5)
            try:
                if engine.enc is not None and not state["loading"]:
                    engine.reindex(500)
            except Exception:
                pass
    threading.Thread(target=background_index, daemon=True).start()

    @app.get("/status")
    async def status():
        st = engine.status()
        st["loading"] = state["loading"]
        st["startup_error"] = state["error"]
        st["gpu_lock"] = {"enabled": lock.enabled, "exclusive_waiter": lock.exclusive_waiter()}
        return st

    @app.get("/residency")
    async def residency():
        return {"mpns": engine.residency()}

    @app.post("/semantic")
    async def semantic(body: dict):
        if engine.enc is None:
            return JSONResponse({"error": "the encoder is not loaded"}, status_code=503)
        res = await asyncio.to_thread(engine.semantic, body.get("q", ""), int(body.get("limit", 20)), body.get("filters") or {})
        return {"results": res}

    @app.post("/budget")
    async def budget(body: dict):
        return engine.set_budget(body)

    @app.post("/load")
    async def load(body: dict | None = None):
        if state["loading"]:
            return {"loading": True}
        if engine.core is not None and not (body or {}).get("core"):
            return {"loaded": True, "already": True}
        if (body or {}).get("core"):
            store.set_setting("core", body["core"])
            engine.unload_core()
        threading.Thread(target=load_all, daemon=True).start()
        return {"loading": True}

    @app.post("/unload")
    async def unload():
        if engine.requests:
            return JSONResponse({"error": f"{len(engine.requests)} request(s) are running. Stop them first; unloading would cut them off."}, status_code=409)
        before = engine.status()["memory"]
        out = await asyncio.to_thread(engine.unload_core)
        return dict(out, memory_before=before, memory_after=engine.status()["memory"])

    @app.post("/reindex")
    async def reindex():
        if engine.enc is None:
            return JSONResponse({"error": "the encoder is not loaded"}, status_code=503)
        return await asyncio.to_thread(engine.reindex, 5000)

    @app.post("/evict")
    async def evict(body: dict | None = None):
        mid = (body or {}).get("mpn_id")
        if engine.cache is None:
            return {"evicted": []}
        return {"evicted": [str(m) for m in engine.cache.drop(int(mid) if mid else None)], "cache_bytes": engine.cache.bytes()}

    @app.post("/invalidate")
    async def invalidate(body: dict | None = None):
        engine.flat_index = None
        return {"ok": True}

    @app.post("/activate")
    async def activate(body: dict):
        if engine.requests:
            return JSONResponse({"error": "Requests are running. A checkpoint is switched only between requests."}, status_code=409)
        try:
            with engine.gen_lock:
                return await asyncio.to_thread(engine.activate, body.get("checkpoint"))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/cancel/{request_id}")
    async def cancel(request_id: str):
        return {"cancelled": engine.cancel(request_id)}

    def stream(messages, mode, rev, request_id, budget, options=None):
        gen = engine.run(messages, mode, rev, request_id, budget, options)
        rid = {"id": None}

        async def agen():
            loop = asyncio.get_running_loop()
            it = iter(gen)
            try:
                while True:
                    item = await loop.run_in_executor(None, lambda: next(it, None))
                    if item is None:
                        break
                    ev, data = item
                    if ev == "start":
                        rid["id"] = data["request_id"]
                    yield ev, data
            finally:
                if rid["id"]:
                    engine.cancel(rid["id"])        # client went away: stop generating and release
                gen.close()
        return agen()

    @app.post("/run")
    async def run(body: dict):
        mode = body.get("mode", "yaob")
        if mode not in ("yaob", "retrieval", "core"):
            return JSONResponse({"error": "mode must be yaob, retrieval or core"}, status_code=400)

        async def out():
            async for ev, data in stream(body.get("messages", []), mode, body.get("rev"), body.get("request_id"), body.get("budget"), body.get("options")):
                yield sse(ev, data)
        return StreamingResponse(out(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def v1_models():
        """The ways to ask the model, as OpenAI-style model ids. `yaob-n10k` is the full path (learned routing,
        node text, procedures, and specialist adapters up to the budget's max_active_adapters, 0 by default); `yaob-n10k-retrieval` is ordinary retrieval over the same
        nodes with the same core, for comparison."""
        st = engine.status()
        core = st.get("core") or {}
        ck = st.get("checkpoint")
        lim = st.get("limits") or {}
        # n_ctx is the context one request actually gets here (what clients should fit to); n_ctx_train is the core's.
        common = {"object": "model", "created": 1794355200, "owned_by": "fylume",
                  "meta": {"n_ctx": lim.get("context_tokens"), "n_ctx_train": core.get("max_context"), "max_output_tokens": lim.get("output_tokens"),
                           "n_params": core.get("params_total"), "core": core.get("name"), "max_active_adapters": lim.get("max_active_adapters")}}
        return {"object": "list", "data": [
            dict(common, id=mid, description=desc, status={"loaded": bool(core.get("loaded")), "checkpoint": ck.get("id") if isinstance(ck, dict) else ck})
            for mid, desc in V1_MODELS.items()]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        y = body.get("yaob") or {}
        mode = y.get("mode") or V1_MODE.get(str(body.get("model") or ""), "yaob")
        body["messages"] = [dict(m, content=text_of(m.get("content"))) for m in body.get("messages", []) if m.get("role") in ("system", "user", "assistant")]
        budget = dict(body.get("budget") or {})
        # max_tokens is the client's upper bound on generated tokens. It is honoured up to the runtime's answer cap
        # (budget output_tokens); it is not a reservation and never replaces the cap.
        want = body.get("max_completion_tokens") if body.get("max_completion_tokens") is not None else body.get("max_tokens")
        if want is not None:
            budget["requested_output_tokens"] = want
        cid = "chatcmpl-" + os.urandom(6).hex()
        created = int(time.time())
        model_id = body.get("model") if body.get("model") in V1_MODELS else "yaob-n10k"
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id}
        agen = stream(body.get("messages", []), mode, y.get("rev"), None, budget, y.get("options"))
        if body.get("stream"):
            # Refusals (no checkpoint, bad request, budget) arrive before generation starts. Send them as a real HTTP
            # error, not as an error chunk in a 200 stream: OpenAI-style clients ignore those and show an empty reply.
            # Everything before the route event (start, budget fitting) produces no text, so a refusal there can still be
            # an HTTP error. Chat clients then show it, and learn the context size from a 413.
            head = []
            async for item in agen:
                head.append(item)
                if item[0] in ("route", "token", "done", "error"):
                    break
            if not head:
                return JSONResponse({"error": {"message": "The runtime produced no response.", "type": "yaob_error"}}, status_code=500)
            if head[-1][0] == "error":
                await agen.aclose()
                d = head[-1][1]
                err = {"message": d["error"], "type": "yaob_error", "code": d.get("status")}
                if d.get("breakdown"):
                    err.update(breakdown=d["breakdown"], limits=d.get("limits"))
                log.warning("refused before generation: status %s %s", d.get("status"), json.dumps({k: d.get(k) for k in ("breakdown", "limits")}))
                return JSONResponse({"error": err}, status_code=int(d.get("status") or 500))

            async def replay():
                for item in head:
                    yield item
                async for item in agen:
                    yield item

            async def out():
                ext: dict = {}
                async for ev, data in replay():
                    if ev == "start":
                        ext["rev"] = data["rev"]
                        yield f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]))}\n\n".encode()
                    elif ev == "token":
                        yield f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {'content': data['text']}, 'finish_reason': None}]))}\n\n".encode()
                    elif ev == "route":
                        ext["route"] = data
                    elif ev == "validation":
                        ext["validation"] = data
                    elif ev == "done":
                        ext.update(timing=data["timing"], memory=data["memory"])
                        request_log(cid, model_id, mode, ext, data)
                        u = data["usage"]
                        usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                                 "total_tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)}
                        yield f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], usage=usage, yaob=ext))}\n\n".encode()
                    elif ev == "error":
                        # Failed after text had started: say so in the reply itself, which every client shows.
                        note = f"\n\n[Yaob stopped: {data['error']}]"
                        yield f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {'content': note}, 'finish_reason': None}]))}\n\n".encode()
                        yield f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], yaob=dict(ext, error=data)))}\n\n".encode()
                yield b"data: [DONE]\n\n"
            return StreamingResponse(out(), media_type="text/event-stream")
        text, ext, usage, err = "", {}, {}, None
        async for ev, data in agen:
            if ev == "start":
                ext["rev"] = data["rev"]
            elif ev == "route":
                ext["route"] = data
            elif ev == "validation":
                ext["validation"] = data
            elif ev == "done":
                text = data["text"]
                ext.update(timing=data["timing"], memory=data["memory"])
                request_log(cid, model_id, mode, ext, data)
                u = data["usage"]
                usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                         "total_tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)}
            elif ev == "error":
                err = data
        if err:
            e = {"message": err["error"], "type": "yaob_error", "code": err.get("status")}
            if err.get("breakdown"):
                e.update(breakdown=err["breakdown"], limits=err.get("limits"))
            return JSONResponse({"error": e}, status_code=int(err.get("status", 500)))
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage, "yaob": ext}

    return app


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("MPN_RUNTIME_PORT", "8841")), log_level="info", access_log=False)


if __name__ == "__main__":
    main()
