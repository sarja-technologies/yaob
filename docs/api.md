# MPN Studio API

Base URL: `http://<host>:8840`. JSON in, JSON out. All ids are decimal **strings** (never JavaScript numbers). Errors are `{"error": "sentence", "detail": {...}}` with a 4xx/5xx status. Interactive reference: `/api/docs`.

Access is by network location (`MPN_ALLOWED_NETS`); there is no login. Cross-site browser writes are refused.

Every mutating call returns `rev`, the model revision it created. Geometry calls (`/api/layout/*`, `/api/board/*`) never create a revision.

## Status

| Call | Returns |
|---|---|
| `GET /api/health` | `{ok, version, head, time}` |
| `GET /api/status` | counts (canonical live nodes, by type/origin, memberships, edges, tombstones), dirty summary (`to_index`, `trained_stale`, `untrained`, `trained_current`), runtime status or `{available:false, reason}`, memory, budget, `undo`/`redo` targets, active jobs |

## Board (geometry only)

| Call | Body / query | Returns |
|---|---|---|
| `GET /api/board` | | `{bounds:[minx,miny,maxx,maxy], spacing, mpns:[region], bundles:[{a,b,edges,shared}], saved_views, objects, view, head}` |
| `POST /api/viewport` | `{bbox:[minx,miny,maxx,maxy], scale (px per world unit), cap?, labels?, edges?, edge_cap?}` | `{level:"nodes"|"cells", total_in_view, nodes?|cells?, regions, edges?, head, ms}` |
| `PUT /api/board/view` | `{x, y, zoom, panels}` | saves the last view and panel arrangement |
| `POST /api/board/views` / `DELETE /api/board/views/{id}` | `{name, x, y, zoom}` | named saved views |
| `POST /api/layout/move` | `{moves:[{id, x, y}]}` | `{moved}`; pins the nodes |
| `POST /api/layout/recompute` | | rebuilds the hierarchical layout (pinned nodes stay) |
| `POST/PATCH/DELETE /api/board/objects[/{id}]` | `{kind:"group"|"note"|"panel", x, y, w, h, props}` | board-only objects |

A viewport **node**: `{id, x, y, t (type), s (status), v (validation), o (origin), m (primary mpn id), nm (membership count), deg (in-degree), pin, learned:"trained"|"stale"|"none", indexed, title?}`. `title` is present only for the nodes that should be labelled at this zoom. A **cell**: `{x, y, n, m}` (an aggregate of `n` nodes). A **region**: `{kind:"mpn"|"domain", id, x, y, r, name, count, ...}`. An **edge**: `{id, s, d, t, w}`.

The server never returns more than `cap` nodes or cells. When the viewport holds more nodes than the cap, or nodes would be under 3 px apart, it returns cells.

## Nodes

| Call | Notes |
|---|---|
| `GET /api/nodes/{id}?rev=` | full record: cells, memberships, sources, first 40 edges, `state` (index/learned status with a plain explanation), schema roles, `problems`, `position`, `cost` |
| `POST /api/nodes` | `{title, summary?, type?, aliases?, cells?, memberships:[{mpn_id, domain_id?, role?}], sources?:[{title,url,license,...}], position?}`. Returns `effects`: what is live now, after indexing, after training |
| `PATCH /api/nodes/{id}` | any of `title, summary, aliases, type, cells, status ("live"|"disabled"), validation, schema`. 422 with `detail.problems` when cells do not match the schema |
| `POST /api/nodes/validate` | `{cells, schema?}` → `{problems}` without saving |
| `GET /api/nodes/{id}/delete-preview` | memberships, edge count, learned rows that deletion affects |
| `DELETE /api/nodes/{id}` | recoverable: the node becomes a tombstone |
| `POST /api/nodes/{id}/restore` | |
| `POST /api/nodes/{id}/duplicate` | `{mode:"canonical"|"reference", position?}`. `canonical` makes a new node with a `derived_from` edge; `reference` adds a second visual occurrence of the same node |
| `GET /api/nodes/{id}/edges?offset=&limit=` | paginated neighbours |
| `GET /api/nodes/{id}/history` | versions with revision, actor, message |
| `PUT /api/nodes/{id}/memberships/{mpn_id}` | `{role?, domain_id?}` |
| `DELETE /api/nodes/{id}/memberships/{mpn_id}` | removes one membership; the node and its other memberships stay. 409 if it is the only one |
| `POST /api/edges` | `{src, dst, type}` |
| `DELETE /api/edges/{id}` | |
| `GET /api/edge-types` | built-in and in-use relation types |

## Search

`GET /api/search?q=&mode=hybrid|lexical|semantic&limit=&offset=&type=&status=&validation=&origin=&mpn_id=`

Returns `{results:[{id, title, type, status, validation, summary, score, match:"text"|"meaning"|"text+meaning"|"id", snippet?, mpn, mpn_id, x, y}], ms, semantic_available, notice?}`. `x, y` let the board jump to a node that is not loaded. A numeric query or a `yaob:` uid looks the node up directly. When the runtime is down, hybrid falls back to text and says so in `notice`.

## MPNs

| Call | Notes |
|---|---|
| `GET /api/mpns` | each: `installed, enabled, resident, resident_bytes, training ("untrained"|"partly trained"|"trained"), validated, node_count, shared_count, primary_nodes, trained_nodes, manifest` |
| `POST /api/mpns` | `{name, slug?, description?}` |
| `GET /api/mpns/{id}` | with its domain tree |
| `GET /api/mpns/{id}/impact?action=disable|enable|uninstall|install` | what the action would do, in numbers and one explanation |
| `POST /api/mpns/{id}/{action}` | performs it; uninstall tombstones nodes that belong only to this MPN and keeps shared nodes |

## Schemas

| Call | Notes |
|---|---|
| `GET /api/schemas` | all versions with their specs and node counts, plus `runtime_roles` and `field_types` |
| `POST /api/schemas/validate` | a spec → `{problems, diff, compared_with}`. `diff.changes[].effect` says in a sentence what each change does; `diff.breaking`, `migration_required`, `retrain_required` |
| `PUT /api/schemas` | saves a version. A version in use can only change labels and descriptions; a structural change must be saved as the next version |
| `GET /api/schemas/migration?from_id=&from_version=&to_id=&to_version=` | preview: node count, sample validation, whether it can apply |
| `POST /api/schemas/migrate` | `{from_id, from_version, to_id, to_version, node_ids?}` |

## Revisions

`GET /api/revisions`, `POST /api/undo`, `POST /api/redo`, `POST /api/releases {name, rev?, note?}`. Undo and redo are themselves revisions; history is never rewritten.

## Jobs

`GET /api/jobs`, `POST /api/jobs {kind, params}`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/log?offset=` (negative offset = from the end; returns `{text, next, size}` for tailing), `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/resume`. States: `queued, running, cancelling, cancelled, failed, succeeded, interrupted`. `progress.waiting` explains why a queued job has not started. `GET /api/checkpoints`, `POST /api/checkpoints/{id}/activate`.

## Budget and runtime

`GET /api/budget` → `{budget, presets:{conservative,balanced,expanded}, memory}`; `PUT /api/budget` with any budget field, or `{preset, apply_preset:true}`.

`GET /api/runtime` → `{available, core:{name, repo, revision, license, params_total, loaded}, encoder, checkpoint, learned, residency:{mpns:[{mpn_id, name, resident, bytes, last_used}], cache_bytes, cache_limit_bytes}, requests:{active, queued}, memory:{process_rss_mib, torch_allocated_mib, torch_peak_mib}, index:{rows, pending}}` or `{available:false, reason}`.

`POST /api/runtime/{load|unload|reindex|evict}`.

## Run (chat) — server-sent events

`POST /api/run` with `{messages:[{role, content}], mode:"yaob"|"retrieval"|"core", rev?, request_id?}` streams events:

| Event | Data |
|---|---|
| `start` | `{request_id, rev, mode, core:{name, revision}, checkpoint}` |
| `route` | `{ms, mpns:[{id, name, score, active}], skipped:[{id, name, reason}], shards:[{mpn_id, name, state:"loaded now"|"already resident", bytes, ms}], adapters:[{mpn_id, name, bytes, rank, trainable_params, validated, trained_at_rev}], nodes:[{id, title, mpn_id, mpn, score, gate, read_as:["text"|"specialist"], state:"trained"|"edited since training"|"not trained"|"text", via:"router"|"text match"|"referral"|"retrieval"}], procedures:[{node_id, title, impl, args, ok, result, unit, error}], budget:{applied:{...}, reductions:[sentence]}}` |
| `token` | `{text}` |
| `validation` | `{citations:[{n, node_id, title, source:{title, url, license, revision}}], checks:[{node_id, title, kind, ok, detail}]}` |
| `done` | `{text, cancelled, usage:{prompt_tokens, text_evidence_tokens, completion_tokens, adapters_applied}, timing:{route_ms, load_ms, ttft_ms, total_ms, tokens_per_s}, memory:{process_rss_mib, torch_allocated_mib, torch_peak_mib, estimate_mib, resident_cache_bytes, system_available_gib}}` |
| `error` | `{error, status}` |

`mode`: `yaob` = learned routing, the selected MPNs' specialist adapters and node text (`options: {router: "learned"|"baseline", adapters: bool, text: bool}` for ablations); `read_as: "specialist"` means the node's content was in the training set of an adapter applied to this request, at its current content; `retrieval` = ordinary text retrieval only; `core` = the unmodified core with no node access. `POST /api/run/{request_id}/cancel` stops generation and frees the request's memory.

The routing evidence lists what was actually consulted. It is not a record of hidden reasoning, and the UI must not present it as one.

## OpenAI-compatible endpoint

`GET /v1/models`, `POST /v1/chat/completions` (model `yaob-n10k`, `stream` supported). Yaob extensions: request field `yaob: {mode, rev, options}`; the final chunk (or the response) carries `yaob: {rev, route, validation, timing, memory}` with the same shapes as above.

Token budget (`mpnstudio/runtime/budget.py`), one rule per generation call: exact chat-template input tokens + max new tokens + 16 ≤ effective context, where effective context = min(budget `context_tokens`, the core's `max_context`). Defaults: context 8,192, answer cap (`output_tokens`) 1,024. `max_tokens` / `max_completion_tokens` in a request is an upper bound honoured up to the cap; it is never a reservation and never replaces the cap. When input does not fit, the runtime drops node text beyond the best node, then the oldest turns, then shortens the answer allowance (not below 256), then drops the last node's text, and reports each step in `route.budget.reductions`; `route.budget.tokens` carries the effective context, output, requested output, margin and an input breakdown (`total` exact, parts approximate). If the question and instructions alone do not fit, the response is HTTP 413 with `error.breakdown` and `error.limits`; the message contains "(N tokens) exceeds the available context size (M tokens)", which OpenAI-style chat clients parse. Refusals before generation are HTTP errors; a failure after text has started is appended as `[Yaob stopped: …]`. A client `system` message is passed to the core after Yaob's own instructions and is never cut.

`GET /v1/models` `meta`: `n_ctx` (effective context), `max_output_tokens` (answer cap), `max_active_adapters`, `n_ctx_train` (the core's configured maximum).

Specialists: `max_active_adapters` defaults to 0 since eval-001. A request may set it with `budget: {"max_active_adapters": 1}` (the "specialists (experimental)" preset in the Budget panel does the same for every request).
