<script lang="ts" module>
  export interface Evidence {
    start?: any;
    route?: any;
    validation?: any;
    done?: any;
  }
  export interface Msg {
    role: 'user' | 'assistant';
    content: string;
    error?: string;
    streaming?: boolean;
    stopped?: boolean;
    ev?: Evidence;
  }
  // kept at module level so the conversation survives docking, floating and tab switches
  const R = $state({ msgs: [] as Msg[], mode: 'yaob' as 'yaob' | 'retrieval' | 'core', input: '', busy: false, requestId: null as string | null });
  let abort: AbortController | null = null;
</script>

<script lang="ts">
  import { app } from '../state.svelte';
  import { post, fmtInt, fmtBytes, human, plural } from '../api';
  import Icon from '../ui/Icon.svelte';

  let list = $state<HTMLDivElement | null>(null);
  const runtimeDown = $derived(app.status ? !app.status.runtime.available : false);

  function scrollEnd() {
    queueMicrotask(() => list?.scrollTo({ top: list.scrollHeight }));
  }
  async function send() {
    const text = R.input.trim();
    if (!text || R.busy) return;
    R.input = '';
    R.msgs.push({ role: 'user', content: text });
    R.msgs.push({ role: 'assistant', content: '', streaming: true, ev: {} });
    const m = R.msgs[R.msgs.length - 1];
    R.busy = true;
    // crypto.randomUUID needs a secure context, which a LAN address over http is not
    R.requestId = 'r' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
    abort = new AbortController();
    scrollEnd();
    const history = R.msgs.filter((x) => x !== m && !x.error && x.content).map((x) => ({ role: x.role, content: x.content }));
    try {
      const res = await fetch('api/run', {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
        body: JSON.stringify({ messages: history, mode: R.mode, request_id: R.requestId }),
        signal: abort.signal,
      });
      if (!res.ok || !res.body) {
        let msg = `The server answered ${res.status}.`;
        try {
          msg = JSON.parse(await res.text()).error ?? msg;
        } catch {
          /* keep the status sentence */
        }
        m.error = msg;
      } else {
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
          let i: number;
          while ((i = buf.indexOf('\n\n')) >= 0) {
            handle(m, buf.slice(0, i));
            buf = buf.slice(i + 2);
          }
        }
        if (buf.trim()) handle(m, buf);
      }
    } catch (e: any) {
      if (e?.name === 'AbortError') m.stopped = true;
      else m.error = 'The connection to the server was lost before the answer finished.';
    }
    m.streaming = false;
    R.busy = false;
    R.requestId = null;
    abort = null;
    scrollEnd();
  }
  function handle(m: Msg, block: string) {
    let event = 'message';
    const data: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return;
    let d: any;
    try {
      d = JSON.parse(data.join('\n'));
    } catch {
      return;
    }
    const ev = (m.ev ??= {});
    if (event === 'start') {
      ev.start = d;
      if (d.request_id) R.requestId = d.request_id;
    } else if (event === 'route') {
      ev.route = d;
      app.highlight = (d.nodes ?? []).map((n: any) => String(n.id));
    } else if (event === 'token') {
      m.content += d.text ?? '';
      scrollEnd();
    } else if (event === 'validation') ev.validation = d;
    else if (event === 'done') {
      ev.done = d;
      if (typeof d.text === 'string' && d.text.length >= m.content.length) m.content = d.text;
      if (d.cancelled) m.stopped = true;
    } else if (event === 'error') m.error = d.error ?? 'The run failed.';
  }
  async function stop() {
    const id = R.requestId;
    // ask the runtime to stop generating and free the request's memory, then drop the stream
    if (id) await post(`api/run/${id}/cancel`).catch(() => {});
    abort?.abort();
  }
  function clear() {
    R.msgs = [];
    app.highlight = [];
  }
  function key(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  }
  const ms = (v: any) => (typeof v === 'number' ? `${Math.round(v)} ms` : '');
  const isUrl = (s: any) => typeof s === 'string' && /^https?:\/\//i.test(s);
  function summary(ev: Evidence): string {
    const r = ev.route;
    const parts: string[] = [];
    if (r?.mpns) parts.push(plural(r.mpns.filter((x: any) => x.active !== false).length, 'MPN'));
    if (r?.nodes) parts.push(plural(r.nodes.length, 'node'));
    if (r?.procedures?.length) parts.push(plural(r.procedures.length, 'procedure'));
    if (ev.validation?.citations?.length) parts.push(plural(ev.validation.citations.length, 'citation'));
    return parts.join(', ');
  }
  const hasEvidence = (ev?: Evidence) => !!ev && (ev.route || ev.validation || ev.done);
</script>

<div class="run">
  <div class="msgs scroll" bind:this={list}>
    {#if !R.msgs.length}
      <div class="empty">
        <p class="lead">Ask the model</p>
        <p class="muted">After each answer you can see which MPNs and nodes were consulted.</p>
        {#if runtimeDown}<p class="problem">The runtime is not running, so nothing can be generated right now.</p>{/if}
      </div>
    {/if}
    {#each R.msgs as m, i (i)}
      <div class="msg {m.role}">
        {#if m.role === 'user'}
          <p class="who muted">You</p>
          <p class="text">{m.content}</p>
        {:else}
          <p class="who muted">{app.modelName}{m.ev?.start ? ` at revision ${m.ev.start.rev}` : ''}</p>
          {#if m.content}<p class="text">{m.content}</p>{/if}
          {#if m.streaming && !m.content}<p class="muted">Waiting for the runtime</p>{/if}
          {#if m.stopped}<p class="muted">Stopped.</p>{/if}
          {#if m.error}<p class="error-line">{m.error}</p>{/if}
          {#if hasEvidence(m.ev)}
            {@const ev = m.ev!}
            <details class="ev">
              <summary>What was consulted <span class="muted">{summary(ev)}</span></summary>
              <div class="evbody">
                {#if ev.route?.mpns?.length}
                  <div class="grp">
                    <h4>MPNs consulted</h4>
                    <ul>
                      {#each ev.route.mpns as x}<li><span>{x.name}</span>{" "}<span class="muted num">score {Number(x.score).toFixed(2)}{x.active === false ? ', not used' : ''}</span></li>{/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.route?.skipped?.length}
                  <div class="grp">
                    <h4>MPNs skipped</h4>
                    <ul>
                      {#each ev.route.skipped as x}<li><span>{x.name}</span>{' '}<span class="muted">{x.reason}</span></li>{/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.route?.shards?.length}
                  <div class="grp">
                    <h4>Learned state loaded</h4>
                    <ul>
                      {#each ev.route.shards as x}<li><span>{x.name}</span>{" "}<span class="muted num">{x.state}, {fmtBytes(x.bytes)}{x.ms != null ? ', ' + ms(x.ms) : ''}</span></li>{/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.route?.adapters?.length}
                  <div class="grp">
                    <h4>Specialists applied</h4>
                    <ul>
                      {#each ev.route.adapters as a}<li><span>{a.name}</span>{" "}<span class="muted num">rank {a.rank}, {fmtInt(a.trainable_params)} parameters, {fmtBytes(a.bytes)}{a.validated === false ? ', did not beat the core on its held-out check' : ''}</span></li>{/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.route?.nodes?.length}
                  <div class="grp">
                    <h4>Nodes read</h4>
                    <ul>
                      {#each ev.route.nodes as x}
                        <li>
                          <button class="linkbtn" title="Show this node on the board" onclick={() => app.goToNode(String(x.id))}>{x.title}</button>
                          {" "}<span class="muted">{x.mpn}{x.read_as?.length ? ', read as ' + x.read_as.join(' and ') : ', routed only'}, via {x.via}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.route?.procedures?.length}
                  <div class="grp">
                    <h4>Procedures run</h4>
                    <ul>
                      {#each ev.route.procedures as p}
                        <li>
                          <button class="linkbtn" title="Show this node on the board" onclick={() => app.goToNode(String(p.node_id))}>{p.title}</button>
                          <div class="mono wrapany">{p.impl}({JSON.stringify(p.args ?? {})}) {p.ok ? '= ' + JSON.stringify(p.result) + (p.unit ? ' ' + p.unit : '') : 'failed: ' + p.error}</div>
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.route?.budget?.reductions?.length}
                  <div class="grp">
                    <h4>Reduced to fit the memory budget</h4>
                    <ul>{#each ev.route.budget.reductions as s}<li>{s}</li>{/each}</ul>
                  </div>
                {/if}
                {#if ev.validation?.citations?.length}
                  <div class="grp">
                    <h4>Citations</h4>
                    <ul>
                      {#each ev.validation.citations as c}
                        <li>
                          <span class="num">[{c.n}]</span>
                          <button class="linkbtn" onclick={() => app.goToNode(String(c.node_id))}>{c.title}</button>
                          {#if c.source}
                            <div class="muted">
                              {#if isUrl(c.source.url)}<a href={c.source.url} target="_blank" rel="noreferrer">{c.source.title}</a>{:else}{c.source.title}{/if}{c.source.license ? ', licence ' + c.source.license : ''}{#if c.source.revision}, revision <span class="mono">{c.source.revision}</span>{/if}
                            </div>
                          {/if}
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.validation?.checks?.length}
                  <div class="grp">
                    <h4>Checks</h4>
                    <ul>
                      {#each ev.validation.checks as c}<li><span>{c.ok ? 'Passed' : 'Failed'}: {c.title}</span>{" "}<span class="muted">{human(c.kind)}{c.detail ? ', ' + c.detail : ''}</span></li>{/each}
                    </ul>
                  </div>
                {/if}
                {#if ev.done}
                  {@const t = ev.done.timing ?? {}}
                  {@const u = ev.done.usage ?? {}}
                  {@const mem = ev.done.memory ?? {}}
                  <div class="grp">
                    <h4>Time and memory</h4>
                    <dl>
                      <dt>Time</dt>
                      <dd class="num">route {ms(t.route_ms)}, load {ms(t.load_ms)}, first token {ms(t.ttft_ms)}, total {ms(t.total_ms)}{t.tokens_per_s != null ? `, ${Number(t.tokens_per_s).toFixed(1)} tokens/s` : ''}</dd>
                      <dt>Tokens</dt>
                      <dd class="num">prompt {fmtInt(u.prompt_tokens)}, text evidence {fmtInt(u.text_evidence_tokens)}, answer {fmtInt(u.completion_tokens)}</dd>
                      <dt>Memory</dt>
                      <dd class="num">process {fmtInt(mem.process_rss_mib)} MiB, tensors {fmtInt(mem.torch_allocated_mib)} MiB (peak {fmtInt(mem.torch_peak_mib)} MiB), estimate {fmtInt(mem.estimate_mib)} MiB</dd>
                    </dl>
                  </div>
                {/if}
                {#if ev.start?.core}
                  <p class="muted">Core <span class="mono">{ev.start.core.name}</span> at <span class="mono">{ev.start.core.revision}</span>{ev.start.checkpoint ? ', checkpoint ' : ''}{#if ev.start.checkpoint}<span class="mono">{typeof ev.start.checkpoint === 'string' ? ev.start.checkpoint : (ev.start.checkpoint.id ?? '')}</span>{/if}</p>
                {/if}
              </div>
            </details>
          {/if}
        {/if}
      </div>
    {/each}
  </div>

  <form class="composer" onsubmit={(e) => (e.preventDefault(), send())}>
    <textarea class="field" rows="2" placeholder="Message" aria-label="Message" bind:value={R.input} onkeydown={key}></textarea>
    <div class="row">
      <select class="field mode" aria-label="Run mode" bind:value={R.mode} disabled={R.busy} title="Yaob: learned routing, the selected MPNs' specialists and node text. Retrieval only: node text found by ordinary retrieval. Core only: the unmodified core.">
        <option value="yaob">Yaob</option>
        <option value="retrieval">Retrieval only</option>
        <option value="core">Core only</option>
      </select>
      {#if R.msgs.length && !R.busy}<button type="button" class="btn" onclick={clear}>Clear</button>{/if}
      <span class="grow"></span>
      {#if R.busy}
        <button type="button" class="btn" onclick={stop}><Icon name="stop" size={12} />Stop</button>
      {:else}
        <button type="submit" class="btn primary" disabled={!R.input.trim()}>Send</button>
      {/if}
    </div>
  </form>
</div>

<style>
  .run {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .msgs {
    flex: 1;
    min-height: 0;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .empty {
    margin: auto;
    text-align: center;
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 300px;
  }
  .empty .problem {
    text-align: left;
  }
  .lead {
    font-size: 15px;
    font-weight: 500;
  }
  .who {
    font-size: 12px;
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 14px;
    line-height: 20px;
  }
  .msg {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .ev {
    margin-top: 4px;
  }
  .ev summary {
    cursor: pointer;
    color: var(--white);
  }
  .evbody {
    padding: 6px 0 2px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  h4 {
    margin: 0;
    font-weight: 500;
    font-size: 13px;
  }
  .grp ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .grp li {
    padding: 1px 0;
    overflow-wrap: anywhere;
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 1px 10px;
    margin: 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
  }
  .wrapany {
    overflow-wrap: anywhere;
    color: var(--muted);
  }
  .linkbtn {
    border: 0;
    background: transparent;
    padding: 0;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: #5a5a5a;
    text-underline-offset: 2px;
    cursor: pointer;
  }
  .composer {
    flex: none;
    padding: 8px 12px 10px;
    border-top: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .composer textarea {
    resize: none;
    font-size: 14px;
    line-height: 20px;
  }
  .mode {
    width: auto;
  }
</style>
