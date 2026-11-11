// Inspector state lives outside the component so an unsaved draft survives docking, floating
// and switching tabs.
import { get, patch, errText, ApiError, type Effects, type NodeDetail, type NodeEdge } from '../api';
import { app } from '../state.svelte';

export interface Draft {
  title: string;
  summary: string;
  type: string;
  validation: string;
  cells: Record<string, Record<string, any>>;
}
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));
function draftOf(n: NodeDetail): Draft {
  const cells = clone(n.cells);
  // every cell of the schema gets an editable object; ones the node does not have yet are pruned again before saving
  for (const r of n.schema.roles) if (!cells[r.key] || typeof cells[r.key] !== 'object') cells[r.key] = {};
  return { title: n.title, summary: n.summary, type: n.type, validation: n.validation, cells };
}

/** What is sent to the API: referral ids are resolved by the server on read and are not stored. */
function cleanCells(cells: Draft['cells'], base?: Draft['cells']) {
  const c = clone(cells);
  if (base) for (const k of Object.keys(c)) if (!(k in base) && Object.keys(c[k] ?? {}).length === 0) delete c[k];
  for (const cell of Object.values(c)) {
    if (Array.isArray(cell?.referrals)) for (const r of cell.referrals) delete r.id;
    if (Array.isArray(cell?.statements)) for (const s of cell.statements) if (s.source === undefined || s.source === null) delete s.source;
  }
  return c;
}

class InspectorState {
  node = $state.raw<NodeDetail | null>(null);
  draft = $state<Draft | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);
  saving = $state(false);
  saveError = $state<string | null>(null);
  problems = $state<string[]>([]);
  effects = $state.raw<(Effects & { rev: number }) | null>(null);
  edges = $state.raw<NodeEdge[]>([]);
  private seq = 0;

  get dirty(): boolean {
    const n = this.node;
    const d = this.draft;
    if (!n || !d) return false;
    return d.title !== n.title || d.summary !== n.summary || d.type !== n.type || d.validation !== n.validation || JSON.stringify(cleanCells(d.cells, n.cells)) !== JSON.stringify(cleanCells(n.cells));
  }

  async load(id: string | null, keepDraft = false) {
    const mine = ++this.seq;
    if (!id) {
      this.node = null;
      this.draft = null;
      this.error = null;
      return;
    }
    const same = this.node?.id === id;
    if (!same) {
      this.effects = null;
      this.problems = [];
      this.saveError = null;
      this.loading = true;
    }
    try {
      const n = await get<NodeDetail>(`api/nodes/${id}`);
      if (mine !== this.seq) return;
      app.nodeCache.set(id, n);
      this.node = n;
      this.edges = n.edges;
      this.error = null;
      if (!(keepDraft && same && this.draft)) this.draft = draftOf(n);
    } catch (e) {
      if (mine !== this.seq) return;
      this.error = errText(e);
      this.node = null;
      this.draft = null;
    }
    this.loading = false;
  }

  revert() {
    if (this.node) this.draft = draftOf(this.node);
    this.problems = [];
    this.saveError = null;
  }

  async save() {
    const n = this.node;
    const d = this.draft;
    if (!n || !d) return;
    const body: Record<string, any> = {};
    if (d.title !== n.title) body.title = d.title;
    if (d.summary !== n.summary) body.summary = d.summary;
    if (d.type !== n.type) body.type = d.type;
    if (d.validation !== n.validation) body.validation = d.validation;
    const cells = cleanCells(d.cells, n.cells);
    if (JSON.stringify(cells) !== JSON.stringify(cleanCells(n.cells))) body.cells = cells;
    this.saving = true;
    this.saveError = null;
    this.problems = [];
    try {
      const r = await patch(`api/nodes/${n.id}`, body);
      this.node = r.node;
      this.edges = r.node.edges;
      this.draft = draftOf(r.node);
      this.effects = r.effects ? { ...r.effects, rev: r.rev } : null;
      await app.afterEdit(r);
    } catch (e) {
      this.saveError = errText(e);
      this.problems = e instanceof ApiError ? (e.detail?.problems ?? []) : [];
    }
    this.saving = false;
  }

  async moreEdges() {
    const n = this.node;
    if (!n) return;
    try {
      const r = await get<{ edges: NodeEdge[] }>(`api/nodes/${n.id}/edges?offset=${this.edges.length}&limit=40`);
      if (this.node?.id === n.id) this.edges = [...this.edges, ...r.edges];
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
}

export const insp = new InspectorState();
