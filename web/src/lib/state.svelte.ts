import { get, post, put, errText, type BoardData, type Mpn, type Status, type VEdge, type NodeDetail } from './api';

export type PanelId = 'library' | 'inspector' | 'run' | 'schema' | 'jobs' | 'budget' | 'runtime';
export type Side = 'left' | 'right';
export const PANELS: { id: PanelId; title: string; home: Side }[] = [
  { id: 'library', title: 'MPN library', home: 'left' },
  { id: 'inspector', title: 'Node', home: 'right' },
  { id: 'run', title: 'Run', home: 'right' },
  { id: 'schema', title: 'Schemas', home: 'right' },
  { id: 'jobs', title: 'Jobs', home: 'right' },
  { id: 'budget', title: 'Memory budget', home: 'right' },
  { id: 'runtime', title: 'Runtime', home: 'right' },
];
export const panelTitle = (id: PanelId) => PANELS.find((p) => p.id === id)!.title;
export const panelHome = (id: PanelId) => PANELS.find((p) => p.id === id)!.home;

export interface PanelPlace {
  where: Side | 'float' | 'closed';
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface PanelsState {
  left: { open: boolean; width: number; active: PanelId | null };
  right: { open: boolean; width: number; active: PanelId | null };
  items: Record<PanelId, PanelPlace>;
  order: PanelId[];
}
export const FLOAT_MIN_W = 320;
export const FLOAT_MIN_H = 240;

function defaultPanels(): PanelsState {
  const items = {} as Record<PanelId, PanelPlace>;
  PANELS.forEach((p, i) => {
    const open = p.id === 'library' || p.id === 'inspector' || p.id === 'run';
    items[p.id] = { where: open ? p.home : 'closed', x: 120 + i * 24, y: 80 + i * 24, w: 440, h: 420 };
  });
  return { left: { open: true, width: 272, active: 'library' }, right: { open: true, width: 372, active: 'inspector' }, items, order: [] };
}

/** Accept a stored arrangement only as far as it is well formed; anything else falls back to the default. */
function mergePanels(saved: any): PanelsState {
  const d = defaultPanels();
  if (!saved || typeof saved !== 'object') return d;
  for (const side of ['left', 'right'] as const) {
    const s = saved[side];
    if (s && typeof s === 'object') {
      if (typeof s.open === 'boolean') d[side].open = s.open;
      if (typeof s.width === 'number') d[side].width = Math.min(640, Math.max(220, s.width));
      if (s.active === null || PANELS.some((p) => p.id === s.active)) d[side].active = s.active;
    }
  }
  for (const p of PANELS) {
    const it = saved.items?.[p.id];
    if (it && ['left', 'right', 'float', 'closed'].includes(it.where)) {
      d.items[p.id] = {
        where: it.where,
        x: Number(it.x) || 100,
        y: Number(it.y) || 80,
        w: Math.max(FLOAT_MIN_W, Number(it.w) || 440),
        h: Math.max(FLOAT_MIN_H, Number(it.h) || 420),
      };
    }
  }
  if (Array.isArray(saved.order)) d.order = saved.order.filter((id: any) => PANELS.some((p) => p.id === id));
  for (const side of ['left', 'right'] as const) {
    const here = PANELS.filter((p) => d.items[p.id].where === side).map((p) => p.id);
    if (!d[side].active || !here.includes(d[side].active!)) d[side].active = here[0] ?? null;
  }
  return d;
}

export interface MenuItem {
  label: string;
  run?: () => void;
  disabled?: boolean;
  why?: string;
  kbd?: string;
  danger?: boolean;
}
export interface Toast {
  id: number;
  text: string;
  kind: 'info' | 'error';
  action?: { label: string; run: () => void };
}
export type Dialog =
  | { kind: 'newNode'; x: number | null; y: number | null }
  | { kind: 'newMpn' }
  | { kind: 'delete'; ids: string[] }
  | { kind: 'connect'; src: string; dst: string; srcTitle: string; dstTitle: string }
  | { kind: 'saveView' }
  | null;

export interface BoardApi {
  jumpTo(x: number, y: number, zoom?: number): void;
  fitAll(): void;
  fitSelection(): void;
  fitRegion(id: string): void;
  reload(geometryChanged?: boolean): void;
  view(): { x: number; y: number; zoom: number };
  center(): { x: number; y: number };
  startConnect(src?: string): void;
}

class AppState {
  status = $state.raw<Status | null>(null);
  offline = $state<string | null>(null);
  board = $state.raw<BoardData | null>(null);
  boardError = $state<string | null>(null);
  mpns = $state.raw<Mpn[]>([]);
  modelName = $state('N10K');
  /** head revision of the model; panels re-read when it changes */
  head = $state(0);
  /** bumped when geometry-only things change (board objects, saved views) */
  boardRev = $state(0);

  selection = $state.raw<string[]>([]);
  selectedEdge = $state.raw<VEdge | null>(null);
  inspected = $state<string | null>(null);
  /** nodes named by the last run's evidence; drawn with a ring */
  highlight = $state.raw<string[]>([]);
  clipboard = $state.raw<string[]>([]);
  tool = $state<'pan' | 'select' | 'connect'>('pan');
  zoom = $state(1);
  hint = $state<string | null>(null);

  panels = $state<PanelsState>(defaultPanels());
  panelsLoaded = false;
  narrow = $state(false);

  toasts = $state<Toast[]>([]);
  dialog = $state.raw<Dialog>(null);
  menu = $state.raw<{ x: number; y: number; items: (MenuItem | 'sep')[] } | null>(null);
  searchOpen = $state(false);

  boardApi: BoardApi | null = null;
  private toastId = 0;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  /** small cache of node records, used for board labels at high zoom and for references */
  nodeCache = new Map<string, NodeDetail>();

  // ---------------------------------------------------------------- loading
  async loadBoard() {
    try {
      const b = await get<BoardData>('api/board');
      this.board = b;
      this.boardError = null;
      if (b.head > this.head) this.head = b.head;
      if (!this.panelsLoaded) {
        this.panels = mergePanels(b.view?.panels);
        this.panelsLoaded = true;
      }
    } catch (e) {
      this.boardError = errText(e);
    }
  }
  async loadMpns() {
    try {
      this.mpns = (await get<{ mpns: Mpn[] }>('api/mpns')).mpns;
    } catch {
      /* the library panel shows its own error */
    }
  }
  async loadStatus() {
    try {
      const s = await get<Status>('api/status');
      this.status = s;
      this.offline = null;
      const head = Number(s.counts?.rev ?? 0);
      if (head !== this.head) this.head = head;
    } catch (e) {
      this.offline = errText(e);
    }
  }
  async loadModelName() {
    try {
      const m = await get<{ data: { id: string }[] }>('v1/models');
      const id = m.data?.[0]?.id ?? '';
      const cfg = id.replace(/^yaob-/i, '');
      if (cfg) this.modelName = cfg.toUpperCase();
    } catch {
      /* keep the default */
    }
  }

  /** Call after any model edit: adopt the new head, refresh what depends on it. */
  async afterEdit(res?: { rev?: number | null }) {
    if (res?.rev && res.rev > this.head) this.head = res.rev;
    this.nodeCache.clear();
    await Promise.all([this.loadStatus(), this.loadBoard(), this.loadMpns()]);
    this.boardApi?.reload();
  }

  async undo() {
    try {
      const r = await post('api/undo');
      await this.afterEdit(r.undone ? { rev: r.head } : undefined);
      this.head = r.head;
      this.toast(r.undone ? `Undone. Now at revision ${r.head}.` : 'Nothing to undo.');
    } catch (e) {
      this.toast(errText(e), 'error');
    }
  }
  async redo() {
    try {
      const r = await post('api/redo');
      await this.afterEdit(r.redone ? { rev: r.head } : undefined);
      this.head = r.head;
      this.toast(r.redone ? `Redone. Now at revision ${r.head}.` : 'Nothing to redo.');
    } catch (e) {
      this.toast(errText(e), 'error');
    }
  }

  // ---------------------------------------------------------------- selection
  select(ids: string[], inspect = true) {
    this.selection = ids;
    this.selectedEdge = null;
    if (inspect && ids.length === 1) this.openNode(ids[0]);
  }
  openNode(id: string) {
    this.inspected = id;
    this.showPanel('inspector');
  }
  /** Jump the board to a node (which may not be loaded), select it and open it. */
  async goToNode(id: string, pos?: { x?: number; y?: number } | null) {
    let x = pos?.x;
    let y = pos?.y;
    if (x == null || y == null) {
      try {
        const n = await get<NodeDetail>(`api/nodes/${id}`);
        x = n.position?.x;
        y = n.position?.y;
      } catch {
        /* the inspector reports the failure */
      }
    }
    if (x != null && y != null) this.boardApi?.jumpTo(x, y, Math.max(this.boardApi.view().zoom, 3.2));
    this.select([id]);
  }

  // ---------------------------------------------------------------- toasts, menus
  toast(text: string, kind: 'info' | 'error' = 'info', action?: Toast['action']) {
    const id = ++this.toastId;
    this.toasts = [...this.toasts.slice(-2), { id, text, kind, action }];
    setTimeout(() => this.dismiss(id), action ? 12000 : kind === 'error' ? 9000 : 5000);
  }
  dismiss(id: number) {
    this.toasts = this.toasts.filter((t) => t.id !== id);
  }
  openMenu(x: number, y: number, items: (MenuItem | 'sep')[]) {
    this.menu = { x, y, items };
  }

  // ---------------------------------------------------------------- panels
  showPanel(id: PanelId) {
    const p = this.panels.items[id];
    if (p.where === 'closed') p.where = panelHome(id);
    if (p.where === 'float') {
      this.raise(id);
    } else {
      const dock = this.panels[p.where];
      dock.open = true;
      dock.active = id;
    }
    this.savePanels();
  }
  closePanel(id: PanelId) {
    const p = this.panels.items[id];
    const was = p.where;
    p.where = 'closed';
    this.panels.order = this.panels.order.filter((x) => x !== id);
    if (was === 'left' || was === 'right') this.fixActive(was);
    this.savePanels();
  }
  floatPanel(id: PanelId, at?: { x: number; y: number }) {
    const p = this.panels.items[id];
    const was = p.where;
    p.where = 'float';
    if (at) {
      p.x = at.x;
      p.y = at.y;
    }
    this.clampFloat(id);
    this.raise(id);
    if (was === 'left' || was === 'right') this.fixActive(was);
    this.savePanels();
  }
  dockPanel(id: PanelId, side: Side = panelHome(id)) {
    const p = this.panels.items[id];
    p.where = side;
    this.panels.order = this.panels.order.filter((x) => x !== id);
    this.panels[side].open = true;
    this.panels[side].active = id;
    this.savePanels();
  }
  raise(id: PanelId) {
    const o = this.panels.order;
    if (o[o.length - 1] !== id) this.panels.order = [...o.filter((x) => x !== id), id];
  }
  clampFloat(id: PanelId) {
    const p = this.panels.items[id];
    const W = window.innerWidth;
    const H = window.innerHeight;
    p.w = Math.max(FLOAT_MIN_W, Math.min(p.w, W - 8));
    p.h = Math.max(FLOAT_MIN_H, Math.min(p.h, H - 40 - 24 - 8));
    p.x = Math.max(4 - p.w + 80, Math.min(p.x, W - 80));
    p.y = Math.max(44, Math.min(p.y, H - 60));
  }
  docked(side: Side): PanelId[] {
    return PANELS.filter((p) => this.panels.items[p.id].where === side).map((p) => p.id);
  }
  private fixActive(side: Side) {
    const here = this.docked(side);
    const dock = this.panels[side];
    if (!dock.active || !here.includes(dock.active)) dock.active = here[0] ?? null;
  }
  isOpen(id: PanelId) {
    return this.panels.items[id].where !== 'closed';
  }

  /** Persist view and panel arrangement on the server, so they survive a reload on any machine. */
  savePanels(delay = 350) {
    if (!this.panelsLoaded) return;
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => this.flushView(), delay);
  }
  flushView(keepalive = false) {
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = null;
    if (!this.panelsLoaded || !this.boardApi) return;
    const v = this.boardApi.view();
    const body = { x: v.x, y: v.y, zoom: v.zoom, panels: $state.snapshot(this.panels) };
    const p = keepalive ? fetch('api/board/view', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), keepalive: true }) : put('api/board/view', body);
    p.catch(() => {});
  }
}

export const app = new AppState();

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
