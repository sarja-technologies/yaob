// Thin client for the MPN Studio API. All URLs are relative so the same build works behind the
// Vite proxy and when served by the API process itself. Ids are decimal strings throughout.

export class ApiError extends Error {
  status: number;
  detail: any;
  constructor(message: string, status: number, detail?: any) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

export async function req<T = any>(method: string, path: string, body?: unknown, signal?: AbortSignal, keepalive = false): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      keepalive,
    });
  } catch (e: any) {
    if (e?.name === 'AbortError') throw e;
    throw new ApiError('The MPN Studio server did not answer.', 0);
  }
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) {
    const msg = data?.error ?? (typeof data?.detail === 'string' ? data.detail : null) ?? (text && text.length < 300 ? text : `The server answered ${res.status}.`);
    throw new ApiError(String(msg), res.status, data?.detail);
  }
  return data as T;
}

export const get = <T = any>(path: string, signal?: AbortSignal) => req<T>('GET', path, undefined, signal);
export const post = <T = any>(path: string, body: unknown = {}, signal?: AbortSignal) => req<T>('POST', path, body, signal);
export const put = <T = any>(path: string, body: unknown = {}) => req<T>('PUT', path, body);
export const patch = <T = any>(path: string, body: unknown = {}) => req<T>('PATCH', path, body);
export const del = <T = any>(path: string) => req<T>('DELETE', path);

export function qs(params: Record<string, string | number | undefined | null>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') u.set(k, String(v));
  const s = u.toString();
  return s ? '?' + s : '';
}

export function errText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

// ---------------------------------------------------------------- shapes
export interface Region {
  kind: 'mpn' | 'domain';
  id: string;
  x: number;
  y: number;
  r: number;
  name: string;
  count: number;
  shared?: number;
  enabled?: boolean;
  installed?: boolean;
  mpn_id?: string;
  depth?: number;
}
export interface VNode {
  id: string;
  x: number;
  y: number;
  t: string;
  s: string;
  v: string;
  o: string;
  m: string;
  nm: number;
  deg: number;
  pin: number;
  learned: 'trained' | 'stale' | 'none';
  indexed: boolean;
  title?: string;
}
export interface VCell {
  x: number;
  y: number;
  n: number;
  m: string;
}
export interface VEdge {
  id: string;
  s: string;
  d: string;
  t: string;
  w: number;
}
export interface Viewport {
  level: 'nodes' | 'cells';
  total_in_view: number;
  cell_size?: number;
  nodes?: VNode[];
  cells?: VCell[];
  regions: Region[];
  edges?: VEdge[];
  head: number;
  ms: number;
}
export interface BoardObject {
  id: string;
  kind: 'group' | 'note' | 'panel' | 'reference';
  x: number;
  y: number;
  w: number;
  h: number;
  props: Record<string, any>;
}
export interface SavedView {
  id: string;
  name: string;
  x: number;
  y: number;
  zoom: number;
}
export interface Bundle {
  a: string;
  b: string;
  edges: number;
  shared: number;
}
export interface BoardData {
  id: string;
  name: string;
  view: { x?: number; y?: number; zoom?: number; panels?: any };
  bounds: [number, number, number, number];
  spacing: number;
  mpns: Region[];
  bundles: Bundle[];
  saved_views: SavedView[];
  objects: BoardObject[];
  head: number;
}
export interface Mpn {
  id: string;
  slug: string;
  name: string;
  description: string;
  installed: boolean;
  enabled: boolean;
  resident: boolean;
  resident_bytes: number;
  training: string;
  validated: boolean;
  node_count: number;
  shared_count: number;
  primary_nodes: number;
  trained_nodes: number;
}
export interface Domain {
  id: string;
  mpn_id: string;
  parent_id: string | null;
  name: string;
  depth: number;
  node_count: number;
}
export interface SchemaField {
  key: string;
  type: string;
  required?: boolean;
  description?: string;
  max_items?: number;
  min?: number;
  max?: number;
}
export interface SchemaRole {
  key: string;
  label: string;
  runtime_role: string;
  description?: string;
  fields: SchemaField[];
}
export interface SchemaSpec {
  id: string;
  version: number;
  name: string;
  description?: string;
  extends?: { id: string; version: number };
  roles: SchemaRole[];
}
export interface SchemaEntry {
  id: string;
  version: number;
  name: string;
  status: string;
  cells: number;
  nodes: number;
  spec: SchemaSpec;
}
export interface Membership {
  mpn_id: string;
  domain_id: string | null;
  role: string;
  mpn_name: string;
  mpn_enabled: boolean;
  mpn_installed: boolean;
}
export interface NodeSource {
  id: string;
  kind: string;
  title: string;
  url: string;
  license: string;
  license_url: string;
  revision: string;
  retrieved_at: string;
  attribution: string;
  locator: string;
}
export interface NodeEdge {
  id: string;
  src: string;
  dst: string;
  type: string;
  weight: number;
  origin: string;
  other: { id: string; title: string; type: string; status: string };
  direction: 'in' | 'out';
}
export interface NodeDetail {
  id: string;
  uid: string;
  type: string;
  title: string;
  summary: string;
  status: string;
  validation: string;
  origin: string;
  schema: { id: string; version: number; name: string; roles: SchemaRole[] };
  content_hash: string;
  rev: number;
  aliases: string[];
  cells: Record<string, Record<string, any>>;
  memberships: Membership[];
  sources: NodeSource[];
  edge_count: number;
  edges: NodeEdge[];
  state: { index: string; learned: string; checkpoint: string | null; rows: { bank: string; row: number; state: string }[]; explain: string };
  problems: string[];
  position: { x: number; y: number } | null;
  cost: { record_bytes: number; text_tokens_estimate: number; learned_parameters: number; learned_bytes_f32: number };
}
export interface Effects {
  content_changed?: boolean;
  live_now: string[];
  after_indexing: string[];
  after_training: string[];
}
export interface Status {
  version: string;
  counts: Record<string, any>;
  dirty: { live_nodes: number; to_index: number; trained_stale: number; untrained: number; trained_current: number };
  runtime: { available: boolean; reason?: string; [k: string]: any };
  memory: { total_gib: number; available_gib: number; api_process_mib: number; swap_used_gib: number };
  budget: Record<string, any>;
  undo: number | null;
  redo: number | null;
  jobs_active: number;
}
export interface SearchResult {
  id: string;
  title: string;
  type: string;
  status: string;
  validation: string;
  summary: string;
  score: number;
  match?: string;
  snippet?: string;
  mpn?: string;
  mpn_id?: string;
  x?: number;
  y?: number;
}
export interface Job {
  id: string;
  kind: string;
  state: string;
  params: Record<string, any>;
  progress: Record<string, any>;
  result: Record<string, any>;
  created: string;
  started: string | null;
  finished: string | null;
  attempt: number;
  resume_of: string | null;
  error: string | null;
}

// ---------------------------------------------------------------- small formatters
export const fmtInt = (n: number | null | undefined) => (n == null ? '' : Math.round(n).toLocaleString('en-US'));
export function fmtBytes(n: number): string {
  if (!n) return '0 B';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return (v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)) + ' ' + u[i];
}
export const plural = (n: number, one: string, many = one + 's') => `${fmtInt(n)} ${n === 1 ? one : many}`;
export const human = (s: string) => (s ?? '').replace(/_/g, ' ');
export function when(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(+d)) return iso;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
