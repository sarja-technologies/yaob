"""SQLite schema for the Yaob model store.

Design rules:

* Semantic objects (nodes, memberships, edges, MPNs, domains, node sources) are
  stored as immutable *versions* with a validity interval [rev_from, rev_to).
  A row with rev_to IS NULL is the head version. Every edit transaction creates
  one new revision, so any past revision can be read consistently and undo is
  a new revision that restores what an earlier one closed.
* Board geometry (layout, regions, saved views) lives in separate tables and is
  not versioned with the model: moving a node never changes model semantics.
* Ids are 64-bit integers inside SQLite and decimal strings at every API
  boundary, so they are never passed through a JavaScript number.
"""

SCHEMA_VERSION = 1

DDL = r"""
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ids(kind TEXT PRIMARY KEY, next INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS revisions(
  rev INTEGER PRIMARY KEY,
  parent INTEGER,
  created TEXT NOT NULL,
  actor TEXT NOT NULL,
  kind TEXT NOT NULL,            -- import | edit | undo | redo | mpn | schema | system
  message TEXT NOT NULL DEFAULT '',
  undo_of INTEGER,               -- revision this one reverts (undo) or re-applies (redo)
  stats TEXT NOT NULL DEFAULT '{}'
);

-- Named, sealed model revisions (what a bundle or the runtime can pin by name).
CREATE TABLE IF NOT EXISTS releases(
  name TEXT PRIMARY KEY, rev INTEGER NOT NULL, created TEXT NOT NULL, manifest TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sources(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,            -- wikipedia | wikidata | wikipedia-list | authored | synthetic | user
  origin TEXT NOT NULL,          -- reference | synthetic | user
  title TEXT NOT NULL,
  url TEXT NOT NULL DEFAULT '',
  license TEXT NOT NULL DEFAULT '',
  license_url TEXT NOT NULL DEFAULT '',
  revision TEXT NOT NULL DEFAULT '',
  retrieved_at TEXT NOT NULL DEFAULT '',
  attribution TEXT NOT NULL DEFAULT '',
  UNIQUE(kind, url, revision)
);

CREATE TABLE IF NOT EXISTS cell_schemas(
  id TEXT NOT NULL, version INTEGER NOT NULL, name TEXT NOT NULL,
  spec TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', created TEXT NOT NULL,
  PRIMARY KEY(id, version)
);

CREATE TABLE IF NOT EXISTS mpn_v(
  vid INTEGER PRIMARY KEY, mpn_id INTEGER NOT NULL, slug TEXT NOT NULL, name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'domain',        -- domain | diagnostic | user
  description TEXT NOT NULL DEFAULT '',
  version TEXT NOT NULL DEFAULT '0.1.0',
  manifest TEXT NOT NULL DEFAULT '{}',
  installed INTEGER NOT NULL DEFAULT 1,
  enabled INTEGER NOT NULL DEFAULT 1,
  rev_from INTEGER NOT NULL, rev_to INTEGER
);
CREATE INDEX IF NOT EXISTS mpn_head ON mpn_v(mpn_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS mpn_hist ON mpn_v(mpn_id, rev_from);

CREATE TABLE IF NOT EXISTS domain_v(
  vid INTEGER PRIMARY KEY, domain_id INTEGER NOT NULL, mpn_id INTEGER NOT NULL, parent_id INTEGER,
  slug TEXT NOT NULL, name TEXT NOT NULL, depth INTEGER NOT NULL DEFAULT 1,
  rev_from INTEGER NOT NULL, rev_to INTEGER
);
CREATE INDEX IF NOT EXISTS domain_head ON domain_v(domain_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS domain_mpn_head ON domain_v(mpn_id) WHERE rev_to IS NULL;

CREATE TABLE IF NOT EXISTS node_v(
  vid INTEGER PRIMARY KEY, node_id INTEGER NOT NULL,
  uid TEXT NOT NULL,                           -- portable identity, e.g. yaob:wd:Q48314 or yaob:u:<id>
  type TEXT NOT NULL,                          -- knowledge | procedure | rule | tool | learned_memory
  title TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  aliases TEXT NOT NULL DEFAULT '[]',
  schema_id TEXT NOT NULL, schema_version INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'live',         -- live | disabled | tombstone
  validation TEXT NOT NULL DEFAULT 'unverified', -- unverified | source_backed | checked | failed
  origin TEXT NOT NULL DEFAULT 'reference',    -- reference | synthetic | user | diagnostic
  cells TEXT NOT NULL,                         -- JSON: {role_key: {...}}
  content_hash TEXT NOT NULL,                  -- hash of the fields that learned state and indexes depend on
  rev_from INTEGER NOT NULL, rev_to INTEGER
);
CREATE INDEX IF NOT EXISTS node_head ON node_v(node_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS node_hist ON node_v(node_id, rev_from);
CREATE INDEX IF NOT EXISTS node_uid_head ON node_v(uid) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS node_closed ON node_v(rev_to) WHERE rev_to IS NOT NULL;
CREATE INDEX IF NOT EXISTS node_opened ON node_v(rev_from);

CREATE TABLE IF NOT EXISTS member_v(
  vid INTEGER PRIMARY KEY, node_id INTEGER NOT NULL, mpn_id INTEGER NOT NULL, domain_id INTEGER,
  role TEXT NOT NULL DEFAULT 'primary',        -- primary | shared
  evidence TEXT NOT NULL DEFAULT '{}',
  rev_from INTEGER NOT NULL, rev_to INTEGER
);
CREATE INDEX IF NOT EXISTS member_node_head ON member_v(node_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS member_mpn_head ON member_v(mpn_id, domain_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS member_hist ON member_v(node_id, rev_from);
CREATE INDEX IF NOT EXISTS member_closed ON member_v(rev_to) WHERE rev_to IS NOT NULL;
CREATE INDEX IF NOT EXISTS member_opened ON member_v(rev_from);

CREATE TABLE IF NOT EXISTS edge_v(
  vid INTEGER PRIMARY KEY, edge_id INTEGER NOT NULL, src INTEGER NOT NULL, dst INTEGER NOT NULL,
  type TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
  origin TEXT NOT NULL DEFAULT 'reference',    -- reference | derived | user
  evidence TEXT NOT NULL DEFAULT '{}',
  rev_from INTEGER NOT NULL, rev_to INTEGER
);
CREATE INDEX IF NOT EXISTS edge_head ON edge_v(edge_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS edge_src_head ON edge_v(src) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS edge_dst_head ON edge_v(dst) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS edge_closed ON edge_v(rev_to) WHERE rev_to IS NOT NULL;
CREATE INDEX IF NOT EXISTS edge_opened ON edge_v(rev_from);

CREATE TABLE IF NOT EXISTS node_source_v(
  vid INTEGER PRIMARY KEY, node_id INTEGER NOT NULL, source_id INTEGER NOT NULL, locator TEXT NOT NULL DEFAULT '',
  rev_from INTEGER NOT NULL, rev_to INTEGER
);
CREATE INDEX IF NOT EXISTS nsrc_head ON node_source_v(node_id) WHERE rev_to IS NULL;
CREATE INDEX IF NOT EXISTS nsrc_closed ON node_source_v(rev_to) WHERE rev_to IS NOT NULL;
CREATE INDEX IF NOT EXISTS nsrc_opened ON node_source_v(rev_from);

-- Head-only lexical index. rowid = node_id. Updated in the same transaction as the edit.
CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(
  title, aliases, summary, body, sources, tokenize = 'unicode61 remove_diacritics 2'
);

-- What each derived artefact was built from, per node. A mismatch with node_v.content_hash means "dirty".
CREATE TABLE IF NOT EXISTS derived_state(
  node_id INTEGER PRIMARY KEY,
  embed_hash TEXT, embed_row INTEGER,          -- routing/semantic index row and the content it was computed from
  learned_hash TEXT, learned_checkpoint TEXT   -- content the node's learned state was last trained on
);

-- Text-derived state for nodes that the packaged checkpoint does not cover at their current content:
-- new nodes and nodes edited since packaging. Written by indexing; replaced by the next package.
CREATE TABLE IF NOT EXISTS overlay(
  node_id INTEGER PRIMARY KEY, content_hash TEXT NOT NULL, slots INTEGER NOT NULL, dim INTEGER NOT NULL,
  node_emb BLOB NOT NULL, stmt_emb BLOB NOT NULL, updated TEXT NOT NULL
);

-- ---------------------------------------------------------------- boards (geometry only)
CREATE TABLE IF NOT EXISTS boards(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, created TEXT NOT NULL, view TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS layout(
  board_id INTEGER NOT NULL, node_id INTEGER NOT NULL, x REAL NOT NULL, y REAL NOT NULL,
  pinned INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(board_id, node_id)
) WITHOUT ROWID;
CREATE VIRTUAL TABLE IF NOT EXISTS layout_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE TABLE IF NOT EXISTS regions(
  board_id INTEGER NOT NULL, kind TEXT NOT NULL, ref_id INTEGER NOT NULL,
  cx REAL NOT NULL, cy REAL NOT NULL, r REAL NOT NULL, PRIMARY KEY(board_id, kind, ref_id)
) WITHOUT ROWID;
-- Aggregate pyramid over node positions: level l has square cells of side BASE * 2^l. Maintained incrementally
-- when nodes are placed, moved or removed, so far-zoom viewports never touch individual nodes.
CREATE TABLE IF NOT EXISTS tiles(
  level INTEGER NOT NULL, gx INTEGER NOT NULL, gy INTEGER NOT NULL, n INTEGER NOT NULL, sx REAL NOT NULL, sy REAL NOT NULL,
  PRIMARY KEY(level, gx, gy)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS saved_views(
  id INTEGER PRIMARY KEY, board_id INTEGER NOT NULL, name TEXT NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, zoom REAL NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS board_objects(
  id INTEGER PRIMARY KEY, board_id INTEGER NOT NULL, kind TEXT NOT NULL,   -- group | note | reference | panel
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL DEFAULT 0, h REAL NOT NULL DEFAULT 0,
  props TEXT NOT NULL DEFAULT '{}', created TEXT NOT NULL
);

-- ---------------------------------------------------------------- learned state bookkeeping
CREATE TABLE IF NOT EXISTS banks(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, spec TEXT NOT NULL, core TEXT NOT NULL, created TEXT NOT NULL
);
-- Stable mapping node -> tensor row. Rows are append-only; a deleted node keeps its row (tombstoned)
-- until an explicit compaction writes a new bank with a new mapping.
CREATE TABLE IF NOT EXISTS bank_rows(
  bank_id TEXT NOT NULL, node_id INTEGER NOT NULL, row INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'allocated',     -- allocated | trained | tombstone
  PRIMARY KEY(bank_id, node_id)
) WITHOUT ROWID;
CREATE UNIQUE INDEX IF NOT EXISTS bank_rows_row ON bank_rows(bank_id, row);
CREATE TABLE IF NOT EXISTS checkpoints(
  id TEXT PRIMARY KEY, bank_id TEXT, kind TEXT NOT NULL, path TEXT NOT NULL, parent TEXT,
  step INTEGER NOT NULL DEFAULT 0, rev INTEGER, metrics TEXT NOT NULL DEFAULT '{}',
  sha256 TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'saved',  -- saved | active | previous | retired
  created TEXT NOT NULL, job_id TEXT
);

-- ---------------------------------------------------------------- durable jobs
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,   -- queued | running | cancelling | cancelled | failed | succeeded | interrupted
  params TEXT NOT NULL DEFAULT '{}', progress TEXT NOT NULL DEFAULT '{}', result TEXT NOT NULL DEFAULT '{}',
  created TEXT NOT NULL, started TEXT, finished TEXT, unit TEXT, pid INTEGER, heartbeat TEXT,
  resume_of TEXT, attempt INTEGER NOT NULL DEFAULT 1, error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created);

-- Live-node counts per MPN and per domain at head, updated incrementally in the same transaction as each edit.
CREATE TABLE IF NOT EXISTS summary_counts(
  kind TEXT NOT NULL, ref_id INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 0, shared INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(kind, ref_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

VERSIONED = {
    # table: (entity id column, columns copied when a version is re-opened)
    "node_v": ("node_id", ["node_id", "uid", "type", "title", "summary", "aliases", "schema_id", "schema_version",
                           "status", "validation", "origin", "cells", "content_hash"]),
    "member_v": ("node_id", ["node_id", "mpn_id", "domain_id", "role", "evidence"]),
    "edge_v": ("edge_id", ["edge_id", "src", "dst", "type", "weight", "origin", "evidence"]),
    "mpn_v": ("mpn_id", ["mpn_id", "slug", "name", "kind", "description", "version", "manifest", "installed", "enabled"]),
    "domain_v": ("domain_id", ["domain_id", "mpn_id", "parent_id", "slug", "name", "depth"]),
    "node_source_v": ("node_id", ["node_id", "source_id", "locator"]),
}
