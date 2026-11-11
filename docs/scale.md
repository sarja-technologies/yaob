# Large networks: what is measured, what is designed, what is not claimed

Yaob N10K has 10,000 nodes. This document covers how the same store, API and board behave at larger sizes. It separates three things:

- **Measured** on a synthetic 1,000,000-node fixture on the development workstation (i9-11900F, SATA SSD).
- **Designed**: how the layout grows past one SQLite file. Not built, not measured.
- **Not claimed**: nothing here shows a billion-node network running. A schema that accepts 64-bit ids is not evidence of that.

The fixture is synthetic. Its nodes are placeholders (`origin=synthetic`). It is not a model, is never trained on, and is never counted as Yaob nodes. Build and measure it with:

```bash
python scripts/synth_scale_fixture.py build   data/synthetic-1m.sqlite --nodes 1000000 --mpns 40
python scripts/synth_scale_fixture.py measure data/synthetic-1m.sqlite --out docs/measurements/scale-1m.json
```

## Measured at 1,000,000 synthetic nodes, 3,000,000 edges

Build: 35 s to insert, 32 s to lay out. File size 1.29 GiB.

| Bytes per node | |
|---|---|
| Node record and its indexes | 654 |
| Text index (FTS5) | 227 |
| Memberships | 87 |
| Layout and spatial index | 87 |
| Node sources | 47 |
| **All in, including edges at 3 per node** | **1,390** |
| Per edge, with indexes | 94 |

These records are small (two short statements, one fact). Real N10K nodes are several times larger; `docs/measurements/` holds the N10K figures once the corpus is loaded.

| Query (server side, 60 random viewports each, cap 2,500 items) | p50 | p95 | Items returned, max | Payload p95 |
|---|---|---|---|---|
| Whole board | 6.7 ms | 9.6 ms | 621 aggregate cells | 132 kB |
| One MPN wide | 5.9 ms | 9.6 ms | 1,194 | 69 kB |
| One domain wide | 19.2 ms | 28.0 ms | 2,031 nodes | 368 kB |
| Node level | 4.5 ms | 6.1 ms | 39 nodes | 9 kB |
| Cell level | 4.4 ms | 5.7 ms | 2 nodes | 1 kB |
| Text search, two-word query | 21.4 ms | 26.4 ms | 20 | |
| Fetch one node by id | 0.03 ms | 0.04 ms | | |
| Edit one node (record, text index, counts, new revision) | 4.7 ms | | | |

The first run of this measurement found a defect, kept here because it is the reason the fixture exists: node-level viewports took 1.1 s at this size. SQLite had chosen to walk the position table and probe the spatial index a million times. The query now fixes the join order (spatial index first). At 10,000 nodes the wrong plan cost under a millisecond and would not have been noticed.

## Why the browser never holds the graph

- The board asks for a viewport: a bounding box, a scale and a cap. The server answers with at most `cap` items.
- Far out, it answers from a **tile pyramid**: sixteen levels of square cells, level *l* having side 104 × 2^l world units, each cell storing a node count and centroid. The pyramid is updated when a node is placed, moved or removed (sixteen row updates), so an overview costs a few hundred row reads however large the network is.
- Closer in, when the pyramid estimates that the viewport holds few enough nodes, the server reads those nodes through the R-tree spatial index.
- Regions (MPNs, domains) carry counts from a summary table that each edit updates in its own transaction, for the nodes it touched only.
- Labels are bounded (160 by default, most-linked first). Edges are requested only at node level and capped; at overview the board draws MPN-to-MPN bundles from aggregate counts.
- Search and jump work for nodes that are not loaded: results carry coordinates.
- Node ids are strings end to end.

## Past one file: the designed layout (not built)

A single SQLite file is a sound container for 10^4 to roughly 10^7 nodes on one machine. Beyond that the same contracts are served by partitions:

| Piece | N10K today | Designed growth |
|---|---|---|
| Canonical records | one SQLite file, versioned rows | one store file per MPN (or per domain for very large MPNs), plus a small catalogue file: MPNs, domains, revisions, cross-partition edges. A request touches the catalogue and the partitions of the MPNs it activates |
| Text search | FTS5 in the same file | one FTS index per partition, queried only for active or filtered MPNs; a catalogue-level title index for global jump-to-node |
| Routing keys | per-MPN shard, exhaustive scan inside the active MPNs (about 500 keys each) | two more routing levels inside an MPN: domain prototypes, then an approximate index (IVF or HNSW) per domain shard. Approximation costs recall; it has to be measured against the exhaustive scan before use |
| Learned node state | per-MPN safetensors shard, loaded whole | shards of fixed row count per domain, memory-mapped, loaded by row block. The cache already accounts in bytes and pins in-flight shards; block granularity is the change |
| Spatial index and pyramid | R-tree and tiles tables | per-region R-trees; the pyramid stays one table (its size depends on occupied area, not node count) |
| Revisions | interval-versioned rows, one revision counter | the same counter in the catalogue; partitions record the catalogue revision in their rows |

No distributed database is proposed. Partitioned files on one machine cover the range a workstation can hold.

## Storage budget for 10^9 nodes (extrapolation)

Assumptions: the fixture's record shape and edge density scale linearly; learned state is the current bank layout (6 slots × 384 + 256-wide key + gate per node); statement embeddings are kept (6 × 1024 per node); float16 on disk.

| | Bytes per node | At 10^9 nodes |
|---|---|---|
| Records, text index, memberships, layout, edges | 1,390 | 1.26 TiB |
| Learned node state | 5,122 | 4.66 TiB |
| Frozen statement embeddings | 12,288 | 11.18 TiB |
| **Total** | **18,800** | **about 17 TiB** |

Uncertainty and what moves the number:

- Real records are larger than the fixture's. Using the measured N10K record size instead raises the first row several-fold.
- The statement embeddings dominate. They exist so that an edited or untrained node has usable state without re-encoding. Storing them at 256 dimensions, or recomputing them on demand, would cut the total by more than half; neither has been evaluated for quality.
- Index overheads were measured at 10^6 rows. B-tree depth grows slowly, but this is an extrapolation across three orders of magnitude.

The Spark has 1 TiB free. A billion nodes at these sizes does not fit on it, and nothing in this release says otherwise. What the design does allow is the control model the brief describes: the installed network may be far larger than working memory, because a request loads only the shards of the MPNs it activates and the cache is bounded in bytes.
