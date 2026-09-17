# HierarchyReducerV2 module layout

`HierarchyReducerV2` is intentionally header-only, but no longer monolithic. The public facade stays in `include/SUAPI-CppWrapper/model/OptimizationV2.hpp`; implementation is split under `model/detail/` so each hot path can be reviewed, benchmarked and changed independently.

## Files

### `OptimizationV2.hpp`

Owns the stable reducer API and long-lived state:

- `traverse()` / `traverse_entities()` / `traverse_partitioned()`;
- owning output buckets;
- immutable source caches;
- shared `TraversalVisibilityV2` per-run state;
- public `ReducerStatsV2` result;
- cache-reset contract.

Keep this file small. New algorithms should normally live in one of the detail modules below.

### `detail/OptimizationV2Types.hpp`

Owns data contracts used by reducer internals:

- `ReducerStatsV2` public counter contract;
- material state;
- face/group/instance prototype entries;
- entity-level prototype;
- cached MeshHelper face payload.

These structs contain source data or counters, not traversal policy.

### `detail/OptimizationV2Material.inl`

Owns material/UV/normal helper logic:

- material-state normalization;
- JSON-safe two-sided material keys;
- texture-scale cache construction/lookup;
- inverse-transpose style normal transformation;
- STQ-to-UV conversion.

Any change here needs material, projected-UV, two-sided, mirrored and non-uniform-scale regression coverage.

### `detail/OptimizationV2Cache.inl`

Owns SketchUp source extraction and reusable immutable caches:

- root/definition prototype construction;
- definition cache hit/miss path;
- face cache keys;
- `SUMeshHelper` creation and extraction;
- front/back STQ payload capture.

Performance target: expensive SketchUp source extraction scales with unique source definitions/faces rather than expanded instance occurrences.

### `detail/OptimizationV2Emit.inl`

Owns expanded geometry emission:

- output capacity growth;
- vertex welding/hash lookup;
- final-unit position storage;
- per-occurrence vertex transformation;
- material bucket selection;
- triangle winding and index emission.

Performance target: work here scales with unavoidable expanded output size while keeping hash/vector reallocations bounded.

### `detail/OptimizationV2Traversal.inl`

Owns per-run orchestration:

- reset/begin-run state;
- shared visibility reset/stat synchronization;
- visible/hidden routing;
- hierarchy recursion;
- per-level transform determinant/inverse computation.

This module must use `TraversalVisibilityV2`; it must not introduce a second implementation of tag/folder/scene visibility.

### `TraversalVisibilityV2.hpp`

Shared generic native effective-visibility evaluator used by reducer and wire batching. It owns per-run hidden/override sets and layer/folder visibility caches. SDK errors and pathological folder chains intentionally fail closed to preserve the hardened reducer semantics.

## Dependency direction

```text
OptimizationV2.hpp
  -> OptimizationV2Types.hpp
  -> TraversalVisibilityV2.hpp
  -> Material.inl
  -> Cache.inl
  -> Emit.inl
  -> Traversal.inl

LooseEdgeBatchV2.hpp
  -> TraversalVisibilityV2.hpp
```

Detail modules must not include Blender or pybind types. Blender/Python ownership remains in `Svein_SketchupImporter`.

## Editing rule

Do not grow `OptimizationV2.hpp` back into a monolith. When a new performance feature is more than a small orchestration change, place it in the detail module that owns that concern or create a new narrow module with an explicit contract.

## Correctness gates after this split

The split is intended to be behavior-preserving except that reducer and wire traversal now share the same visibility implementation. Rebuild against the SketchUp SDK and compare against the pre-split V2 for:

- material bucket names/counts;
- vertex/index/face counts;
- UV/front-back UV values;
- mirrored winding;
- normals under non-uniform transforms;
- visible/hidden partition membership;
- repeated-component cache counters;
- loose-edge visible/hidden membership.

Source review is not runtime validation; keep the parent importer PR draft until these checks run.
