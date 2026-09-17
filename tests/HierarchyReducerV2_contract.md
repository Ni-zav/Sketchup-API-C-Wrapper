# HierarchyReducerV2 regression contract

This file documents the native invariants that must be exercised by SDK-backed tests. It intentionally lives beside the wrapper tests so future fixtures can turn each row into executable coverage.

## Core cache invariants

- Repeated component instances must reuse one definition prototype per definition entity ID.
- Repeated faces must reuse one tessellation cache entry per `(face entity ID, two-sided mode)`.
- `mesh_helper_creates` should scale with unique faces, not expanded instance count.
- `reset_geometry_cache()` must invalidate both definition and face caches after model mutation.

## Geometry invariants

- Mirrored transforms must flip triangle winding exactly once.
- Normal transformation must preserve legacy inverse-transform behavior under non-uniform scale.
- Welding must continue to quantize in SketchUp internal units before applying output `unit_scale`.
- Triangle indices outside the local MeshHelper vertex range must never be emitted.
- Degenerate/empty MeshHelper results must remain non-fatal.

## Material and UV invariants

- Front material overrides inherited material when valid.
- Back material is not queried in one-sided mode.
- Two-sided material keys must escape JSON-sensitive characters identically to the importer decoder contract.
- Inherited texture scale is applied only when a face side inherits its material.
- Front/back STQ fallbacks must remain deterministic when the SDK does not return full coordinate arrays.

## Visibility invariants

- Raw hidden state is used when scene hidden-object overrides are disabled.
- Scene hidden-object IDs replace raw hidden state when enabled, matching legacy behavior.
- Layer visibility and nested folder visibility are cached without changing semantics.
- Ancestor invisibility propagates to descendants.
- `visibility_filter=0/1/2` preserves ALL/VISIBLE/HIDDEN behavior.
- Partitioned traversal must classify each face into exactly one visible/hidden output bucket.

## Performance acceptance

For a fixture containing `F` unique faces instantiated `N` times, expect approximately:

- `mesh_helper_creates ~= F`, not `F*N`;
- `definition_cache_misses` proportional to unique definitions;
- `definition_cache_hits` increasing with repeated instances;
- `transformed_vertices` still proportional to expanded output, because flattened geometry must be emitted.

## Required fixture families

1. single unique mesh;
2. repeated component x100 and x1000;
3. nested repeated definitions;
4. mirrored instance;
5. non-uniform scale;
6. inherited material;
7. explicit front material;
8. two-sided material;
9. textured inherited material;
10. hidden entity;
11. hidden tag/layer;
12. nested hidden layer folder;
13. scene visibility overrides;
14. malformed/empty tessellation edge cases;
15. material names containing quotes and backslashes.

Executable tests should compare V2 output against `HierarchyReducerLegacy` for geometry/material/UV parity before V2 is treated as the sole implementation.
