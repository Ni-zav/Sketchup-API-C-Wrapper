# Hierarchy Reducer V2

## Purpose

`HierarchyReducerV2` is the performance-oriented, compatibility-safe successor
to the original `HierarchyReducer`. It keeps the existing `CleanupOptions` and
`ReducedMesh` output contracts so current importers can adopt it without a
format migration, while removing repeated SketchUp C API work from the expanded
instance hot path.

The legacy reducer remains available. V2 is intentionally additive until the
new path has passed Blender visual, visibility, material, UV, hierarchy and
round-trip regression testing.

## Why V2 exists

The original reducer recursively re-enters `definition.entities()` for every
component occurrence and recreates `SUMeshHelper` for every repeated face.
Architectural models commonly contain a small amount of unique definition
geometry instantiated hundreds or thousands of times, so the amount of SDK
extraction could scale with expanded scene geometry instead of unique source
geometry.

Other hot-path costs found during the audit:

- hidden entity/layer/folder membership used linear `std::find` over vectors;
- effective layer/folder visibility was recomputed for every drawing element;
- transformation determinant and inverse were recomputed per face;
- face back material was queried even in front-only mode;
- a now-unused direct-front-material pre-scan walked every face collection;
- the reducer transformed, normalized, UV-scaled, quantized and hash-looked-up
  each triangle corner even though `SUMeshHelper` already returns indexed
  vertices;
- final unit conversion required an additional full pass over output vertices;
- visible/hidden preservation required callers to perform two reducer passes.

## Implemented architecture

### Definition-level prototype cache

V2 caches each component/group definition level by definition entity ID. A
prototype stores the child instances/groups, direct faces, local transforms,
direct inherited-material state, raw hidden state and layer. Repeated component
occurrences therefore reuse the same SDK-derived entity lists.

The cache is independent of scene visibility because raw entity/layer state is
stored and effective visibility is resolved per traversal.

### Face tessellation cache

Each unique face is tessellated once per one-sided/two-sided mode. The cache
stores MeshHelper vertices, normals, triangle indices, STQ coordinates and
direct face materials. Repeated component instances reuse those buffers rather
than rebuilding `SUMeshHelper`.

The key contains both face entity ID and two-sided mode so a one-sided cache
entry never masquerades as a back-UV-capable entry.

### Per-level transform context

The world determinant and inverse transform are calculated once for all faces
at the current hierarchy level. Mirrored winding is therefore determined once
per level rather than once per face.

### Per-vertex, not per-corner work

For each occurrence, every MeshHelper local vertex is transformed, normalised,
UV-scaled and welded exactly once. Triangle emission then becomes integer index
remapping. This removes repeated transform/hash work for vertices shared by
multiple triangles.

### O(1)-average visibility membership

`CleanupOptions` remains vector-based for ABI/Python compatibility, but V2
compiles hidden entity IDs, layer overrides and hidden folder IDs into
`std::unordered_set` at traversal start. Effective layer visibility is cached
per layer for the duration of the traversal.

### Direct final-unit storage

Vertex welding still hashes world positions in SketchUp internal units so the
existing weld tolerance is preserved. A newly inserted output vertex is stored
already multiplied by `unit_scale`, removing the legacy final O(V) scaling
pass.

### One-pass visible/hidden partitioning

`traverse_partitioned()` fills `get_reduced_geometry()` with effectively visible
geometry and `get_hidden_reduced_geometry()` with effectively hidden geometry in
one hierarchy walk. This API is intended to replace the current importer pattern
of traversing the same model twice for `PRESERVE_HIDDEN`.

### Legacy cleanup fallback

The legacy limited-dissolve / tris-to-quads implementation contains topology,
UV and seam guards that must not be silently weakened. Until cleanup V2 is
validated, requesting either cleanup option through `traverse()` routes the
operation through the legacy reducer. `traverse_partitioned()` rejects cleanup
because a hidden/visible split followed by topology cleanup needs an explicit
contract.

## Complexity target

Let:

- `U` be geometry in unique definitions;
- `I` be instance/group occurrences;
- `Vout`/`Tout` be expanded flattened output vertices/triangles;
- `L` be unique layers.

The target flattened path is approximately:

`O(U + I + Vout + Tout + L)`

The unavoidable `Vout + Tout` term remains because flattened output itself must
be emitted. SketchUp tessellation and definition enumeration should scale with
unique source geometry, not repeated occurrences.

## Native performance counters

`ReducerStatsV2` records:

- definition cache hits/misses;
- faces seen/emitted/skipped by visibility;
- MeshHelper creates;
- tessellation cache hits/misses;
- transformed vertex count;
- emitted triangle count;
- layer visibility cache hits/misses;
- instance/group/level counts; and
- whether the legacy cleanup fallback was used.

Import benchmarks should treat `mesh_helper_creates` as a first-class metric.
For a 100-face component instantiated 1,000 times, the desired order of
magnitude is about 100 MeshHelper creations rather than 100,000.

## Correctness invariants

Performance changes must preserve:

- inherited front materials;
- optional two-sided front/back material and UV behavior;
- mirrored winding;
- inverse-transpose-style normal handling used by the previous reducer;
- weld tolerance semantics;
- layer, folder and scene hidden overrides;
- ancestor visibility;
- deterministic material bucket names;
- valid triangle/index buffers; and
- read-only model lifetime assumptions.

## Cache lifetime

Definition/tessellation caches intentionally survive multiple traversals on the
same `HierarchyReducerV2` instance. They are valid only while the source model
is alive and unchanged. Call `reset_geometry_cache()` after mutating a model.

## Remaining work

The V2 reducer deliberately does not claim the entire importer is optimal yet.
The next layers are:

1. switch importer `PRESERVE_HIDDEN` to `traverse_partitioned()`;
2. expose reduced geometry through reference-backed/buffer views instead of STL
   map conversion and repeated NumPy copies;
3. provide a placement-only bulk hierarchy API for hierarchy-preserving import;
4. move hierarchy-preserving definition geometry extraction to the same native
   prototype compiler instead of per-face Python/pybind calls;
5. redesign native cleanup around explicit edge adjacency and then remove the
   legacy fallback;
6. benchmark material-key string/map costs before replacing public names with
   numeric internal material IDs; and
7. only consider parallel CPU expansion after all SketchUp API extraction is
   isolated from worker threads and thread-safety assumptions are proven.
