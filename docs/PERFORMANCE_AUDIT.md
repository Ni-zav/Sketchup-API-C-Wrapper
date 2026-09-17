# SketchUp Wrapper Performance Audit

This document is the running inventory of performance findings discovered while tracing the importer from Blender through pybind into the SketchUp C API. It is intentionally separate from the V2 architecture document: this file records observations, evidence, risk, and follow-up work; `HIERARCHY_REDUCER_V2.md` describes the chosen implementation.

## Confirmed hot paths

### Repeated definition extraction

Legacy `HierarchyReducer` recursively re-enters a component definition for each instance occurrence. This repeats `entities.faces()`, material queries, `SUMeshHelperCreate`, vertex/normal/STQ extraction, and face processing for identical source geometry.

Impact: source extraction scales with expanded scene geometry rather than unique definition geometry.

V2 response: definition prototypes and tessellated faces are cached by entity identity while transforms/material inheritance/visibility remain occurrence-specific.

### Triangle-corner processing

Legacy reduction performs transform, normal handling, UV conversion, quantization and weld lookup for triangle corners. MeshHelper already provides indexed local vertices, so shared vertices receive repeated CPU work.

V2 response: process each MeshHelper vertex once per occurrence, build a local-to-output index map, then emit triangle indices by integer remapping.

### Visibility membership

Legacy scene override IDs are vectors searched repeatedly with `std::find`.

V2 response: compile override IDs into `unordered_set` at traversal start and cache effective layer/folder visibility by layer entity ID.

### Repeated transform derivation

Legacy computes determinant/inverse during face processing even though all faces at a hierarchy level share the same world transform.

V2 response: compute determinant, mirrored flag and inverse once per level.

### Back material work in one-sided mode

Legacy reads back material even when two-sided output is disabled.

V2 response: query/capture back material only for two-sided cache entries.

### Direct-front-material prescan

Legacy retained a collection-wide direct-front-material scan even though the current material resolution no longer consumes that result.

Response: V2 has no such scan; the Python fast builder also removes the equivalent obsolete pass.

### Loose-edge discovery

The wrapper `Entities::edges()` defaults to `stray_only=true`, which already maps to the SketchUp API's standalone-edge query. Older Python/C++ walker code rebuilt a set of all face-loop edge IDs to rediscover the same fact.

Response: V2 walker asks directly for stray edges and Python fallback now mirrors that behavior.

### Python hierarchy boundary volume

The standard importer builds definition meshes separately, then walks the whole hierarchy merely to place groups/instances. Yielding every face creates Python tuples, path lists, wrappers and NumPy matrices that are immediately discarded.

Response: `HierarchyWalkerV2` supports `include_faces=false` and the import runtime scopes this optimization to importer execution while leaving public `Model.walk()` defaults unchanged.

### Reduced-geometry binding copies

Returning STL maps through pybind can copy large `ReducedMesh` values. Reference-backed exposure is unsafe if Python outlives the reducer object.

Response: use an explicit single-snapshot conversion contract until stable buffer-view ownership is designed. Avoid hidden double copies, but never trade lifetime safety for zero-copy claims.

## Confirmed Python/Blender hot paths

- material-slot lookup in hierarchical mesh creation used linear list membership + `list.index`; now dict-backed;
- triangle validity checks were per-triangle Python; now vectorized with NumPy;
- definition faces were enumerated more than once while finding loose edges; now reused/direct stray-edge query is preferred;
- flattened chunk slicing repeatedly summed prefixes; now one cumulative-offset table is built;
- hierarchical meshes were welded per definition and then welded again scene-wide; the second pass is suppressed for the normal path;
- phase progress previously existed in NEOS but not standalone worker integration; standalone now emits monotonic, non-fatal callback progress.

## Complexity target

Let:

- `U` = geometry in unique definitions;
- `I` = hierarchy occurrences;
- `Vout`, `Tout` = expanded flattened output vertices/triangles;
- `L` = unique layers/folders relevant to visibility;
- `M` = materials.

Hierarchy-preserving target: approximately `O(U + I + L + M)` plus Blender object creation.

Flattened target: approximately `O(U + I + Vout + Tout + L + M)`. The expanded output term is unavoidable; repeated SketchUp SDK extraction is not.

## Still open

### Native definition batch for hierarchy-preserving mode

The hierarchy-preserving path still tessellates definitions through Python-facing face calls. The architectural end state is a native `DefinitionGeometryBatch`/prototype API shared by preserve and flatten modes.

### Placement batch

`HierarchyWalkerV2` still creates one Python tuple and one 4x4 NumPy array per yielded placement. A future `PlacementBatch` should return compact arrays of definition IDs, parent IDs, entity IDs, transforms and flags.

### Material key representation

Flattened reducer still uses string material keys and ordered `std::map`. Profile before replacing public keys; likely end state is numeric internal IDs with names encoded only at the Python boundary.

### Buffer representation

Native output currently stores doubles and exposes NumPy copies. Keep doubles during geometry math; consider float32 packed output at the final boundary and owner-backed buffer views only after lifetime contracts are explicit.

### Cleanup

Legacy dissolve/tri-to-quad cleanup remains correctness-sensitive. V2 routes requested cleanup through legacy behavior rather than weakening topology/UV/seam guarantees. A future cleanup implementation should be adjacency-driven and independently benchmarked.

### Parallelism

Do not assume concurrent access to one SketchUp model is safe. First isolate all C API extraction on one thread. Only pure CPU transformation/emission over immutable cached prototypes is a candidate for later parallelization.

## Audit rule

Every performance change must answer four questions:

1. Which repeated operation is removed or reduced?
2. What is the before/after complexity or constant-factor change?
3. Which SketchUp/Blender semantics could change?
4. Which counter, fixture or benchmark will prove the improvement without hiding a regression?

New findings discovered while implementing should be appended here before or with the code change that addresses them.
