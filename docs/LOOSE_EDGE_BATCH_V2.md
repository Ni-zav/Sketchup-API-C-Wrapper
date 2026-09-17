# LooseEdgeBatchV2

`LooseEdgeBatchV2` is the native standalone-edge companion to `HierarchyReducerV2`.

Its purpose is to remove Python hierarchy walking from flattened wire import while preserving the metadata and visibility semantics required by the Blender importer.

## Why it exists

The previous flattened importer collected standalone edges through `Model.walk()` after geometry reduction. For `PRESERVE_HIDDEN` it walked the hierarchy once for visible edges and again for hidden edges.

That path also repeated source calls for repeated component definitions:

- `entities.edges(true)` / edge wrapper materialization;
- `edge.start()` and `edge.end()`;
- soft/smooth flags;
- hidden/layer metadata;
- layer-name lookup;
- Python path list and NumPy transform creation.

The wrapper already knows that `Entities::edges(true)` maps directly to SketchUp's stray-edge query, so standalone wires do not require scanning face loops.

## V2 model

For each unique definition, the collector caches a local prototype containing:

- standalone edge local start/end points;
- edge entity ID;
- raw hidden flag;
- soft/smooth flags;
- layer ID/name/reference;
- nested group/instance child references and local transforms.

Repeated occurrences then perform only occurrence-specific work:

1. accumulate the world transform;
2. evaluate ancestor/entity/layer visibility;
3. transform the two cached local endpoints;
4. scale to the requested output units;
5. emit the existing metadata payload.

This keeps source extraction tied to unique definitions while output expansion remains tied to actual occurrences.

## Partitioned collection

`collect_partitioned()` writes effectively visible and effectively hidden edges in one hierarchy traversal.

The importer can therefore request:

- `get_edges()` for visible output;
- `get_hidden_edges()` for the excluded hidden collection;

without running a second Python/native hierarchy walk.

`collect()` remains available for `ALL`, `VISIBLE_ONLY`, or compatibility uses controlled by `CleanupOptions.visibility_filter`.

## Metadata contract

Each record preserves:

- two world-space output-unit endpoints;
- hierarchy path;
- source edge entity ID;
- raw hidden flag;
- effective visibility flag;
- soft/smooth flags;
- layer ID and layer name when available.

The importer binding converts this directly to the historical edge dictionary shape, so `flattened_builder` and `SOURCE_EDGE_PAYLOAD_PROP` do not need a format migration.

## Counters

`LooseEdgeStatsV2` reports:

- hierarchy levels;
- definition cache hits/misses;
- group/instance occurrences;
- edges seen/emitted/visibility-skipped;
- layer visibility cache hits/misses.

For a repeated-component benchmark, definition-cache misses should stay near the number of unique edge-bearing definitions while hits grow with occurrence count.

## Correctness constraints

The optimized path must preserve:

- raw hidden metadata independently of effective visibility;
- scene hidden-object overrides;
- tag/layer visibility and layer-folder visibility;
- ancestor visibility propagation;
- mirrored/scaled/transformed endpoint positions;
- hierarchy path naming compatible with the existing walker;
- source edge IDs and soft/smooth metadata.

## Current design debt: duplicated visibility evaluator

`HierarchyReducerV2` and `LooseEdgeBatchV2` currently contain parallel implementations of scene/entity/layer/folder visibility evaluation. This was intentionally accepted for the first additive wire slice so geometry-reducer code did not need another invasive rewrite before the new edge ABI existed.

It is now a refactor target, not a desired final state.

Next native cleanup:

1. extract a shared `TraversalVisibilityV2`/visibility-state helper;
2. make reducer V2, loose-edge V2, and future placement batching use it;
3. add parity tests for layer-folder errors, scene overrides, raw hidden flags, and ancestor visibility;
4. remove duplicated visibility code only after both callers use the shared helper.

## Threading

The collector is intentionally single-threaded while touching SketchUp model/entity APIs. As with reducer V2, only pure CPU work over immutable cached prototypes should be considered for later parallelization after API extraction is complete.
