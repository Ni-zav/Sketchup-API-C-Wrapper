#ifndef OptimizationV2_hpp
#define OptimizationV2_hpp

#include "SUAPI-CppWrapper/model/Optimization.hpp"
#include "SUAPI-CppWrapper/model/TraversalVisibilityV2.hpp"
#include "SUAPI-CppWrapper/model/detail/OptimizationV2Types.hpp"

#include <SketchUpAPI/model/mesh_helper.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace CW {

// HierarchyReducerV2 keeps the legacy ReducedMesh/CleanupOptions ABI while
// moving repeated SketchUp API work out of the expanded-instance hot path.
//
// Implementation is intentionally split under model/detail/ so traversal,
// caching/tessellation, material math and emission can evolve independently.
// The referenced Model must remain alive and must not be mutated while this
// reducer is in use. Call reset_geometry_cache() after a model mutation.
class HierarchyReducerV2 {
public:
  explicit HierarchyReducerV2(Model &model) : m_model(model) {}

  void traverse(const CleanupOptions &options = CleanupOptions()) {
    if (options.limited_dissolve || options.tris_to_quads) {
      reset_run_state();
      HierarchyReducer legacy(m_model);
      legacy.traverse(options);
      m_buckets = legacy.get_reduced_geometry();
      m_stats.used_legacy_cleanup_fallback = true;
      return;
    }

    begin_run(options, false);
    process_level(root_level(), Transformation(), default_material_state(),
                  options, 0, true);
    sync_visibility_stats();
  }

  void traverse_entities(const Entities &entities,
                         const CleanupOptions &options = CleanupOptions()) {
    if (options.limited_dissolve || options.tris_to_quads) {
      reset_run_state();
      HierarchyReducer legacy(m_model);
      legacy.traverse_entities(entities, options);
      m_buckets = legacy.get_reduced_geometry();
      m_stats.used_legacy_cleanup_fallback = true;
      return;
    }

    begin_run(options, false);
    const EntityLevelPrototype root = build_level(entities);
    process_level(root, Transformation(), default_material_state(), options, 0,
                  true);
    sync_visibility_stats();
  }

  // One hierarchy walk classifies both effectively-visible and effectively-
  // hidden geometry for PRESERVE_HIDDEN imports.
  void traverse_partitioned(const CleanupOptions &options = CleanupOptions()) {
    if (options.limited_dissolve || options.tris_to_quads) {
      throw std::logic_error(
          "HierarchyReducerV2::traverse_partitioned does not support topology "
          "cleanup; run cleanup after partitioning or use traverse().");
    }

    begin_run(options, true);
    process_level(root_level(), Transformation(), default_material_state(),
                  options, 0, true);
    sync_visibility_stats();
  }

  const std::map<std::string, ReducedMesh> &get_reduced_geometry() const {
    return m_buckets;
  }

  const std::map<std::string, ReducedMesh> &
  get_hidden_reduced_geometry() const {
    return m_hidden_buckets;
  }

  const ReducerStatsV2 &stats() const { return m_stats; }

  // Immutable source caches survive normal traversals so different visibility
  // configurations can reuse SketchUp extraction. Clear after model mutation.
  void reset_geometry_cache() {
    m_root_cache.reset();
    m_definition_cache.clear();
    m_face_cache.clear();
    m_texture_scale_cache.clear();
    m_texture_scale_cache_ready = false;
  }

private:
  using MaterialState = detail::ReducerMaterialStateV2;
  using FaceEntry = detail::ReducerFaceEntryV2;
  using InstanceEntry = detail::ReducerInstanceEntryV2;
  using GroupEntry = detail::ReducerGroupEntryV2;
  using EntityLevelPrototype = detail::ReducerEntityLevelPrototypeV2;
  using CachedFaceGeometry = detail::ReducerCachedFaceGeometryV2;

  Model &m_model;
  std::map<std::string, ReducedMesh> m_buckets;
  std::map<std::string, ReducedMesh> m_hidden_buckets;

  // Source/model caches: persist across compatible traversals.
  std::optional<EntityLevelPrototype> m_root_cache;
  std::unordered_map<std::int32_t, EntityLevelPrototype> m_definition_cache;
  std::unordered_map<std::uint64_t, CachedFaceGeometry> m_face_cache;
  std::unordered_map<std::string, std::pair<double, double>>
      m_texture_scale_cache;
  bool m_texture_scale_cache_ready = false;

  // Per-run state.
  TraversalVisibilityV2 m_visibility;
  ReducerStatsV2 m_stats;
  bool m_partition_output = false;

#include "SUAPI-CppWrapper/model/detail/OptimizationV2Material.inl"
#include "SUAPI-CppWrapper/model/detail/OptimizationV2Cache.inl"
#include "SUAPI-CppWrapper/model/detail/OptimizationV2Emit.inl"
#include "SUAPI-CppWrapper/model/detail/OptimizationV2Traversal.inl"
};

} // namespace CW

#endif // OptimizationV2_hpp
