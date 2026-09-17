#ifndef OptimizationV2Types_hpp
#define OptimizationV2Types_hpp

#include "SUAPI-CppWrapper/Transformation.hpp"
#include "SUAPI-CppWrapper/model/ComponentInstance.hpp"
#include "SUAPI-CppWrapper/model/Entities.hpp"
#include "SUAPI-CppWrapper/model/Face.hpp"
#include "SUAPI-CppWrapper/model/Group.hpp"
#include "SUAPI-CppWrapper/model/Layer.hpp"
#include "SUAPI-CppWrapper/model/Material.hpp"

#include <SketchUpAPI/geometry.h>

#include <cstdint>
#include <string>
#include <vector>

namespace CW {

// Public counters remain in the wrapper ABI. Keeping them in a dedicated header
// lets bindings and benchmarks depend on the contract without pulling in the
// full reducer implementation.
struct ReducerStatsV2 {
  std::uint64_t entity_levels = 0;
  std::uint64_t root_cache_hits = 0;
  std::uint64_t root_cache_misses = 0;
  std::uint64_t definition_cache_hits = 0;
  std::uint64_t definition_cache_misses = 0;
  std::uint64_t instances = 0;
  std::uint64_t groups = 0;
  std::uint64_t faces_seen = 0;
  std::uint64_t faces_emitted = 0;
  std::uint64_t faces_visibility_skipped = 0;
  std::uint64_t mesh_helper_creates = 0;
  std::uint64_t tessellation_cache_hits = 0;
  std::uint64_t tessellation_cache_misses = 0;
  std::uint64_t transformed_vertices = 0;
  std::uint64_t triangles_emitted = 0;
  std::uint64_t layer_visibility_cache_hits = 0;
  std::uint64_t layer_visibility_cache_misses = 0;
  std::uint64_t material_scale_cache_builds = 0;
  bool used_legacy_cleanup_fallback = false;
};

namespace detail {

struct ReducerMaterialStateV2 {
  Material material;
  std::string name = "SketchUp_Default";
  bool valid = false;
};

struct ReducerFaceEntryV2 {
  Face face;
  std::int32_t entity_id = 0;
  bool raw_hidden = false;
  Layer layer;
};

struct ReducerInstanceEntryV2 {
  ComponentInstance instance;
  std::int32_t entity_id = 0;
  std::int32_t definition_id = 0;
  Entities child_entities;
  Transformation local_transform;
  ReducerMaterialStateV2 direct_material;
  bool raw_hidden = false;
  Layer layer;
};

struct ReducerGroupEntryV2 {
  Group group;
  std::int32_t entity_id = 0;
  std::int32_t definition_id = 0;
  Entities child_entities;
  Transformation local_transform;
  ReducerMaterialStateV2 direct_material;
  bool raw_hidden = false;
  Layer layer;
};

struct ReducerEntityLevelPrototypeV2 {
  std::vector<ReducerInstanceEntryV2> instances;
  std::vector<ReducerGroupEntryV2> groups;
  std::vector<ReducerFaceEntryV2> faces;
};

struct ReducerCachedFaceGeometryV2 {
  std::vector<SUPoint3D> vertices;
  std::vector<SUVector3D> normals;
  std::vector<std::size_t> indices;
  std::vector<SUPoint3D> front_stq;
  std::vector<SUPoint3D> back_stq;
  ReducerMaterialStateV2 front_material;
  ReducerMaterialStateV2 back_material;
  bool valid = false;
  bool has_back_stq = false;
};

} // namespace detail
} // namespace CW

#endif // OptimizationV2Types_hpp
