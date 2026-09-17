#ifndef OptimizationV2_hpp
#define OptimizationV2_hpp

#include "SUAPI-CppWrapper/model/Optimization.hpp"
#include "SUAPI-CppWrapper/model/ComponentDefinition.hpp"
#include "SUAPI-CppWrapper/model/ComponentInstance.hpp"
#include "SUAPI-CppWrapper/model/DrawingElement.hpp"
#include "SUAPI-CppWrapper/model/Group.hpp"
#include "SUAPI-CppWrapper/model/Layer.hpp"

#include <SketchUpAPI/model/layer.h>
#include <SketchUpAPI/model/layer_folder.h>
#include <SketchUpAPI/model/mesh_helper.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace CW {

// Performance counters are deliberately part of the native reducer contract so
// importers can prove where time/SDK work went instead of relying on wall-clock
// time alone.
struct ReducerStatsV2 {
  std::uint64_t entity_levels = 0;
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
  bool used_legacy_cleanup_fallback = false;
};

// HierarchyReducerV2 keeps the legacy ReducedMesh/CleanupOptions ABI, but moves
// repeated SketchUp API work out of the expanded-instance hot path.
//
// Important lifetime rule: the referenced Model must remain alive and must not
// be mutated while this reducer is in use. Call reset_geometry_cache() after a
// model mutation before traversing again.
class HierarchyReducerV2 {
public:
  explicit HierarchyReducerV2(Model &model) : m_model(model) {}

  void traverse(const CleanupOptions &options = CleanupOptions()) {
    if (options.limited_dissolve || options.tris_to_quads) {
      // The legacy cleanup implementation contains topology-preservation logic
      // that has not yet been reimplemented in V2. Preserve correctness by
      // routing explicitly requested cleanup through the legacy reducer.
      reset_run_state();
      HierarchyReducer legacy(m_model);
      legacy.traverse(options);
      m_buckets = legacy.get_reduced_geometry();
      m_stats.used_legacy_cleanup_fallback = true;
      return;
    }

    begin_run(options, false);
    const EntityLevelPrototype root = build_level(m_model.entities());
    process_level(root, Transformation(), default_material_state(), options, 0,
                  true);
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
  }

  // One hierarchy walk can classify both effectively-visible and effectively-
  // hidden geometry. This is intended for PRESERVE_HIDDEN importers that would
  // otherwise traverse/tessellate the same model twice.
  void traverse_partitioned(const CleanupOptions &options = CleanupOptions()) {
    if (options.limited_dissolve || options.tris_to_quads) {
      throw std::logic_error(
          "HierarchyReducerV2::traverse_partitioned does not support topology "
          "cleanup; run cleanup after partitioning or use traverse().");
    }

    begin_run(options, true);
    const EntityLevelPrototype root = build_level(m_model.entities());
    process_level(root, Transformation(), default_material_state(), options, 0,
                  true);
  }

  const std::map<std::string, ReducedMesh> &get_reduced_geometry() const {
    return m_buckets;
  }

  const std::map<std::string, ReducedMesh> &
  get_hidden_reduced_geometry() const {
    return m_hidden_buckets;
  }

  const ReducerStatsV2 &stats() const { return m_stats; }

  // Definition and tessellation caches intentionally survive normal traversals
  // so different visibility passes on the same read-only model can reuse SDK
  // extraction. Explicitly clear them after any model mutation.
  void reset_geometry_cache() {
    m_definition_cache.clear();
    m_face_cache.clear();
  }

private:
  struct MaterialState {
    Material material;
    std::string name = "SketchUp_Default";
    bool valid = false;
  };

  struct FaceEntry {
    Face face;
    std::int32_t entity_id = 0;
    bool raw_hidden = false;
    Layer layer;
  };

  struct InstanceEntry {
    ComponentInstance instance;
    std::int32_t entity_id = 0;
    std::int32_t definition_id = 0;
    Entities child_entities;
    Transformation local_transform;
    MaterialState direct_material;
    bool raw_hidden = false;
    Layer layer;
  };

  struct GroupEntry {
    Group group;
    std::int32_t entity_id = 0;
    std::int32_t definition_id = 0;
    Entities child_entities;
    Transformation local_transform;
    MaterialState direct_material;
    bool raw_hidden = false;
    Layer layer;
  };

  struct EntityLevelPrototype {
    std::vector<InstanceEntry> instances;
    std::vector<GroupEntry> groups;
    std::vector<FaceEntry> faces;
  };

  struct CachedFaceGeometry {
    std::vector<SUPoint3D> vertices;
    std::vector<SUVector3D> normals;
    std::vector<std::size_t> indices;
    std::vector<SUPoint3D> front_stq;
    std::vector<SUPoint3D> back_stq;
    MaterialState front_material;
    MaterialState back_material;
    bool has_back_stq = false;
  };

  Model &m_model;
  std::map<std::string, ReducedMesh> m_buckets;
  std::map<std::string, ReducedMesh> m_hidden_buckets;
  std::unordered_map<std::int32_t, EntityLevelPrototype> m_definition_cache;
  std::unordered_map<std::uint64_t, CachedFaceGeometry> m_face_cache;
  std::unordered_map<std::string, std::pair<double, double>>
      m_texture_scale_cache;
  std::unordered_set<std::int32_t> m_hidden_entity_ids;
  std::unordered_set<std::int32_t> m_layer_override_ids;
  std::unordered_set<std::int32_t> m_hidden_layer_folder_ids;
  std::unordered_map<std::int32_t, bool> m_layer_visibility_cache;
  ReducerStatsV2 m_stats;
  bool m_partition_output = false;

  static MaterialState default_material_state() { return MaterialState(); }

  static MaterialState material_state(const Material &material) {
    MaterialState state;
    state.material = material;
    state.valid = material.is_valid();
    if (state.valid) {
      state.name = material.name().std_string();
      if (state.name.empty())
        state.name = "SketchUp_Default";
    }
    return state;
  }

  void reset_run_state() {
    m_buckets.clear();
    m_hidden_buckets.clear();
    m_texture_scale_cache.clear();
    m_hidden_entity_ids.clear();
    m_layer_override_ids.clear();
    m_hidden_layer_folder_ids.clear();
    m_layer_visibility_cache.clear();
    m_stats = ReducerStatsV2();
    m_partition_output = false;
  }

  void begin_run(const CleanupOptions &options, bool partition_output) {
    reset_run_state();
    m_partition_output = partition_output;

    m_hidden_entity_ids.reserve(options.hidden_entity_ids.size());
    m_hidden_entity_ids.insert(options.hidden_entity_ids.begin(),
                               options.hidden_entity_ids.end());
    m_layer_override_ids.reserve(options.layer_override_ids.size());
    m_layer_override_ids.insert(options.layer_override_ids.begin(),
                                options.layer_override_ids.end());
    m_hidden_layer_folder_ids.reserve(options.hidden_layer_folder_ids.size());
    m_hidden_layer_folder_ids.insert(options.hidden_layer_folder_ids.begin(),
                                     options.hidden_layer_folder_ids.end());

    const std::vector<Material> materials = m_model.materials();
    m_texture_scale_cache.reserve(materials.size());
    for (const auto &material : materials) {
      if (!material.is_valid())
        continue;
      try {
        const Texture texture = material.texture();
        if (texture.is_valid()) {
          m_texture_scale_cache.emplace(
              material.name().std_string(),
              std::make_pair(texture.s_scale(), texture.t_scale()));
        }
      } catch (...) {
        // Texture metadata is optional. Geometry import must continue even if a
        // particular material cannot expose a usable texture object.
      }
    }
  }

  EntityLevelPrototype build_level(const Entities &entities) {
    EntityLevelPrototype level;

    const std::vector<ComponentInstance> instances = entities.instances();
    level.instances.reserve(instances.size());
    for (const auto &instance : instances) {
      const ComponentDefinition definition = instance.definition();
      level.instances.push_back(InstanceEntry{
          instance,
          static_cast<std::int32_t>(instance.entityID()),
          static_cast<std::int32_t>(definition.entityID()),
          definition.entities(),
          instance.transformation(),
          material_state(instance.material()),
          instance.hidden(),
          instance.layer(),
      });
    }

    const std::vector<Group> groups = entities.groups();
    level.groups.reserve(groups.size());
    for (const auto &group : groups) {
      const ComponentDefinition definition = group.definition();
      level.groups.push_back(GroupEntry{
          group,
          static_cast<std::int32_t>(group.entityID()),
          static_cast<std::int32_t>(definition.entityID()),
          group.entities(),
          group.transformation(),
          material_state(group.material()),
          group.hidden(),
          group.layer(),
      });
    }

    const std::vector<Face> faces = entities.faces();
    level.faces.reserve(faces.size());
    for (const auto &face : faces) {
      level.faces.push_back(FaceEntry{
          face,
          static_cast<std::int32_t>(face.entityID()),
          face.hidden(),
          face.layer(),
      });
    }

    return level;
  }

  const EntityLevelPrototype &definition_level(std::int32_t definition_id,
                                                const Entities &entities) {
    auto existing = m_definition_cache.find(definition_id);
    if (existing != m_definition_cache.end()) {
      ++m_stats.definition_cache_hits;
      return existing->second;
    }

    ++m_stats.definition_cache_misses;
    auto inserted = m_definition_cache.emplace(definition_id, build_level(entities));
    return inserted.first->second;
  }

  bool layer_visible(const Layer &layer, const CleanupOptions &options) {
    if (!layer.is_valid())
      return true;

    const std::int32_t layer_id =
        static_cast<std::int32_t>(layer.entityID());
    auto cached = m_layer_visibility_cache.find(layer_id);
    if (cached != m_layer_visibility_cache.end()) {
      ++m_stats.layer_visibility_cache_hits;
      return cached->second;
    }
    ++m_stats.layer_visibility_cache_misses;

    bool visible = true;
    if (SULayerGetVisibility(layer.ref(), &visible) != SU_ERROR_NONE)
      visible = true;

    if (options.use_scene_hidden_layers &&
        m_layer_override_ids.find(layer_id) != m_layer_override_ids.end()) {
      visible = !visible;
    }

    if (visible) {
      SULayerFolderRef folder = SU_INVALID;
      SUResult result = SULayerGetParentLayerFolder(layer.ref(), &folder);
      int guard = 0;
      while (result == SU_ERROR_NONE && SUIsValid(folder) && guard++ < 128) {
        bool folder_visible = true;
        if (SULayerFolderGetVisibility(folder, &folder_visible) ==
                SU_ERROR_NONE &&
            !folder_visible) {
          visible = false;
          break;
        }

        if (options.use_scene_hidden_layers) {
          std::int32_t folder_id = 0;
          const SUEntityRef folder_entity = SULayerFolderToEntity(folder);
          if (SUEntityGetID(folder_entity, &folder_id) == SU_ERROR_NONE &&
              m_hidden_layer_folder_ids.find(folder_id) !=
                  m_hidden_layer_folder_ids.end()) {
            visible = false;
            break;
          }
        }

        SULayerFolderRef parent = SU_INVALID;
        result = SULayerFolderGetParentLayerFolder(folder, &parent);
        folder = parent;
      }
    }

    m_layer_visibility_cache.emplace(layer_id, visible);
    return visible;
  }

  bool drawing_element_visible(std::int32_t entity_id, bool raw_hidden,
                               const Layer &layer,
                               const CleanupOptions &options) {
    const bool hidden =
        options.use_scene_hidden_objects
            ? (m_hidden_entity_ids.find(entity_id) != m_hidden_entity_ids.end())
            : raw_hidden;
    return !hidden && layer_visible(layer, options);
  }

  bool include_for_visibility(bool effective_visible,
                              const CleanupOptions &options) const {
    if (m_partition_output)
      return true;
    if (options.visibility_filter == 1)
      return effective_visible;
    if (options.visibility_filter == 2)
      return !effective_visible;
    return true;
  }

  std::map<std::string, ReducedMesh> &target_buckets(bool effective_visible) {
    if (m_partition_output && !effective_visible)
      return m_hidden_buckets;
    return m_buckets;
  }

  static std::uint64_t face_cache_key(std::int32_t face_id,
                                      bool two_sided_materials) {
    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(face_id))
            << 1) |
           (two_sided_materials ? 1ULL : 0ULL);
  }

  const CachedFaceGeometry *load_face_geometry(const FaceEntry &entry,
                                                bool two_sided_materials) {
    const std::uint64_t key =
        face_cache_key(entry.entity_id, two_sided_materials);
    auto cached = m_face_cache.find(key);
    if (cached != m_face_cache.end()) {
      ++m_stats.tessellation_cache_hits;
      return &cached->second;
    }

    ++m_stats.tessellation_cache_misses;
    CachedFaceGeometry geometry;
    geometry.front_material = material_state(entry.face.material());
    if (two_sided_materials)
      geometry.back_material = material_state(entry.face.back_material());

    SUMeshHelperRef mesh_ref = SU_INVALID;
    if (SUMeshHelperCreate(&mesh_ref, entry.face.ref()) != SU_ERROR_NONE)
      return nullptr;
    ++m_stats.mesh_helper_creates;

    std::size_t num_vertices = 0;
    std::size_t num_triangles = 0;
    const SUResult vertex_count_result =
        SUMeshHelperGetNumVertices(mesh_ref, &num_vertices);
    const SUResult triangle_count_result =
        SUMeshHelperGetNumTriangles(mesh_ref, &num_triangles);
    if (vertex_count_result != SU_ERROR_NONE ||
        triangle_count_result != SU_ERROR_NONE || num_vertices == 0 ||
        num_triangles == 0) {
      SUMeshHelperRelease(&mesh_ref);
      auto inserted = m_face_cache.emplace(key, std::move(geometry));
      return &inserted.first->second;
    }

    geometry.vertices.resize(num_vertices);
    geometry.normals.resize(num_vertices, SUVector3D{0.0, 0.0, 1.0});
    geometry.indices.resize(num_triangles * 3);
    geometry.front_stq.resize(num_vertices, SUPoint3D{0.0, 0.0, 1.0});

    std::size_t vertex_count = 0;
    std::size_t normal_count = 0;
    std::size_t index_count = 0;
    const SUResult vertices_result = SUMeshHelperGetVertices(
        mesh_ref, num_vertices, geometry.vertices.data(), &vertex_count);
    const SUResult normals_result = SUMeshHelperGetNormals(
        mesh_ref, num_vertices, geometry.normals.data(), &normal_count);
    const SUResult indices_result = SUMeshHelperGetVertexIndices(
        mesh_ref, num_triangles * 3, geometry.indices.data(), &index_count);

    if (vertices_result != SU_ERROR_NONE || indices_result != SU_ERROR_NONE ||
        vertex_count != num_vertices || index_count < 3) {
      SUMeshHelperRelease(&mesh_ref);
      return nullptr;
    }

    if (normals_result != SU_ERROR_NONE || normal_count != num_vertices) {
      std::fill(geometry.normals.begin(), geometry.normals.end(),
                SUVector3D{0.0, 0.0, 1.0});
    }

    index_count -= index_count % 3;
    geometry.indices.resize(index_count);

    std::size_t front_stq_count = 0;
    if (SUMeshHelperGetFrontSTQCoords(mesh_ref, num_vertices,
                                      geometry.front_stq.data(),
                                      &front_stq_count) != SU_ERROR_NONE ||
        front_stq_count != num_vertices) {
      std::fill(geometry.front_stq.begin(), geometry.front_stq.end(),
                SUPoint3D{0.0, 0.0, 1.0});
    }

    if (two_sided_materials) {
      geometry.back_stq.resize(num_vertices, SUPoint3D{0.0, 0.0, 1.0});
      std::size_t back_stq_count = 0;
      geometry.has_back_stq =
          SUMeshHelperGetBackSTQCoords(mesh_ref, num_vertices,
                                       geometry.back_stq.data(),
                                       &back_stq_count) == SU_ERROR_NONE &&
          back_stq_count == num_vertices;
      if (!geometry.has_back_stq) {
        std::fill(geometry.back_stq.begin(), geometry.back_stq.end(),
                  SUPoint3D{0.0, 0.0, 1.0});
      }
    }

    SUMeshHelperRelease(&mesh_ref);
    auto inserted = m_face_cache.emplace(key, std::move(geometry));
    return &inserted.first->second;
  }

  static void reserve_append(ReducedMesh &mesh, std::size_t vertices,
                             std::size_t indices) {
    const std::size_t desired_vertices = mesh.vertices.size() + vertices;
    if (mesh.vertices.capacity() < desired_vertices) {
      const std::size_t grown =
          std::max(desired_vertices, mesh.vertices.capacity() * 2 + 64);
      mesh.vertices.reserve(grown);
      mesh.normals.reserve(grown);
      mesh.uvs.reserve(grown);
      mesh.back_uvs.reserve(grown);
    }

    const std::size_t desired_indices = mesh.indices.size() + indices;
    if (mesh.indices.capacity() < desired_indices) {
      mesh.indices.reserve(
          std::max(desired_indices, mesh.indices.capacity() * 2 + 192));
    }

    const std::size_t desired_faces = mesh.face_sizes.size() + indices / 3;
    if (mesh.face_sizes.capacity() < desired_faces) {
      mesh.face_sizes.reserve(
          std::max(desired_faces, mesh.face_sizes.capacity() * 2 + 64));
    }

    const std::size_t desired_unique = mesh.unique_map.size() + vertices;
    const float load_factor = mesh.unique_map.max_load_factor();
    const std::size_t hash_capacity = static_cast<std::size_t>(
        static_cast<double>(mesh.unique_map.bucket_count()) * load_factor);
    if (desired_unique > hash_capacity) {
      mesh.unique_map.reserve(
          std::max(desired_unique, hash_capacity * 2 + 64));
    }
  }

  static int32_t find_or_add_vertex(ReducedMesh &mesh, const SUPoint3D &pos,
                                    const SUVector3D &norm,
                                    const SUPoint2D &uv,
                                    const SUPoint2D *back_uv,
                                    double unit_scale) {
    const int64_t kx =
        static_cast<int64_t>(std::round(pos.x * ReducedMesh::POS_SCALE));
    const int64_t ky =
        static_cast<int64_t>(std::round(pos.y * ReducedMesh::POS_SCALE));
    const int64_t kz =
        static_cast<int64_t>(std::round(pos.z * ReducedMesh::POS_SCALE));
    const int32_t knx =
        static_cast<int32_t>(std::round(norm.x * ReducedMesh::NORMAL_SCALE));
    const int32_t kny =
        static_cast<int32_t>(std::round(norm.y * ReducedMesh::NORMAL_SCALE));
    const int32_t knz =
        static_cast<int32_t>(std::round(norm.z * ReducedMesh::NORMAL_SCALE));
    const int64_t ku =
        static_cast<int64_t>(std::round(uv.x * ReducedMesh::UV_SCALE));
    const int64_t kv =
        static_cast<int64_t>(std::round(uv.y * ReducedMesh::UV_SCALE));

    int64_t kbu = std::numeric_limits<int64_t>::min();
    int64_t kbv = std::numeric_limits<int64_t>::min();
    if (back_uv != nullptr) {
      kbu = static_cast<int64_t>(
          std::round(back_uv->x * ReducedMesh::UV_SCALE));
      kbv = static_cast<int64_t>(
          std::round(back_uv->y * ReducedMesh::UV_SCALE));
    }

    const ReducedMesh::VertexKey key =
        std::make_tuple(kx, ky, kz, knx, kny, knz, ku, kv, kbu, kbv);
    const auto existing = mesh.unique_map.find(key);
    if (existing != mesh.unique_map.end())
      return existing->second;

    const int32_t index = static_cast<int32_t>(mesh.vertices.size());
    SUPoint3D stored_position = pos;
    stored_position.x *= unit_scale;
    stored_position.y *= unit_scale;
    stored_position.z *= unit_scale;
    mesh.vertices.push_back(stored_position);
    mesh.normals.push_back(norm);
    mesh.uvs.push_back(uv);
    if (back_uv != nullptr)
      mesh.back_uvs.push_back(*back_uv);
    mesh.unique_map.emplace(key, index);
    return index;
  }

  std::pair<double, double> texture_scale(const MaterialState &material) const {
    if (!material.valid)
      return {1.0, 1.0};
    const auto found = m_texture_scale_cache.find(material.name);
    if (found == m_texture_scale_cache.end())
      return {1.0, 1.0};
    return found->second;
  }

  static SUVector3D transformed_normal(const SUVector3D &normal,
                                       const Transformation &world,
                                       const Transformation &inverse,
                                       bool has_inverse) {
    SUVector3D transformed;
    if (has_inverse) {
      transformed = {
          inverse[0] * normal.x + inverse[1] * normal.y + inverse[2] * normal.z,
          inverse[4] * normal.x + inverse[5] * normal.y + inverse[6] * normal.z,
          inverse[8] * normal.x + inverse[9] * normal.y + inverse[10] * normal.z,
      };
    } else {
      transformed = world * CW::Vector3D(normal);
    }

    const double length_squared = transformed.x * transformed.x +
                                  transformed.y * transformed.y +
                                  transformed.z * transformed.z;
    if (length_squared > 1e-12) {
      const double inv_length = 1.0 / std::sqrt(length_squared);
      transformed.x *= inv_length;
      transformed.y *= inv_length;
      transformed.z *= inv_length;
    } else {
      transformed = {0.0, 0.0, 1.0};
    }
    return transformed;
  }

  static SUPoint2D scaled_uv(const SUPoint3D &stq, double s_scale,
                             double t_scale) {
    const double q = std::fabs(stq.z) > 1e-20 ? stq.z : 1.0;
    return {(stq.x / q) * s_scale, (stq.y / q) * t_scale};
  }

  void process_face(const FaceEntry &entry, const Transformation &world,
                    const Transformation &inverse, bool has_inverse,
                    bool mirrored, const MaterialState &inherited_material,
                    const CleanupOptions &options, bool ancestor_visible) {
    ++m_stats.faces_seen;
    const bool effective_visible =
        ancestor_visible && drawing_element_visible(
                                entry.entity_id, entry.raw_hidden, entry.layer,
                                options);
    if (!include_for_visibility(effective_visible, options)) {
      ++m_stats.faces_visibility_skipped;
      return;
    }

    const CachedFaceGeometry *geometry =
        load_face_geometry(entry, options.two_sided_materials);
    if (geometry == nullptr || geometry->vertices.empty() ||
        geometry->indices.empty())
      return;

    const MaterialState &front = geometry->front_material;
    const MaterialState &back = geometry->back_material;
    const MaterialState &effective_front =
        front.valid ? front : inherited_material;
    const MaterialState &effective_back =
        back.valid ? back : inherited_material;

    std::string material_key;
    if (options.two_sided_materials) {
      material_key = "__TWO_SIDED__:[\"" + effective_front.name + "\",\"" +
                     effective_back.name + "\"]";
    } else {
      material_key = effective_front.name;
    }

    auto &buckets = target_buckets(effective_visible);
    ReducedMesh &mesh = buckets.try_emplace(material_key).first->second;
    reserve_append(mesh, geometry->vertices.size(), geometry->indices.size());

    double front_s = 1.0;
    double front_t = 1.0;
    if (!front.valid) {
      const auto scale = texture_scale(effective_front);
      front_s = scale.first;
      front_t = scale.second;
    }

    double back_s = 1.0;
    double back_t = 1.0;
    if (options.two_sided_materials && !back.valid) {
      const auto scale = texture_scale(effective_back);
      back_s = scale.first;
      back_t = scale.second;
    }

    std::vector<int32_t> local_to_bucket(geometry->vertices.size(), -1);
    for (std::size_t i = 0; i < geometry->vertices.size(); ++i) {
      const CW::Point3D transformed_point =
          world * CW::Point3D(geometry->vertices[i]);
      const SUVector3D normal = transformed_normal(
          geometry->normals[i], world, inverse, has_inverse);
      const SUPoint2D front_uv =
          scaled_uv(geometry->front_stq[i], front_s, front_t);

      if (options.two_sided_materials) {
        const SUPoint2D back_uv =
            scaled_uv(geometry->back_stq[i], back_s, back_t);
        local_to_bucket[i] = find_or_add_vertex(
            mesh, transformed_point, normal, front_uv, &back_uv,
            options.unit_scale);
      } else {
        local_to_bucket[i] = find_or_add_vertex(
            mesh, transformed_point, normal, front_uv, nullptr,
            options.unit_scale);
      }
      ++m_stats.transformed_vertices;
    }

    for (std::size_t offset = 0; offset + 2 < geometry->indices.size();
         offset += 3) {
      std::size_t i0 = geometry->indices[offset];
      std::size_t i1 = geometry->indices[offset + 1];
      std::size_t i2 = geometry->indices[offset + 2];
      if (mirrored)
        std::swap(i1, i2);
      if (i0 >= local_to_bucket.size() || i1 >= local_to_bucket.size() ||
          i2 >= local_to_bucket.size())
        continue;
      const int32_t a = local_to_bucket[i0];
      const int32_t b = local_to_bucket[i1];
      const int32_t c = local_to_bucket[i2];
      if (a < 0 || b < 0 || c < 0)
        continue;
      mesh.indices.push_back(a);
      mesh.indices.push_back(b);
      mesh.indices.push_back(c);
      mesh.face_sizes.push_back(3);
      ++m_stats.triangles_emitted;
    }

    ++m_stats.faces_emitted;
  }

  void process_level(const EntityLevelPrototype &level,
                     const Transformation &world,
                     const MaterialState &inherited_material,
                     const CleanupOptions &options, int depth,
                     bool ancestor_visible) {
    if (depth > 100)
      return;
    ++m_stats.entity_levels;

    for (const auto &entry : level.instances) {
      ++m_stats.instances;
      const bool visible = ancestor_visible && drawing_element_visible(
                                                   entry.entity_id,
                                                   entry.raw_hidden,
                                                   entry.layer, options);
      const MaterialState &material = entry.direct_material.valid
                                          ? entry.direct_material
                                          : inherited_material;
      const Transformation child_world = world * entry.local_transform;
      const EntityLevelPrototype &child =
          definition_level(entry.definition_id, entry.child_entities);
      process_level(child, child_world, material, options, depth + 1, visible);
    }

    for (const auto &entry : level.groups) {
      ++m_stats.groups;
      const bool visible = ancestor_visible && drawing_element_visible(
                                                   entry.entity_id,
                                                   entry.raw_hidden,
                                                   entry.layer, options);
      const MaterialState &material = entry.direct_material.valid
                                          ? entry.direct_material
                                          : inherited_material;
      const Transformation child_world = world * entry.local_transform;
      const EntityLevelPrototype &child =
          definition_level(entry.definition_id, entry.child_entities);
      process_level(child, child_world, material, options, depth + 1, visible);
    }

    const double determinant = world.determinant();
    const bool mirrored = determinant < 0.0;
    const bool has_inverse = std::fabs(determinant) > 1e-12;
    const Transformation inverse =
        has_inverse ? world.inverse() : Transformation();

    for (const auto &entry : level.faces) {
      process_face(entry, world, inverse, has_inverse, mirrored,
                   inherited_material, options, ancestor_visible);
    }
  }
};

} // namespace CW

#endif // OptimizationV2_hpp
