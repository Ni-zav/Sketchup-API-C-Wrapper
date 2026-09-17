#ifndef LooseEdgeBatchV2_hpp
#define LooseEdgeBatchV2_hpp

#include "SUAPI-CppWrapper/String.hpp"
#include "SUAPI-CppWrapper/Transformation.hpp"
#include "SUAPI-CppWrapper/model/ComponentDefinition.hpp"
#include "SUAPI-CppWrapper/model/ComponentInstance.hpp"
#include "SUAPI-CppWrapper/model/Edge.hpp"
#include "SUAPI-CppWrapper/model/Group.hpp"
#include "SUAPI-CppWrapper/model/Layer.hpp"
#include "SUAPI-CppWrapper/model/Model.hpp"
#include "SUAPI-CppWrapper/model/Optimization.hpp"
#include "SUAPI-CppWrapper/model/TraversalVisibilityV2.hpp"
#include "SUAPI-CppWrapper/model/Vertex.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace CW {

struct LooseEdgeRecordV2 {
  Point3D start;
  Point3D end;
  std::int32_t entity_id = 0;
  std::int32_t layer_id = 0;
  std::string layer_name;
  std::vector<std::string> path;
  bool raw_hidden = false;
  bool soft = false;
  bool smooth = false;
  bool effective_visible = true;
};

struct LooseEdgeStatsV2 {
  std::uint64_t entity_levels = 0;
  std::uint64_t definition_cache_hits = 0;
  std::uint64_t definition_cache_misses = 0;
  std::uint64_t instances = 0;
  std::uint64_t groups = 0;
  std::uint64_t edges_seen = 0;
  std::uint64_t edges_emitted = 0;
  std::uint64_t edges_visibility_skipped = 0;
  std::uint64_t layer_visibility_cache_hits = 0;
  std::uint64_t layer_visibility_cache_misses = 0;
};

// Native standalone-edge collector with the same effective-visibility inputs as
// HierarchyReducerV2. Source edge endpoints and metadata are cached per unique
// definition; repeated occurrences only apply the accumulated transform and
// shared TraversalVisibilityV2 state.
class LooseEdgeBatchV2 {
public:
  explicit LooseEdgeBatchV2(Model &model) : m_model(model) {}

  void collect(const CleanupOptions &options = CleanupOptions()) {
    begin_run(options, false);
    const LevelPrototype root = build_level(m_model.entities());
    std::vector<std::string> path;
    process_level(root, Transformation(), path, options, 0, true);
  }

  void collect_partitioned(const CleanupOptions &options = CleanupOptions()) {
    begin_run(options, true);
    const LevelPrototype root = build_level(m_model.entities());
    std::vector<std::string> path;
    process_level(root, Transformation(), path, options, 0, true);
  }

  const std::vector<LooseEdgeRecordV2> &edges() const { return m_edges; }
  const std::vector<LooseEdgeRecordV2> &hidden_edges() const {
    return m_hidden_edges;
  }

  LooseEdgeStatsV2 stats() const {
    LooseEdgeStatsV2 result = m_stats;
    const TraversalVisibilityStatsV2 &visibility_stats = m_visibility.stats();
    result.layer_visibility_cache_hits =
        visibility_stats.layer_visibility_cache_hits;
    result.layer_visibility_cache_misses =
        visibility_stats.layer_visibility_cache_misses;
    return result;
  }

  void reset_geometry_cache() { m_definition_cache.clear(); }

private:
  struct EdgeEntry {
    Point3D start;
    Point3D end;
    std::int32_t entity_id = 0;
    std::int32_t layer_id = 0;
    std::string layer_name;
    bool raw_hidden = false;
    bool soft = false;
    bool smooth = false;
    Layer layer;
  };

  struct InstanceEntry {
    std::int32_t entity_id = 0;
    std::int32_t definition_id = 0;
    Entities child_entities;
    Transformation local_transform;
    std::string path_name;
    bool raw_hidden = false;
    Layer layer;
  };

  struct GroupEntry {
    std::int32_t entity_id = 0;
    std::int32_t definition_id = 0;
    Entities child_entities;
    Transformation local_transform;
    std::string path_name;
    bool raw_hidden = false;
    Layer layer;
  };

  struct LevelPrototype {
    std::vector<EdgeEntry> edges;
    std::vector<InstanceEntry> instances;
    std::vector<GroupEntry> groups;
  };

  Model &m_model;
  std::vector<LooseEdgeRecordV2> m_edges;
  std::vector<LooseEdgeRecordV2> m_hidden_edges;
  std::unordered_map<std::int32_t, LevelPrototype> m_definition_cache;
  TraversalVisibilityV2 m_visibility;
  LooseEdgeStatsV2 m_stats;
  bool m_partition_output = false;

  void begin_run(const CleanupOptions &options, bool partition_output) {
    m_edges.clear();
    m_hidden_edges.clear();
    m_stats = LooseEdgeStatsV2();
    m_partition_output = partition_output;
    m_visibility.reset(options);
  }

  static std::string layer_name(const Layer &layer) {
    if (!layer.is_valid())
      return std::string();
    try {
      return layer.name().std_string();
    } catch (...) {
      return std::string();
    }
  }

  LevelPrototype build_level(const Entities &entities) {
    LevelPrototype level;

    const std::vector<Edge> edges = entities.edges(true);
    level.edges.reserve(edges.size());
    for (const auto &edge : edges) {
      const Layer layer = edge.layer();
      level.edges.push_back(EdgeEntry{
          edge.start().position(),
          edge.end().position(),
          static_cast<std::int32_t>(edge.entityID()),
          layer.is_valid() ? static_cast<std::int32_t>(layer.entityID()) : 0,
          layer_name(layer),
          edge.hidden(),
          edge.soft(),
          edge.smooth(),
          layer,
      });
    }

    const std::vector<ComponentInstance> instances = entities.instances();
    level.instances.reserve(instances.size());
    for (const auto &instance : instances) {
      const ComponentDefinition definition = instance.definition();
      std::string name = instance.name();
      if (name.empty())
        name = definition.name().std_string();
      name += "_" + std::to_string(instance.entityID());
      level.instances.push_back(InstanceEntry{
          static_cast<std::int32_t>(instance.entityID()),
          static_cast<std::int32_t>(definition.entityID()),
          definition.entities(),
          instance.transformation(),
          std::move(name),
          instance.hidden(),
          instance.layer(),
      });
    }

    const std::vector<Group> groups = entities.groups();
    level.groups.reserve(groups.size());
    for (const auto &group : groups) {
      const ComponentDefinition definition = group.definition();
      std::string name = group.name();
      if (name.empty())
        name = "Group_" + std::to_string(group.entityID());
      level.groups.push_back(GroupEntry{
          static_cast<std::int32_t>(group.entityID()),
          static_cast<std::int32_t>(definition.entityID()),
          group.entities(),
          group.transformation(),
          std::move(name),
          group.hidden(),
          group.layer(),
      });
    }

    return level;
  }

  const LevelPrototype &definition_level(std::int32_t definition_id,
                                         const Entities &entities) {
    const auto existing = m_definition_cache.find(definition_id);
    if (existing != m_definition_cache.end()) {
      ++m_stats.definition_cache_hits;
      return existing->second;
    }

    ++m_stats.definition_cache_misses;
    auto inserted = m_definition_cache.emplace(definition_id, build_level(entities));
    return inserted.first->second;
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

  std::vector<LooseEdgeRecordV2> &target_edges(bool effective_visible) {
    if (m_partition_output && !effective_visible)
      return m_hidden_edges;
    return m_edges;
  }

  static Point3D scaled_world_point(const Transformation &world,
                                    const Point3D &local,
                                    double unit_scale) {
    Point3D result = world * local;
    result.x *= unit_scale;
    result.y *= unit_scale;
    result.z *= unit_scale;
    return result;
  }

  void process_level(const LevelPrototype &level, const Transformation &world,
                     std::vector<std::string> &path,
                     const CleanupOptions &options, int depth,
                     bool ancestor_visible) {
    if (depth > 100)
      return;
    ++m_stats.entity_levels;

    for (const auto &entry : level.edges) {
      ++m_stats.edges_seen;
      const bool effective_visible =
          ancestor_visible && m_visibility.drawing_element_visible(
                                  entry.entity_id, entry.raw_hidden, entry.layer);
      if (!include_for_visibility(effective_visible, options)) {
        ++m_stats.edges_visibility_skipped;
        continue;
      }

      LooseEdgeRecordV2 record;
      record.start = scaled_world_point(world, entry.start, options.unit_scale);
      record.end = scaled_world_point(world, entry.end, options.unit_scale);
      record.entity_id = entry.entity_id;
      record.layer_id = entry.layer_id;
      record.layer_name = entry.layer_name;
      record.path = path;
      record.raw_hidden = entry.raw_hidden;
      record.soft = entry.soft;
      record.smooth = entry.smooth;
      record.effective_visible = effective_visible;
      target_edges(effective_visible).push_back(std::move(record));
      ++m_stats.edges_emitted;
    }

    for (const auto &entry : level.groups) {
      ++m_stats.groups;
      const bool visible = ancestor_visible &&
                           m_visibility.drawing_element_visible(
                               entry.entity_id, entry.raw_hidden, entry.layer);
      const Transformation child_world = world * entry.local_transform;
      const LevelPrototype &child =
          definition_level(entry.definition_id, entry.child_entities);
      path.push_back(entry.path_name);
      process_level(child, child_world, path, options, depth + 1, visible);
      path.pop_back();
    }

    for (const auto &entry : level.instances) {
      ++m_stats.instances;
      const bool visible = ancestor_visible &&
                           m_visibility.drawing_element_visible(
                               entry.entity_id, entry.raw_hidden, entry.layer);
      const Transformation child_world = world * entry.local_transform;
      const LevelPrototype &child =
          definition_level(entry.definition_id, entry.child_entities);
      path.push_back(entry.path_name);
      process_level(child, child_world, path, options, depth + 1, visible);
      path.pop_back();
    }
  }
};

} // namespace CW

#endif // LooseEdgeBatchV2_hpp
