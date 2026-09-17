#ifndef TraversalVisibilityV2_hpp
#define TraversalVisibilityV2_hpp

#include "SUAPI-CppWrapper/model/Layer.hpp"
#include "SUAPI-CppWrapper/model/Optimization.hpp"

#include <SketchUpAPI/model/layer.h>
#include <SketchUpAPI/model/layer_folder.h>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace CW {

struct TraversalVisibilityStatsV2 {
  std::uint64_t layer_visibility_cache_hits = 0;
  std::uint64_t layer_visibility_cache_misses = 0;
};

// Shared effective-visibility evaluator for native traversal pipelines.
//
// This class owns only per-run visibility state. Geometry/entity prototypes may
// safely outlive a reset(), while visibility caches and scene override sets are
// rebuilt for every import/traversal configuration.
class TraversalVisibilityV2 {
public:
  TraversalVisibilityV2() = default;

  void reset(const CleanupOptions &options) {
    m_use_scene_hidden_objects = options.use_scene_hidden_objects;
    m_use_scene_hidden_layers = options.use_scene_hidden_layers;

    m_hidden_entity_ids.clear();
    m_layer_override_ids.clear();
    m_hidden_layer_folder_ids.clear();
    m_layer_visibility_cache.clear();
    m_stats = TraversalVisibilityStatsV2();

    m_hidden_entity_ids.reserve(options.hidden_entity_ids.size());
    m_hidden_entity_ids.insert(options.hidden_entity_ids.begin(),
                               options.hidden_entity_ids.end());
    m_layer_override_ids.reserve(options.layer_override_ids.size());
    m_layer_override_ids.insert(options.layer_override_ids.begin(),
                                options.layer_override_ids.end());
    m_hidden_layer_folder_ids.reserve(options.hidden_layer_folder_ids.size());
    m_hidden_layer_folder_ids.insert(options.hidden_layer_folder_ids.begin(),
                                     options.hidden_layer_folder_ids.end());
  }

  bool drawing_element_visible(std::int32_t entity_id, bool raw_hidden,
                               const Layer &layer) {
    const bool hidden =
        m_use_scene_hidden_objects
            ? (m_hidden_entity_ids.find(entity_id) != m_hidden_entity_ids.end())
            : raw_hidden;
    return !hidden && layer_visible(layer);
  }

  bool layer_visible(const Layer &layer) {
    if (!layer.is_valid())
      return true;

    const std::int32_t layer_id = static_cast<std::int32_t>(layer.entityID());
    const auto cached = m_layer_visibility_cache.find(layer_id);
    if (cached != m_layer_visibility_cache.end()) {
      ++m_stats.layer_visibility_cache_hits;
      return cached->second;
    }
    ++m_stats.layer_visibility_cache_misses;

    bool visible = true;
    if (SULayerGetVisibility(layer.ref(), &visible) != SU_ERROR_NONE)
      visible = true;

    if (m_use_scene_hidden_layers &&
        m_layer_override_ids.find(layer_id) != m_layer_override_ids.end()) {
      visible = !visible;
    }

    if (visible) {
      SULayerFolderRef folder = SU_INVALID;
      SUResult result = SULayerGetParentLayerFolder(layer.ref(), &folder);
      int guard = 0;
      while (result == SU_ERROR_NONE && SUIsValid(folder) && guard++ < 128) {
        bool folder_visible = true;
        if (SULayerFolderGetVisibility(folder, &folder_visible) == SU_ERROR_NONE &&
            !folder_visible) {
          visible = false;
          break;
        }

        if (m_use_scene_hidden_layers) {
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

  const TraversalVisibilityStatsV2 &stats() const { return m_stats; }

private:
  bool m_use_scene_hidden_objects = false;
  bool m_use_scene_hidden_layers = false;
  std::unordered_set<std::int32_t> m_hidden_entity_ids;
  std::unordered_set<std::int32_t> m_layer_override_ids;
  std::unordered_set<std::int32_t> m_hidden_layer_folder_ids;
  std::unordered_map<std::int32_t, bool> m_layer_visibility_cache;
  TraversalVisibilityStatsV2 m_stats;
};

} // namespace CW

#endif // TraversalVisibilityV2_hpp
