void reset_run_state() {
  m_buckets.clear();
  m_hidden_buckets.clear();
  m_stats = ReducerStatsV2();
  m_partition_output = false;
}

void sync_visibility_stats() {
  const auto &visibility_stats = m_visibility.stats();
  m_stats.layer_visibility_cache_hits =
      visibility_stats.layer_visibility_cache_hits;
  m_stats.layer_visibility_cache_misses =
      visibility_stats.layer_visibility_cache_misses;
}

void begin_run(const CleanupOptions &options, bool partition_output) {
  reset_run_state();
  m_partition_output = partition_output;
  m_visibility.reset(options);
  ensure_texture_scale_cache();
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
    const bool visible =
        ancestor_visible &&
        m_visibility.drawing_element_visible(entry.entity_id, entry.raw_hidden,
                                             entry.layer);
    const MaterialState &material =
        entry.direct_material.valid ? entry.direct_material : inherited_material;
    const Transformation child_world = world * entry.local_transform;
    const EntityLevelPrototype &child =
        definition_level(entry.definition_id, entry.child_entities);
    process_level(child, child_world, material, options, depth + 1, visible);
  }

  for (const auto &entry : level.groups) {
    ++m_stats.groups;
    const bool visible =
        ancestor_visible &&
        m_visibility.drawing_element_visible(entry.entity_id, entry.raw_hidden,
                                             entry.layer);
    const MaterialState &material =
        entry.direct_material.valid ? entry.direct_material : inherited_material;
    const Transformation child_world = world * entry.local_transform;
    const EntityLevelPrototype &child =
        definition_level(entry.definition_id, entry.child_entities);
    process_level(child, child_world, material, options, depth + 1, visible);
  }

  if (level.faces.empty())
    return;

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
