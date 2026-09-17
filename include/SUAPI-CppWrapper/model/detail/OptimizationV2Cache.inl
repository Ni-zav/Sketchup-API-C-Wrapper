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

const EntityLevelPrototype &root_level() {
  if (m_root_cache.has_value()) {
    ++m_stats.root_cache_hits;
    return *m_root_cache;
  }

  ++m_stats.root_cache_misses;
  m_root_cache.emplace(build_level(m_model.entities()));
  return *m_root_cache;
}

const EntityLevelPrototype &definition_level(std::int32_t definition_id,
                                              const Entities &entities) {
  auto existing = m_definition_cache.find(definition_id);
  if (existing != m_definition_cache.end()) {
    ++m_stats.definition_cache_hits;
    return existing->second;
  }

  ++m_stats.definition_cache_misses;
  auto inserted =
      m_definition_cache.emplace(definition_id, build_level(entities));
  return inserted.first->second;
}

static std::uint64_t face_cache_key(std::int32_t face_id,
                                    bool two_sided_materials) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(face_id)) << 1) |
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
  geometry.back_material = material_state(entry.face.back_material());

  SUMeshHelperRef mesh_ref = SU_INVALID;
  if (SUMeshHelperCreate(&mesh_ref, entry.face.ref()) != SU_ERROR_NONE) {
    auto inserted = m_face_cache.emplace(key, std::move(geometry));
    return &inserted.first->second;
  }
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
    geometry.vertices.clear();
    geometry.normals.clear();
    geometry.indices.clear();
    geometry.front_stq.clear();
    auto inserted = m_face_cache.emplace(key, std::move(geometry));
    return &inserted.first->second;
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

  const bool needs_back_stq =
      two_sided_materials ||
      (!geometry.front_material.valid && geometry.back_material.valid);
  if (needs_back_stq) {
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

  geometry.valid = true;
  SUMeshHelperRelease(&mesh_ref);
  auto inserted = m_face_cache.emplace(key, std::move(geometry));
  return &inserted.first->second;
}
