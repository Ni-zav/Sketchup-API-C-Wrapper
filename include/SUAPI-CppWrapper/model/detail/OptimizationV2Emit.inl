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

void process_face(const FaceEntry &entry, const Transformation &world,
                  const Transformation &inverse, bool has_inverse,
                  bool mirrored, const MaterialState &inherited_material,
                  const CleanupOptions &options, bool ancestor_visible) {
  ++m_stats.faces_seen;
  const bool effective_visible =
      ancestor_visible &&
      m_visibility.drawing_element_visible(entry.entity_id, entry.raw_hidden,
                                           entry.layer);
  if (!include_for_visibility(effective_visible, options)) {
    ++m_stats.faces_visibility_skipped;
    return;
  }

  const CachedFaceGeometry *geometry =
      load_face_geometry(entry, options.two_sided_materials);
  if (geometry == nullptr || !geometry->valid || geometry->vertices.empty() ||
      geometry->indices.empty())
    return;

  const MaterialState &front = geometry->front_material;
  const MaterialState &back = geometry->back_material;
  const MaterialState &effective_front =
      front.valid ? front : inherited_material;
  const MaterialState &effective_back =
      back.valid ? back : inherited_material;
  const MaterialState &effective_single =
      front.valid ? front : (back.valid ? back : inherited_material);

  const std::string material_key =
      options.two_sided_materials
          ? two_sided_material_key(effective_front.name, effective_back.name)
          : effective_single.name;

  auto &buckets = target_buckets(effective_visible);
  ReducedMesh &mesh = buckets.try_emplace(material_key).first->second;
  reserve_append(mesh, geometry->vertices.size(), geometry->indices.size());

  double front_s = 1.0;
  double front_t = 1.0;
  if (!front.valid) {
    const auto scale = texture_scale(effective_single);
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
    const SUPoint3D &single_stq =
        (!front.valid && back.valid && geometry->has_back_stq)
            ? geometry->back_stq[i]
            : geometry->front_stq[i];
    const SUPoint2D front_uv = scaled_uv(single_stq, front_s, front_t);

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
