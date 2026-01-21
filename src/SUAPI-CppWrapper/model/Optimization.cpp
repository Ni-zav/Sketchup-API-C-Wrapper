#include "SUAPI-CppWrapper/model/Optimization.hpp"
#include "SUAPI-CppWrapper/model/ComponentDefinition.hpp"
#include "SUAPI-CppWrapper/model/ComponentInstance.hpp"
#include "SUAPI-CppWrapper/model/GeometryInputHelper.hpp"
#include "SUAPI-CppWrapper/model/Group.hpp"
#include <SketchUpAPI/geometry.h>
#include <SketchUpAPI/model/face.h>
#include <SketchUpAPI/model/mesh_helper.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

#define _USE_MATH_DEFINES
#include <math.h>

namespace CW {

HierarchyReducer::HierarchyReducer(Model &model) : m_model(model) {}

const std::map<std::string, ReducedMesh> &
HierarchyReducer::get_reduced_geometry() const {
  return m_buckets;
}

void HierarchyReducer::traverse(const CleanupOptions &options) {
  printf("C++: Starting model traversal. Dissolve: %d, Tris-to-Quads: %d, "
         "Angle: %.3f\n",
         options.limited_dissolve, options.tris_to_quads,
         options.angle_limit_radians);

  cache_texture_scales(); // Fix: Ensure texture scales are cached for UV
                          // calculation
  process_entities(m_model.entities(), Transformation(), Material(), 0);

  finalize(options);
}

void HierarchyReducer::finalize(const CleanupOptions &options) {
  if (options.limited_dissolve || options.tris_to_quads) {
    for (auto &it : m_buckets) {
      size_t before = it.second.face_sizes.size();
      apply_mesh_cleanup(it.second, options);
      size_t after = it.second.face_sizes.size();
      if (before != after) {
        printf("C++: Optimized [%s]: %zu -> %zu faces\n", it.first.c_str(),
               before, after);
      }
    }
  }

  // Final pass: Convert all vertices from inches to meters (or requested scale)
  for (auto &it : m_buckets) {
    for (auto &v : it.second.vertices) {
      v.x *= options.unit_scale;
      v.y *= options.unit_scale;
      v.z *= options.unit_scale;
    }
  }
}

void HierarchyReducer::traverse_entities(const Entities &entities,
                                         const CleanupOptions &options) {
  cache_texture_scales();
  process_entities(entities, Transformation(), Material(), 0);

  finalize(options);
}

void HierarchyReducer::cache_texture_scales() {
  std::vector<Material> materials = m_model.materials();
  for (const auto &mat : materials) {
    if (!mat.is_valid())
      continue;
    Texture tex = mat.texture();
    if (tex.is_valid()) {
      m_texture_scale_cache[mat.name().std_string()] =
          std::make_pair(tex.s_scale(), tex.t_scale());
    }
  }
}

void HierarchyReducer::process_entities(const Entities &entities,
                                        const Transformation &transform,
                                        Material inherited_material,
                                        int depth) {
  if (depth > 100) {
    printf(
        "C++: Warning: Max recursion depth (100) reached. Aborting branch.\n");
    return;
  }

  std::vector<ComponentInstance> instances = entities.instances();
  for (const auto &inst : instances) {
    Transformation child_transform = inst.transformation();
    Transformation new_transform = transform * child_transform;
    Material mat = inst.material();
    Material active_mat = mat.is_valid() ? mat : inherited_material;
    process_entities(inst.definition().entities(), new_transform, active_mat,
                     depth + 1);
  }

  std::vector<Group> groups = entities.groups();
  for (const auto &grp : groups) {
    Transformation child_transform = grp.transformation();
    Transformation new_transform = transform * child_transform;
    Material mat = grp.material();
    Material active_mat = mat.is_valid() ? mat : inherited_material;
    process_entities(grp.entities(), new_transform, active_mat, depth + 1);
  }

  std::vector<Face> faces = entities.faces();
  for (auto &face : faces) {
    process_face(face, transform, inherited_material);
  }
}

void HierarchyReducer::process_face(Face &face, const Transformation &transform,
                                    Material inherited_material) {
  Material front_mat = face.material();
  Material back_mat = face.back_material();
  bool has_front = front_mat.is_valid();
  bool has_back = back_mat.is_valid();

  double det = transform.determinant();
  bool is_mirrored = (det < 0.0);

  // Material selection logic: Direct Front > Inherited
  // Matches unoptimized behavior (mesh_builder.py) by ignoring back materials.
  Material active_mat;
  bool use_front_side = true;
  bool is_direct = false;

  if (has_front) {
    active_mat = front_mat;
    is_direct = true;
  } else {
    active_mat = inherited_material;
    is_direct = false;
  }

  std::string mat_name = "SketchUp_Default";
  if (active_mat.is_valid()) {
    mat_name = active_mat.name().std_string();
  }

  if (m_buckets.find(mat_name) == m_buckets.end()) {
    m_buckets[mat_name] = ReducedMesh();
  }
  ReducedMesh &mesh_buffer = m_buckets[mat_name];

  SUMeshHelperRef mesh_ref = SU_INVALID;
  SUMeshHelperCreate(&mesh_ref, face.ref());

  size_t num_vertices = 0;
  SUMeshHelperGetNumVertices(mesh_ref, &num_vertices);
  size_t num_triangles = 0;
  SUMeshHelperGetNumTriangles(mesh_ref, &num_triangles);

  if (num_vertices == 0 || num_triangles == 0) {
    SUMeshHelperRelease(&mesh_ref);
    return;
  }

  std::vector<SUPoint3D> verts(num_vertices);
  std::vector<SUVector3D> norms(num_vertices);
  std::vector<size_t> indices_flat(num_triangles * 3);

  size_t v_count = 0;
  SUMeshHelperGetVertices(mesh_ref, num_vertices, verts.data(), &v_count);
  size_t n_count = 0;
  SUMeshHelperGetNormals(mesh_ref, num_vertices, norms.data(), &n_count);
  size_t i_count = 0;
  SUMeshHelperGetVertexIndices(mesh_ref, num_triangles * 3, indices_flat.data(),
                               &i_count);

  // UV Scale logic
  double s_scale = 1.0;
  double t_scale = 1.0;
  if (!is_direct && active_mat.is_valid()) {
    auto it_scale = m_texture_scale_cache.find(mat_name);
    if (it_scale != m_texture_scale_cache.end()) {
      s_scale = it_scale->second.first;
      t_scale = it_scale->second.second;
    }
  }

  std::vector<SUPoint3D> stq(num_vertices);
  size_t stq_count = 0;
  if (use_front_side) {
    SUMeshHelperGetFrontSTQCoords(mesh_ref, num_vertices, stq.data(),
                                  &stq_count);
  } else {
    SUMeshHelperGetBackSTQCoords(mesh_ref, num_vertices, stq.data(),
                                 &stq_count);
  }

  // If the transformation is mirrored, we must flip the winding to stay CCW in
  // world space. This is ESSENTIAL for baked geometry where the instance
  // matrix is discarded.
  bool flip_winding = is_mirrored;

  for (size_t i = 0; i < indices_flat.size(); i += 3) {
    for (int j_off = 0; j_off < 3; j_off++) {
      // Winding flip: swap index 1 and 2 if flip_winding is true
      int j = j_off;
      if (flip_winding) {
        if (j_off == 1)
          j = 2;
        else if (j_off == 2)
          j = 1;
      }

      size_t idx = indices_flat[i + j];
      if (idx >= num_vertices)
        continue;

      // Transform point.
      // Note: We do NOT scale to meters here yet, to maintain welding
      // precision. Scaling is performed as a final pass in finalize().
      CW::Point3D p_raw(verts[idx]);
      CW::Point3D p_trans_cw = transform * p_raw;

      // Transform normal. Normals are direction vectors, they don't get scaled
      // by INCH_TO_METER but do get transformed.
      SUVector3D n_trans = transform * CW::Vector3D(norms[idx]);

      double len_sq =
          n_trans.x * n_trans.x + n_trans.y * n_trans.y + n_trans.z * n_trans.z;
      if (len_sq > 1e-12) {
        double len = sqrt(len_sq);
        n_trans.x /= len;
        n_trans.y /= len;
        n_trans.z /= len;
      } else {
        n_trans = {0.0, 0.0, 1.0};
      }

      double q = (stq[idx].z == 0.0) ? 1.0 : stq[idx].z;
      SUPoint2D uv_val;
      uv_val.x = (stq[idx].x / q) * s_scale;
      uv_val.y = (stq[idx].y / q) * t_scale;

      add_vertex(mesh_buffer, p_trans_cw, n_trans, uv_val);
    }
    mesh_buffer.face_sizes.push_back(3);
  }

  SUMeshHelperRelease(&mesh_ref);
}

void HierarchyReducer::add_vertex(ReducedMesh &mesh, const SUPoint3D &pos,
                                  const SUVector3D &norm, const SUPoint2D &uv) {
  int64_t kx = static_cast<int64_t>(std::round(pos.x * ReducedMesh::POS_SCALE));
  int64_t ky = static_cast<int64_t>(std::round(pos.y * ReducedMesh::POS_SCALE));
  int64_t kz = static_cast<int64_t>(std::round(pos.z * ReducedMesh::POS_SCALE));

  int32_t knx =
      static_cast<int32_t>(std::round(norm.x * ReducedMesh::NORMAL_SCALE));
  int32_t kny =
      static_cast<int32_t>(std::round(norm.y * ReducedMesh::NORMAL_SCALE));
  int32_t knz =
      static_cast<int32_t>(std::round(norm.z * ReducedMesh::NORMAL_SCALE));

  int64_t ku = static_cast<int64_t>(std::round(uv.x * ReducedMesh::UV_SCALE));
  int64_t kv = static_cast<int64_t>(std::round(uv.y * ReducedMesh::UV_SCALE));

  ReducedMesh::VertexKey key =
      std::make_tuple(kx, ky, kz, knx, kny, knz, ku, kv);

  auto it = mesh.unique_map.find(key);
  if (it != mesh.unique_map.end()) {
    mesh.indices.push_back(it->second);
  } else {
    int32_t new_idx = static_cast<int32_t>(mesh.vertices.size());
    mesh.vertices.push_back(pos);
    mesh.normals.push_back(norm);
    mesh.uvs.push_back(uv);
    mesh.unique_map[key] = new_idx;
    mesh.indices.push_back(new_idx);
  }
}

// Helper to compute geometric face normal using Newell's method (robust for
// ngons)
SUVector3D compute_face_normal(const std::vector<int32_t> &loop,
                               const std::vector<SUPoint3D> &vertices) {
  if (loop.size() < 3)
    return {0.0, 0.0, 1.0};

  double nx = 0, ny = 0, nz = 0;
  for (size_t i = 0; i < loop.size(); ++i) {
    const auto &curr = vertices[loop[i]];
    const auto &next = vertices[loop[(i + 1) % loop.size()]];
    nx += (curr.y - next.y) * (curr.z + next.z);
    ny += (curr.z - next.z) * (curr.x + next.x);
    nz += (curr.x - next.x) * (curr.y + next.y);
  }

  double len = sqrt(nx * nx + ny * ny + nz * nz);
  if (len < 1e-12)
    return {0.0, 0.0, 1.0};
  return {nx / len, ny / len, nz / len};
}

void HierarchyReducer::apply_mesh_cleanup(ReducedMesh &mesh,
                                          const CleanupOptions &options) {
  if (mesh.indices.empty())
    return;

  using PosKey = std::tuple<int64_t, int64_t, int64_t>;
  std::vector<PosKey> pos_keys;
  pos_keys.reserve(mesh.vertices.size());
  for (const auto &v : mesh.vertices) {
    pos_keys.push_back(
        {static_cast<int64_t>(std::round(v.x * ReducedMesh::POS_SCALE)),
         static_cast<int64_t>(std::round(v.y * ReducedMesh::POS_SCALE)),
         static_cast<int64_t>(std::round(v.z * ReducedMesh::POS_SCALE))});
  }

  struct PosEdge {
    PosKey p1, p2;
    bool operator<(const PosEdge &other) const {
      return std::tie(p1, p2) < std::tie(other.p1, other.p2);
    }
    bool operator==(const PosEdge &other) const {
      return p1 == other.p1 && p2 == other.p2;
    }
  };
  struct PosEdgeHasher {
    size_t operator()(const PosEdge &e) const {
      size_t h = 0;
      auto combine = [&](int64_t v) {
        h ^= std::hash<int64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
      };
      combine(std::get<0>(e.p1));
      combine(std::get<1>(e.p1));
      combine(std::get<2>(e.p1));
      combine(std::get<0>(e.p2));
      combine(std::get<1>(e.p2));
      combine(std::get<2>(e.p2));
      return h;
    }
  };

  std::vector<std::vector<int32_t>> active_faces;
  active_faces.reserve(mesh.face_sizes.size()); // Pre-allocate for performance
  size_t offset = 0;
  for (int32_t size : mesh.face_sizes) {
    std::vector<int32_t> loop;
    loop.reserve(size); // Pre-allocate loop
    for (int32_t i = 0; i < size; ++i)
      loop.push_back(mesh.indices[offset + i]);
    active_faces.push_back(std::move(loop));
    offset += size;
  }

  auto clean_collinear = [&](std::vector<int32_t> &loop) {
    if (loop.size() < 3)
      return;
    std::vector<int32_t> next_loop;
    for (size_t i = 0; i < loop.size(); i++) {
      const auto &p1 = pos_keys[loop[(i + loop.size() - 1) % loop.size()]];
      const auto &p2 = pos_keys[loop[i]];
      const auto &p3 = pos_keys[loop[(i + 1) % loop.size()]];

      double dx1 = (double)(std::get<0>(p2) - std::get<0>(p1));
      double dy1 = (double)(std::get<1>(p2) - std::get<1>(p1));
      double dz1 = (double)(std::get<2>(p2) - std::get<2>(p1));
      double dx2 = (double)(std::get<0>(p3) - std::get<0>(p2));
      double dy2 = (double)(std::get<1>(p3) - std::get<1>(p2));
      double dz2 = (double)(std::get<2>(p3) - std::get<2>(p2));

      double cx = dy1 * dz2 - dz1 * dy2;
      double cy = dz1 * dx2 - dx1 * dz2;
      double cz = dx1 * dy2 - dy1 * dx2;
      double area_sq = cx * cx + cy * cy + cz * cz;
      double len_sq1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
      double len_sq2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

      if (area_sq < 1e-10 * len_sq1 * len_sq2)
        continue; // Collinear
      next_loop.push_back(loop[i]);
    }
    loop = std::move(next_loop);
  };

  // Initial cleaning
  for (auto &loop : active_faces) {
    clean_collinear(loop);
  }

  std::vector<bool> face_invalid(active_faces.size(), false);
  std::unordered_map<PosEdge, std::vector<size_t>, PosEdgeHasher> edge_to_faces;
  auto rebuild_edge_map = [&]() {
    edge_to_faces.clear();
    for (size_t f = 0; f < active_faces.size(); f++) {
      if (face_invalid[f])
        continue;
      int32_t size = static_cast<int32_t>(active_faces[f].size());
      for (int32_t i = 0; i < size; i++) {
        const auto &pk1 = pos_keys[active_faces[f][i]];
        const auto &pk2 = pos_keys[active_faces[f][(i + 1) % size]];
        PosEdge e = {std::min(pk1, pk2), std::max(pk1, pk2)};
        edge_to_faces[e].push_back(f);
      }
    }
  };

  rebuild_edge_map();

  bool changed = true;
  int pass = 0;

  std::vector<SUVector3D> face_normals(active_faces.size());
  for (size_t f = 0; f < active_faces.size(); f++) {
    face_normals[f] = compute_face_normal(active_faces[f], mesh.vertices);
  }

  // Performance: reduced max passes from 40 to 20, with early exit on low merge
  // rate
  while (changed && pass < 20) {
    changed = false;
    pass++;
    size_t merges_this_pass = 0; // Track merge efficiency

    // Greedy pass: attempt to merge everything we can in one go
    for (size_t f1 = 0; f1 < active_faces.size(); f1++) {
      if (face_invalid[f1])
        continue;
      if (options.tris_to_quads && !options.limited_dissolve &&
          active_faces[f1].size() >= 4)
        continue;

      // We treat n1 as constant for the life of f1 in this pass
      // UNLESS f1 is modified.

      for (size_t i = 0; i < active_faces[f1].size(); i++) {
        const auto &pk1 = pos_keys[active_faces[f1][i]];
        const auto &pk2 =
            pos_keys[active_faces[f1][(i + 1) % active_faces[f1].size()]];
        PosEdge e = {std::min(pk1, pk2), std::max(pk1, pk2)};

        auto it_neighbors = edge_to_faces.find(e);
        if (it_neighbors == edge_to_faces.end())
          continue;

        bool merged_this_edge = false;
        for (size_t f2 : it_neighbors->second) {
          if (f2 == f1 || face_invalid[f2])
            continue;
          if (options.tris_to_quads && !options.limited_dissolve &&
              active_faces[f2].size() != 3)
            continue;

          // Geometric planarity check - Normals are cached!
          const SUVector3D &n1 = face_normals[f1];
          const SUVector3D &n2 = face_normals[f2];
          double dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;

          if (dot > std::cos(options.angle_limit_radians + 0.001)) {
            // Find shared edge in f2
            int j1 = -1, j2 = -1;
            size_t size2 = active_faces[f2].size();
            for (size_t k = 0; k < size2; k++) {
              if (pos_keys[active_faces[f2][k]] == pk2)
                j1 = (int)k;
              if (pos_keys[active_faces[f2][k]] == pk1)
                j2 = (int)k;
            }
            if (j1 == -1 || j2 == -1 || (j1 + 1) % (int)size2 != j2)
              continue;

            // UV/Normal Seam Check: Only merge if the indices are identical.
            // If the indices differ, it means the vertices have different UVs
            // or Normals. Note: Shared edge has opposite winding in f2: (j2 ->
            // j1) matches (i -> i+1)
            if (active_faces[f1][i] != active_faces[f2][j2] ||
                active_faces[f1][(i + 1) % active_faces[f1].size()] !=
                    active_faces[f2][j1]) {
              continue; // Skip merge to preserve UV/Normal seam
            }

            // Synthesis new loop
            std::vector<int32_t> new_loop;
            size_t size1 = active_faces[f1].size();
            new_loop.reserve(size1 + size2);

            for (size_t k = 0; k <= i; k++)
              new_loop.push_back(active_faces[f1][k]);
            for (size_t k = 1; k < size2 - 1; k++) {
              new_loop.push_back(active_faces[f2][(j2 + k) % size2]);
            }
            for (size_t k = i + 1; k < size1; k++) {
              new_loop.push_back(active_faces[f1][k]);
            }

            clean_collinear(new_loop);

            if (new_loop.size() >= 3) {
              active_faces[f1] = std::move(new_loop);
              face_invalid[f2] = true;
              // Update f1 normal for future merges in this pass
              face_normals[f1] =
                  compute_face_normal(active_faces[f1], mesh.vertices);
              changed = true;
              merged_this_edge = true;
              merges_this_pass++; // Track merges for early exit
              break;
            }
          }
        }
        if (merged_this_edge) {
          i = (size_t)-1; // Restart edge loop for f1 since it was modified
        }
      }
    }
    // Early exit: if merge rate drops below 1% after first few passes, stop
    if (pass > 5 && merges_this_pass < active_faces.size() / 100 + 1) {
      break;
    }
    if (changed)
      rebuild_edge_map();
  }

  mesh.indices.clear();
  mesh.face_sizes.clear();
  for (size_t f = 0; f < active_faces.size(); f++) {
    if (face_invalid[f])
      continue;
    const auto &loop = active_faces[f];
    if (loop.size() >= 3) {
      mesh.indices.insert(mesh.indices.end(), loop.begin(), loop.end());
      mesh.face_sizes.push_back(static_cast<int32_t>(loop.size()));
    }
  }
}

} /* namespace CW */
