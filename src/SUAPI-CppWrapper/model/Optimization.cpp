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
#include <map>
#include <set>
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
  cache_texture_scales();
  process_entities(m_model.entities(), Transformation(), Material());

  if (options.limited_dissolve || options.tris_to_quads) {
    for (auto &it : m_buckets) {
      apply_mesh_cleanup(it.second, options);
    }
  }
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
                                        Material inherited_material) {
  std::vector<ComponentInstance> instances = entities.instances();
  for (const auto &inst : instances) {
    Transformation child_transform = inst.transformation();
    Transformation new_transform = transform * child_transform;
    Material mat = inst.material();
    Material active_mat = mat.is_valid() ? mat : inherited_material;
    process_entities(inst.definition().entities(), new_transform, active_mat);
  }

  std::vector<Group> groups = entities.groups();
  for (const auto &grp : groups) {
    Transformation child_transform = grp.transformation();
    Transformation new_transform = transform * child_transform;
    Material mat = grp.material();
    Material active_mat = mat.is_valid() ? mat : inherited_material;
    process_entities(grp.entities(), new_transform, active_mat);
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

  Material active_mat =
      has_front ? front_mat : (has_back ? back_mat : inherited_material);

  std::string mat_name = "SketchUp_Default";
  if (active_mat.is_valid()) {
    mat_name = active_mat.name().std_string();
  }

  bool is_direct = has_front || has_back;
  double s_scale = 1.0;
  double t_scale = 1.0;

  if (!is_direct && active_mat.is_valid()) {
    auto it_scale = m_texture_scale_cache.find(mat_name);
    if (it_scale != m_texture_scale_cache.end()) {
      s_scale = it_scale->second.first;
      t_scale = it_scale->second.second;
    }
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
  std::vector<SUPoint3D> stq(num_vertices);
  std::vector<SUVector3D> norms(num_vertices);
  std::vector<size_t> indices_flat(num_triangles * 3);

  size_t v_count = 0;
  SUMeshHelperGetVertices(mesh_ref, num_vertices, verts.data(), &v_count);
  size_t stq_count = 0;
  if (has_back && !has_front) {
    SUMeshHelperGetBackSTQCoords(mesh_ref, num_vertices, stq.data(),
                                 &stq_count);
  } else {
    SUMeshHelperGetFrontSTQCoords(mesh_ref, num_vertices, stq.data(),
                                  &stq_count);
  }
  size_t n_count = 0;
  SUMeshHelperGetNormals(mesh_ref, num_vertices, norms.data(), &n_count);
  size_t i_count = 0;
  SUMeshHelperGetVertexIndices(mesh_ref, num_triangles * 3, indices_flat.data(),
                               &i_count);

  for (size_t i = 0; i < indices_flat.size(); i += 3) {
    for (int j = 0; j < 3; j++) {
      size_t idx = indices_flat[i + j];
      if (idx >= num_vertices)
        continue;

      CW::Point3D p_trans_cw = transform * CW::Point3D(verts[idx]);
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

void HierarchyReducer::apply_mesh_cleanup(ReducedMesh &mesh,
                                          const CleanupOptions &options) {
  if (mesh.indices.empty())
    return;

  // 1. Build position map for better adjacency (bypass UV splitting)
  using PosKey = std::tuple<int64_t, int64_t, int64_t>;
  std::vector<PosKey> pos_keys;
  pos_keys.reserve(mesh.vertices.size());
  for (const auto &v : mesh.vertices) {
    pos_keys.push_back(
        {static_cast<int64_t>(std::round(v.x * ReducedMesh::POS_SCALE)),
         static_cast<int64_t>(std::round(v.y * ReducedMesh::POS_SCALE)),
         static_cast<int64_t>(std::round(v.z * ReducedMesh::POS_SCALE))});
  }

  // 2. Build face offsets O(N)
  std::vector<size_t> face_offsets(mesh.face_sizes.size() + 1, 0);
  for (size_t i = 0; i < mesh.face_sizes.size(); ++i) {
    face_offsets[i + 1] = face_offsets[i] + mesh.face_sizes[i];
  }

  // 3. Position-based adjacency map
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

  std::unordered_map<PosEdge, std::vector<size_t>, PosEdgeHasher> edge_to_faces;
  for (size_t f = 0; f < mesh.face_sizes.size(); f++) {
    size_t offset = face_offsets[f];
    int32_t size = mesh.face_sizes[f];
    for (int32_t i = 0; i < size; i++) {
      const auto &pk1 = pos_keys[mesh.indices[offset + i]];
      const auto &pk2 = pos_keys[mesh.indices[offset + (i + 1) % size]];
      PosEdge e = {std::min(pk1, pk2), std::max(pk1, pk2)};
      edge_to_faces[e].push_back(f);
    }
  }

  // 4. Iterative Merge (Greedy for Quads, then Dissolve for Ngons)
  std::vector<bool> face_invalid(mesh.face_sizes.size(), false);
  std::vector<std::vector<int32_t>> active_faces;
  active_faces.reserve(mesh.face_sizes.size());
  for (size_t f = 0; f < mesh.face_sizes.size(); ++f) {
    std::vector<int32_t> face_indices;
    size_t start = face_offsets[f];
    for (int32_t i = 0; i < mesh.face_sizes[f]; ++i)
      face_indices.push_back(mesh.indices[start + i]);
    active_faces.push_back(std::move(face_indices));
  }

  bool changed = true;
  int pass = 0;
  while (changed && pass < 5) { // Limit iterations
    changed = false;
    pass++;
    for (size_t f1 = 0; f1 < active_faces.size(); ++f1) {
      if (face_invalid[f1])
        continue;

      // If only Tris-to-Quads, stop if already a quad
      if (options.tris_to_quads && !options.limited_dissolve &&
          active_faces[f1].size() >= 4)
        continue;

      for (size_t i = 0; i < active_faces[f1].size(); ++i) {
        const auto &pk1 = pos_keys[active_faces[f1][i]];
        const auto &pk2 =
            pos_keys[active_faces[f1][(i + 1) % active_faces[f1].size()]];
        PosEdge e = {std::min(pk1, pk2), std::max(pk1, pk2)};

        const auto &neighbors = edge_to_faces[e];
        for (size_t f2 : neighbors) {
          if (f2 <= f1 || face_invalid[f2])
            continue;

          // Check planarity
          const SUVector3D &n1 = mesh.normals[active_faces[f1][0]];
          const SUVector3D &n2 = mesh.normals[active_faces[f2][0]];
          double dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
          if (dot > std::cos(options.angle_limit_radians)) {
            // MERGE f2 into f1
            // 1. Identify common positions in f2 for edge pk2-pk1
            int j1 = -1, j2 = -1;
            for (int k = 0; k < active_faces[f2].size(); ++k) {
              if (pos_keys[active_faces[f2][k]] == pk2)
                j1 = k;
              if (pos_keys[active_faces[f2][k]] == pk1)
                j2 = k;
            }
            if (j1 == -1 || j2 == -1 ||
                ((j1 + 1) % active_faces[f2].size() != j2))
              continue;

            // 2. Synthesize new loop
            std::vector<int32_t> new_loop;
            // Part from f1: from i+1 back around to i
            for (int k = 0; k < active_faces[f1].size(); ++k) {
              new_loop.push_back(
                  active_faces[f1][(i + 1 + k) % active_faces[f1].size()]);
              if (new_loop.back() == active_faces[f1][i])
                break;
            }
            // Part from f2: from j2 around to j1 (skipping j1-j2 edge which is
            // shared) Wait, j1->j2 is pk2->pk1. In f1, pk1->pk2 was at i ->
            // i+1. Loop F1: ... -> pk1 -> pk2 -> ... Loop F2: ... -> pk2 -> pk1
            // -> ... Merged: ... -> pk1 -> (rest of F2) -> pk2 -> (rest of F1)
            // -> ...
            new_loop.clear();
            for (int k = 0; k < active_faces[f1].size(); k++) {
              int32_t v =
                  active_faces[f1][(i + 1 + k) % active_faces[f1].size()];
              new_loop.push_back(v);
            }
            // Insert F2 vertices between pk2 and pk1
            // pk2 is at original i+1. We just added it.
            std::vector<int32_t> insert_f2;
            for (int k = 1; k < active_faces[f2].size() - 1; k++) {
              insert_f2.push_back(
                  active_faces[f2][(j1 + 1 + k) % active_faces[f2].size()]);
            }
            new_loop.insert(new_loop.begin() + 1, insert_f2.begin(),
                            insert_f2.end());

            active_faces[f1] = std::move(new_loop);
            face_invalid[f2] = true;
            changed = true;

            // Update edge map for f1's new edges (optional but better)
            // To keep it simple, we'll rely on the while loop for further
            // merges
            break;
          }
        }
        if (changed)
          break;
      }
    }
  }

  // 5. Commit changes back to mesh
  mesh.indices.clear();
  mesh.face_sizes.clear();
  for (size_t f = 0; f < active_faces.size(); ++f) {
    if (face_invalid[f])
      continue;
    // Basic collinear reduction (dissolve mid-edge vertices)
    const auto &loop = active_faces[f];
    if (loop.size() < 3)
      continue;

    std::vector<int32_t> clean_loop;
    for (size_t i = 0; i < loop.size(); i++) {
      const auto &p1 = pos_keys[loop[(i + loop.size() - 1) % loop.size()]];
      const auto &p2 = pos_keys[loop[i]];
      const auto &p3 = pos_keys[loop[(i + 1) % loop.size()]];

      // Vector p1->p2 and p2->p3
      double dx1 = static_cast<double>(std::get<0>(p2) - std::get<0>(p1));
      double dy1 = static_cast<double>(std::get<1>(p2) - std::get<1>(p1));
      double dz1 = static_cast<double>(std::get<2>(p2) - std::get<2>(p1));
      double dx2 = static_cast<double>(std::get<0>(p3) - std::get<0>(p2));
      double dy2 = static_cast<double>(std::get<1>(p3) - std::get<1>(p2));
      double dz2 = static_cast<double>(std::get<2>(p3) - std::get<2>(p2));

      // Cross product for collinear check
      double cx = dy1 * dz2 - dz1 * dy2;
      double cy = dz1 * dx2 - dx1 * dz2;
      double cz = dx1 * dy2 - dy1 * dx2;
      double area_sq = cx * cx + cy * cy + cz * cz;
      double len_sq1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
      double len_sq2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

      if (area_sq < 1e-9 * len_sq1 * len_sq2) {
        // Collinear, skip p2
      } else {
        clean_loop.push_back(loop[i]);
      }
    }

    if (clean_loop.size() >= 3) {
      mesh.indices.insert(mesh.indices.end(), clean_loop.begin(),
                          clean_loop.end());
      mesh.face_sizes.push_back(static_cast<int32_t>(clean_loop.size()));
    }
  }
}

} /* namespace CW */
