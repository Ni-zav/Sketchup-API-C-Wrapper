#include "SUAPI-CppWrapper/model/Optimization.hpp"
#include "SUAPI-CppWrapper/model/ComponentDefinition.hpp"
#include "SUAPI-CppWrapper/model/ComponentInstance.hpp"
#include "SUAPI-CppWrapper/model/GeometryInputHelper.hpp" // For tessellation maybe? Use internal mesh helper API directly
#include "SUAPI-CppWrapper/model/Group.hpp"
#include <SketchUpAPI/geometry.h>
#include <SketchUpAPI/model/face.h>
#include <SketchUpAPI/model/mesh_helper.h>
#include <cmath>

namespace CW {

HierarchyReducer::HierarchyReducer(Model &model) : m_model(model) {}

const std::map<std::string, ReducedMesh> &
HierarchyReducer::get_reduced_geometry() const {
  return m_buckets;
}

void HierarchyReducer::traverse() {
  cache_texture_scales();
  process_entities(m_model.entities(), Transformation(), Material());
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
  // Process Instances
  std::vector<ComponentInstance> instances = entities.instances();
  for (const auto &inst : instances) {
    Transformation child_transform = inst.transformation();
    Transformation new_transform = transform * child_transform;

    // Instance level material inheritance
    Material mat = inst.material();
    Material active_mat = mat.is_valid() ? mat : inherited_material;

    process_entities(inst.definition().entities(), new_transform, active_mat);
  }

  // Process Groups
  std::vector<Group> groups = entities.groups();
  for (const auto &grp : groups) {
    Transformation child_transform = grp.transformation();
    Transformation new_transform = transform * child_transform;

    // Group level material inheritance
    Material mat = grp.material();
    Material active_mat = mat.is_valid() ? mat : inherited_material;

    process_entities(grp.entities(), new_transform, active_mat);
  }

  // Process Faces
  std::vector<Face> faces = entities.faces();
  for (auto &face : faces) {
    process_face(face, transform, inherited_material);
  }
}

void HierarchyReducer::process_face(Face &face, const Transformation &transform,
                                    Material inherited_material) {
  // 1. Determine Material Bucket
  Material mat = face.material();
  Material active_mat = mat.is_valid() ? mat : inherited_material;

  std::string mat_name = "SketchUp_Default";
  if (active_mat.is_valid()) {
    mat_name = active_mat.name().std_string();
  }

  // Determine if we need UV scaling
  double s_scale = 1.0;
  double t_scale = 1.0;

  // STQ coordinates from SUMeshHelper are already normalized (0.0 to 1.0)
  // if the material is applied directly to the face.
  // They are in inches (raw units) if the material is inherited from a parent.
  if (!mat.is_valid() && active_mat.is_valid()) {
    auto it_scale = m_texture_scale_cache.find(mat_name);
    if (it_scale != m_texture_scale_cache.end()) {
      s_scale = it_scale->second.first;
      t_scale = it_scale->second.second;
    }
  }

  // Create bucket if not exists
  if (m_buckets.find(mat_name) == m_buckets.end()) {
    m_buckets[mat_name] = ReducedMesh();
  }
  ReducedMesh &mesh_buffer = m_buckets[mat_name];

  // 2. Tessellate Face
  // Use SUMeshHelper to get triangulated data with STQ
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

  // Get raw data
  std::vector<SUPoint3D> verts(num_vertices);
  std::vector<SUPoint3D> stq(num_vertices);
  std::vector<SUVector3D> norms(num_vertices);
  std::vector<size_t> indices_flat(num_triangles * 3);

  size_t v_count = 0;
  SUMeshHelperGetVertices(mesh_ref, num_vertices, verts.data(), &v_count);

  size_t stq_count = 0;
  SUMeshHelperGetFrontSTQCoords(mesh_ref, num_vertices, stq.data(), &stq_count);

  size_t n_count = 0;
  SUMeshHelperGetNormals(mesh_ref, num_vertices, norms.data(), &n_count);

  size_t i_count = 0;
  SUMeshHelperGetVertexIndices(mesh_ref, num_triangles * 3, indices_flat.data(),
                               &i_count);

  // 3. Transform & Weld
  for (size_t i : indices_flat) {
    if (i >= num_vertices)
      continue; // Safety check

    // Transform Position
    CW::Point3D p_trans_cw = transform * CW::Point3D(verts[i]);
    SUPoint3D p_trans = p_trans_cw;

    // Transform Normal
    SUVector3D n_trans = transform * CW::Vector3D(norms[i]);

    // Normalize normal after transform (scaling might affect it)
    double len_sq =
        n_trans.x * n_trans.x + n_trans.y * n_trans.y + n_trans.z * n_trans.z;
    if (len_sq > 1e-12) {
      double len = sqrt(len_sq);
      n_trans.x /= len;
      n_trans.y /= len;
      n_trans.z /= len;
    } else {
      // Fallback for degenerate normals
      n_trans = {0.0, 0.0, 1.0};
    }

    // Convert STQ to UV with scaling for SketchUp textures
    SUPoint2D uv_val;
    if (stq[i].z != 0.0) {
      uv_val.x = (stq[i].x / stq[i].z) * s_scale;
      uv_val.y = (stq[i].y / stq[i].z) * t_scale;
    } else {
      uv_val.x = stq[i].x * s_scale;
      uv_val.y = stq[i].y * t_scale;
    }

    // Add welded
    add_vertex(mesh_buffer, p_trans, n_trans, uv_val);
  }

  SUMeshHelperRelease(&mesh_ref);
}

void HierarchyReducer::add_vertex(ReducedMesh &mesh, const SUPoint3D &pos,
                                  const SUVector3D &norm, const SUPoint2D &uv) {
  // Quantize for spatial hashing
  int64_t kx =
      static_cast<int64_t>(std::round(pos.x * ReducedMesh::SCALE_FACTOR));
  int64_t ky =
      static_cast<int64_t>(std::round(pos.y * ReducedMesh::SCALE_FACTOR));
  int64_t kz =
      static_cast<int64_t>(std::round(pos.z * ReducedMesh::SCALE_FACTOR));

  ReducedMesh::VertexKey key = std::make_tuple(kx, ky, kz);

  auto it = mesh.unique_map.find(key);
  if (it != mesh.unique_map.end()) {
    // Exists - verify normal/uv similarity?
    // For pure geometry welding like `Remove Doubles`, checking position is
    // usually sufficient. But if normal/UV is distinct, we usually split.
    // `op_prepare.py` uses strict BMesh remove doubles which merges by
    // distance. It might merge verts with different UVs, resulting in UV seam
    // averaging or one winning. For safety, let's just merge by position for
    // now (optimization focus).

    // Add index
    mesh.indices.push_back(it->second);
  } else {
    // New vertex
    int32_t new_idx = static_cast<int32_t>(mesh.vertices.size());
    mesh.vertices.push_back(pos);
    mesh.normals.push_back(norm);
    mesh.uvs.push_back(uv);

    mesh.unique_map[key] = new_idx;
    mesh.indices.push_back(new_idx);
  }
}

} /* namespace CW */
