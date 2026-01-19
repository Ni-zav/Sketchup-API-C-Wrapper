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
  process_entities(m_model.entities(), Transformation());
}

void HierarchyReducer::process_entities(const Entities &entities,
                                        const Transformation &transform) {
  // Process Instances
  std::vector<ComponentInstance> instances = entities.instances();
  for (const auto &inst : instances) {
    Transformation child_transform = inst.transformation();
    Transformation new_transform = transform * child_transform;
    process_entities(inst.definition().entities(), new_transform);
  }

  // Process Groups
  std::vector<Group> groups = entities.groups();
  for (const auto &grp : groups) {
    Transformation child_transform = grp.transformation();
    Transformation new_transform = transform * child_transform;
    process_entities(grp.entities(), new_transform);
  }

  // Process Faces
  std::vector<Face> faces = entities.faces();
  for (auto &face : faces) {
    process_face(face, transform);
  }
}

void HierarchyReducer::process_face(Face &face,
                                    const Transformation &transform) {
  // 1. Determine Material Bucket
  Material mat = face.material();
  std::string mat_name = "Default";
  if (mat) {
    mat_name = mat.name().std_string();
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
  std::vector<SUPoint3D> stq(num_vertices); // We'll convert to UV
  std::vector<SUVector3D> norms(num_vertices);
  std::vector<size_t> indices_flat(num_triangles * 3);

  SUMeshHelperGetVertices(mesh_ref, num_vertices, &verts[0]);
  SUMeshHelperGetFrontSTQCoords(mesh_ref, num_vertices,
                                &stq[0]); // Import front only for optimization
  SUMeshHelperGetNormals(mesh_ref, num_vertices, &norms[0]);
  SUMeshHelperGetVertexIndices(mesh_ref, num_triangles * 3, &indices_flat[0]);

  // 3. Transform & Weld
  for (size_t i : indices_flat) {
    // Transform Position
    // Use C++ wrapper operator*
    CW::Point3D p_cw(verts[i]);
    CW::Point3D p_trans_cw = transform * p_cw;
    SUPoint3D p_trans = p_trans_cw;

    // Transform Normal
    CW::Vector3D n_cw(norms[i]);
    CW::Vector3D n_trans_cw = transform * n_cw;
    SUVector3D n_trans = n_trans_cw;

    // Normalize normal after transform (scaling might affect it)
    double len = sqrt(n_trans.x * n_trans.x + n_trans.y * n_trans.y +
                      n_trans.z * n_trans.z);
    if (len > 1e-6) {
      n_trans.x /= len;
      n_trans.y /= len;
      n_trans.z /= len;
    }

    // Convert STQ to UV
    SUPoint2D uv_val;
    if (stq[i].z != 0.0) {
      uv_val.x = stq[i].x / stq[i].z;
      uv_val.y = stq[i].y / stq[i].z;
    } else {
      uv_val.x = stq[i].x;
      uv_val.y = stq[i].y;
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
