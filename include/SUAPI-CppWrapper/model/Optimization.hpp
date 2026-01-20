#ifndef Optimization_hpp
#define Optimization_hpp

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "SUAPI-CppWrapper/Transformation.hpp"
#include "SUAPI-CppWrapper/model/Entities.hpp"
#include "SUAPI-CppWrapper/model/Face.hpp"
#include "SUAPI-CppWrapper/model/Material.hpp"
#include "SUAPI-CppWrapper/model/Model.hpp"
#include "SUAPI-CppWrapper/model/Texture.hpp"
#include <SketchUpAPI/geometry.h>

namespace CW {

// Structure to hold merged geometry for a specific material
struct ReducedMesh {
  std::vector<SUPoint3D> vertices;
  std::vector<SUVector3D> normals;
  std::vector<SUPoint2D> uvs;
  std::vector<int32_t> indices;
  std::vector<int32_t> face_sizes;

  // Spatial hashing for vertex welding
  // Key: (px, py, pz, nx, ny, nz, u, v) quantized
  using VertexKey = std::tuple<int64_t, int64_t, int64_t, int32_t, int32_t,
                               int32_t, int64_t, int64_t>;

  struct KeyHasher {
    std::size_t operator()(const VertexKey &k) const {
      size_t seed = 0;
      auto combine = [&](auto val) {
        seed ^= std::hash<decltype(val)>{}(val) + 0x9e3779b9 + (seed << 6) +
                (seed >> 2);
      };
      combine(std::get<0>(k));
      combine(std::get<1>(k));
      combine(std::get<2>(k));
      combine(std::get<3>(k));
      combine(std::get<4>(k));
      combine(std::get<5>(k));
      combine(std::get<6>(k));
      combine(std::get<7>(k));
      return seed;
    }
  };

  std::unordered_map<VertexKey, int32_t, KeyHasher> unique_map;

  // Tolerance for welding (internal units are inches)
  static constexpr double WELD_TOLERANCE_INCH = 0.004;
  static constexpr double POS_SCALE = 1.0 / WELD_TOLERANCE_INCH;
  static constexpr double NORMAL_SCALE = 1000.0;
  static constexpr double UV_SCALE = 10000.0;
};

struct CleanupOptions {
  bool limited_dissolve = false;
  bool tris_to_quads = false;
  double angle_limit_radians = 0.0872665; // ~5 degrees
};

class HierarchyReducer {
public:
  HierarchyReducer(Model &model);

  // Traverses the entire model and flattens geometry
  void traverse(const CleanupOptions &options = CleanupOptions());

  // Returns the buckets
  // Key: Material Name (or "Default")
  const std::map<std::string, ReducedMesh> &get_reduced_geometry() const;

private:
  Model &m_model;
  std::map<std::string, ReducedMesh> m_buckets;

  // Cache for texture scales to apply UV scaling during flattening
  std::map<std::string, std::pair<double, double>> m_texture_scale_cache;

  void process_entities(const Entities &entities,
                        const Transformation &transform,
                        Material inherited_material);
  void process_face(Face &face, const Transformation &transform,
                    Material inherited_material);

  // Helper to add vertex with welding
  void add_vertex(ReducedMesh &mesh, const SUPoint3D &pos,
                  const SUVector3D &norm, const SUPoint2D &uv);

  // Helper to load texture scales
  void cache_texture_scales();

  // Mesh cleanup logic
  void apply_mesh_cleanup(ReducedMesh &mesh, const CleanupOptions &options);
};

} /* namespace CW */

#endif /* Optimization_hpp */
