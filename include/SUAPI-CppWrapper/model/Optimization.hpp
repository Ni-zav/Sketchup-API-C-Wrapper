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
#include "SUAPI-CppWrapper/model/Model.hpp"
#include <SketchUpAPI/geometry.h>


namespace CW {

// Structure to hold merged geometry for a specific material
struct ReducedMesh {
  std::vector<SUPoint3D> vertices;
  std::vector<SUVector3D> normals;
  std::vector<SUPoint2D> uvs;
  std::vector<int32_t> indices;

  // Spatial hashing for vertex welding
  // Key: (x_quantized, y_quantized, z_quantized) -> index
  using VertexKey = std::tuple<int64_t, int64_t, int64_t>;

  struct KeyHasher {
    std::size_t operator()(const VertexKey &k) const {
      auto h1 = std::hash<int64_t>{}(std::get<0>(k));
      auto h2 = std::hash<int64_t>{}(std::get<1>(k));
      auto h3 = std::hash<int64_t>{}(std::get<2>(k));
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  std::unordered_map<VertexKey, int32_t, KeyHasher> unique_map;

  // Tolerance for welding (internal units are inches, but efficient hash uses
  // integer quantization) 0.0001 meters approx 0.0039 inches.
  static constexpr double WELD_TOLERANCE_INCH = 0.004;
  static constexpr double SCALE_FACTOR = 1.0 / WELD_TOLERANCE_INCH;
};

class HierarchyReducer {
public:
  HierarchyReducer(Model &model);

  // Traverses the entire model and flattens geometry
  void traverse();

  // Returns the buckets
  // Key: Material Name (or "Default")
  const std::map<std::string, ReducedMesh> &get_reduced_geometry() const;

private:
  Model &m_model;
  std::map<std::string, ReducedMesh> m_buckets;

  void process_entities(const Entities &entities,
                        const Transformation &transform);
  void process_face(Face &face, const Transformation &transform);

  // Helper to add vertex with welding
  void add_vertex(ReducedMesh &mesh, const SUPoint3D &pos,
                  const SUVector3D &norm, const SUPoint2D &uv);
};

} /* namespace CW */

#endif /* Optimization_hpp */
