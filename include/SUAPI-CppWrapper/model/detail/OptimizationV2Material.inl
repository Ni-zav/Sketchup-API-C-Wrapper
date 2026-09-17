static MaterialState default_material_state() { return MaterialState(); }

static MaterialState material_state(const Material &material) {
  MaterialState state;
  state.material = material;
  state.valid = material.is_valid();
  if (state.valid) {
    state.name = material.name().std_string();
    if (state.name.empty())
      state.name = "SketchUp_Default";
  }
  return state;
}

static std::string json_escape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size() + 4);
  for (char ch : value) {
    switch (ch) {
    case '\\': escaped += "\\\\"; break;
    case '"': escaped += "\\\""; break;
    case '\b': escaped += "\\b"; break;
    case '\f': escaped += "\\f"; break;
    case '\n': escaped += "\\n"; break;
    case '\r': escaped += "\\r"; break;
    case '\t': escaped += "\\t"; break;
    default: escaped += ch; break;
    }
  }
  return escaped;
}

static std::string two_sided_material_key(const std::string &front,
                                          const std::string &back) {
  return "__TWO_SIDED__:[\"" + json_escape(front) + "\",\"" +
         json_escape(back) + "\"]";
}

void ensure_texture_scale_cache() {
  if (m_texture_scale_cache_ready)
    return;

  ++m_stats.material_scale_cache_builds;
  const std::vector<Material> materials = m_model.materials();
  m_texture_scale_cache.clear();
  m_texture_scale_cache.reserve(materials.size());
  for (const auto &material : materials) {
    if (!material.is_valid())
      continue;
    try {
      const Texture texture = material.texture();
      if (texture.is_valid()) {
        m_texture_scale_cache.emplace(
            material.name().std_string(),
            std::make_pair(texture.s_scale(), texture.t_scale()));
      }
    } catch (...) {
      // Optional metadata failure must not invalidate geometry import.
    }
  }
  m_texture_scale_cache_ready = true;
}

std::pair<double, double> texture_scale(const MaterialState &material) const {
  if (!material.valid)
    return {1.0, 1.0};
  const auto found = m_texture_scale_cache.find(material.name);
  if (found == m_texture_scale_cache.end())
    return {1.0, 1.0};
  return found->second;
}

static SUVector3D transformed_normal(const SUVector3D &normal,
                                     const Transformation &world,
                                     const Transformation &inverse,
                                     bool has_inverse) {
  SUVector3D transformed;
  if (has_inverse) {
    transformed = {
        inverse[0] * normal.x + inverse[1] * normal.y + inverse[2] * normal.z,
        inverse[4] * normal.x + inverse[5] * normal.y + inverse[6] * normal.z,
        inverse[8] * normal.x + inverse[9] * normal.y + inverse[10] * normal.z,
    };
  } else {
    transformed = world * CW::Vector3D(normal);
  }

  const double length_squared = transformed.x * transformed.x +
                                transformed.y * transformed.y +
                                transformed.z * transformed.z;
  if (length_squared > 1e-12) {
    const double inv_length = 1.0 / std::sqrt(length_squared);
    transformed.x *= inv_length;
    transformed.y *= inv_length;
    transformed.z *= inv_length;
  } else {
    transformed = {0.0, 0.0, 1.0};
  }
  return transformed;
}

static SUPoint2D scaled_uv(const SUPoint3D &stq, double s_scale,
                           double t_scale) {
  // Match legacy semantics exactly: only an exact zero q is replaced.
  const double q = stq.z == 0.0 ? 1.0 : stq.z;
  return {(stq.x / q) * s_scale, (stq.y / q) * t_scale};
}
