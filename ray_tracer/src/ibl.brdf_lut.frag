#version 330 core
in vec2 f_tex_coord;
out vec2 out_colour;

// Implementation courtesy of https://learnopengl.com/
const float PI = 3.14159265359;

vec2 Hammersley(uint i, uint N) {
  uint bits = i;
  bits = (bits << 16) | (bits >> 16);
  bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
  bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
  bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
  bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
  float vdC = float(bits) / 4294967296.0;
  return vec2(float(i) / N, vdC);
}

vec3 ImportanceSample_GGX(vec2 Xi, vec3 N, float a)
{
  float phi = 2 * PI * Xi.x;
  float cos2_theta = (1 - Xi.y) / (1 + (a*a - 1) * Xi.y);
  float cos_theta = sqrt(cos2_theta);
  float sin_theta = sqrt(1 - cos2_theta);
  vec3 H = vec3(
    cos(phi) * sin_theta,
    sin(phi) * sin_theta,
    cos_theta
  );
  vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
  vec3 tangent = cross(up, N);
  vec3 bitangent = cross(N, tangent);
  return normalize(
    H.x * tangent +
    H.y * bitangent +
    H.z * N
  );
}

float G_Smith_Schlick(float NdotV, float NdotL, float a) {
  float k = a / 2;
  return
    (NdotL / (NdotL * (1-k) + k)) *
    (NdotV / (NdotV * (1-k) + k));
}

vec2 IntegrateBRDF(float NdotV, float roughness)
{
  vec3 V = vec3(
    sqrt(1 - NdotV * NdotV),
    0,
    NdotV
  );
  vec3 N = vec3(0, 0, 1);
  float a = roughness * roughness;

  float A = 0;
  float B = 0;
  const uint SAMPLE_COUNT = 1024u;
  for (uint i = 0u; i < SAMPLE_COUNT; i++) {
    vec2 Xi = Hammersley(i, SAMPLE_COUNT);
    vec3 H = ImportanceSample_GGX(Xi, N, a);
    vec3 L = reflect(-V, H);

    float NdotV = max(dot(N, V), 0);
    float NdotL = max(L.z, 0);
    float NdotH = max(H.z, 0);
    float HdotV = max(dot(H, V), 0);

    if (NdotL > 0) {
      float G = G_Smith_Schlick(NdotV, NdotL, a) * (HdotV / (NdotH * NdotV));
      float F = pow(1 - HdotV, 5);
      A += (1 - F) * G;
      B += F * G;
    }
  }
  return vec2(A, B) / float(SAMPLE_COUNT);
}

void main() {
  out_colour = IntegrateBRDF(f_tex_coord.x, f_tex_coord.y);
}
