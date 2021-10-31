#version 330 core
uniform vec3 light_pos;
uniform vec3 cam_pos;
uniform samplerCube irradiance_map;
uniform samplerCube radiance_map;
uniform sampler2D brdf_lut;
uniform float metallic;
uniform float roughness;
in vec3 f_pos;
in vec3 f_normal;
out vec4 out_colour;

const vec3 light_colour = vec3(0.6) * 50;

// Implementation courtesy of https://learnopengl.com/
const float PI = 3.14159265359;
float sq(float x) { return x * x; }

float D_GGX(float NdotH, float a) {
  return
    a * a /
    (PI * sq(sq(NdotH) * (a*a - 1) + 1));
}

float G_Smith_Schlick_Analytic(float NdotV, float NdotL, float a) {
  float k = sq(a + 1) / 8;
  return
    (NdotL / (NdotL * (1-k) + k)) *
    (NdotV / (NdotV * (1-k) + k));
}

vec3 F_Schlick(vec3 F0, float cosTheta) {
  return F0 + (1 - F0) * pow(1 - cosTheta, 5);
}

vec3 F_Schlick_Lagarde(vec3 F0, float cosTheta, float roughness) {
  return F0 + (max(vec3(1 - roughness), F0) - F0) * pow(1 - cosTheta, 5);
}

void main() {
  vec3 albedo = vec3(0.9, 0.8, 0.5);
  float AO = 0.5;

  vec3 N = f_normal;
  vec3 V = normalize(cam_pos - f_pos);
  vec3 R = reflect(-V, N);
  float NdotV = max(dot(N, V), 0);

  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  // Outgoing radiance
  vec3 Lo = vec3(0);

  // Direct lighting
  {
    vec3 L_unnorm = light_pos - f_pos;
    vec3 L = normalize(L_unnorm);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0);
    float NdotH = max(dot(N, H), 0);
    float HdotV = max(dot(H, V), 0);

    vec3 kS = F_Schlick(F0, HdotV);
    float D = D_GGX(NdotH, roughness * roughness);
    float G = G_Smith_Schlick_Analytic(NdotV, NdotL, roughness);

    vec3 kD = (1 - kS) * (1 - metallic);
    vec3 BRDF =
      kD * albedo / PI +
      kS * D * G / (4 * NdotV * NdotL + 1e-5);
    vec3 Li = light_colour / dot(L_unnorm, L_unnorm);
    Lo += BRDF * Li * NdotL;
  }

  // Ambient lighting
  {
    vec3 kS = F_Schlick_Lagarde(F0, NdotV, roughness);
    vec3 kD = (1 - kS) * (1 - metallic);

    vec3 radiance = textureLod(radiance_map, R, roughness * 5).rgb;
    vec2 LUT = texture(brdf_lut, vec2(NdotV, roughness)).rg;

    vec3 ambient =
      kD * texture(irradiance_map, N).rgb * albedo +
      radiance * (kS * LUT.x + LUT.y);
    Lo += ambient * AO;
  }

  vec3 k = Lo / (Lo + 1); // Tone mapping
  out_colour = vec4(pow(k, vec3(1/2.2)), 1);
}
