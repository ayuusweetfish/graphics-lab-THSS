use crate::gl;
use crate::scene_loader;

use core::mem::{size_of, size_of_val};

pub struct IBL {
  vao: gl::uint,
  vbo: gl::uint,
  prog: gl::uint,
  uni_vp: gl::int,
  uni_cam_pos: gl::int,
  uni_light_pos: gl::int,
  uni_metallic: gl::int,
  uni_roughness: gl::int,
  num_vertices: gl::int,

  skybox: gl::uint,
  irradiance_map: gl::uint,
  radiance_map: gl::uint,
  brdf_lut: gl::uint,
}

impl IBL {
  pub fn new(
    vertices: &[scene_loader::Vertex],
  ) -> Self {
    let mut vao = 0;
    gl::GenVertexArrays(1, &mut vao);
    gl::BindVertexArray(vao);
    let mut vbo = 0;
    gl::GenBuffers(1, &mut vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

    // Vertex attributes
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(
      0,
      3, gl::FLOAT, gl::FALSE,
      size_of::<scene_loader::Vertex>() as gl::int,
      (0 * size_of::<f32>()) as *const _,
    );
    gl::EnableVertexAttribArray(1);
    gl::VertexAttribPointer(
      1,
      3, gl::FLOAT, gl::FALSE,
      size_of::<scene_loader::Vertex>() as gl::int,
      (3 * size_of::<f32>()) as *const _,
    );

    // Upload vertex data to buffer
    gl::BufferData(
      gl::ARRAY_BUFFER,
      size_of_val(vertices) as isize,
      vertices.as_ptr().cast(),
      gl::STATIC_DRAW,
    );
    let num_vertices = vertices.len() as gl::int;

    // Program
    let prog = crate::program(r"
      #version 330 core
      uniform mat4 VP;
      layout (location = 0) in vec3 v_pos;
      layout (location = 1) in vec3 v_normal;
      out vec3 f_pos;
      out vec3 f_normal;

      void main() {
        gl_Position = VP * vec4(v_pos, 1.0);
        f_pos = v_pos;
        f_normal = v_normal;
      }
    ", r"
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

// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

      void main() {
        vec3 albedo = vec3(0.9, 0.8, 0.5);
        float ao = 0.5;

        vec3 N = f_normal;
        vec3 V = normalize(cam_pos - f_pos);
        vec3 R = reflect(-V, N);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);

    // calculate per-light radiance
    vec3 L = normalize(light_pos - f_pos);
    vec3 H = normalize(V + L);
    float dist = length(light_pos - f_pos);
    float attenuation = 1.0 / (dist * dist);
    vec3 radiance = light_colour * attenuation;

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;

    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;

    // scale light by NdotL
    float NdotL = max(dot(N, L), 0.0);

    // add to outgoing radiance Lo
    Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again

    // ambient lighting (we now use IBL as the ambient term)
    kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    vec3 irradiance = texture(irradiance_map, N).rgb;
    vec3 diffuse      = irradiance * albedo;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(radiance_map, R,  roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf  = texture(brdf_lut, vec2(max(dot(N, V), 0.0), roughness)).rg;
    specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;

    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2));
        
        out_colour = vec4(color, 1);
      }
    ");

    let uni_vp = gl::GetUniformLocation(prog, "VP\0".as_ptr().cast());
    let uni_cam_pos = gl::GetUniformLocation(prog, "cam_pos\0".as_ptr().cast());
    let uni_light_pos = gl::GetUniformLocation(prog, "light_pos\0".as_ptr().cast());
    let uni_metallic = gl::GetUniformLocation(prog, "metallic\0".as_ptr().cast());
    let uni_roughness = gl::GetUniformLocation(prog, "roughness\0".as_ptr().cast());
    let uni_irradiance_map = gl::GetUniformLocation(prog, "irradiance_map\0".as_ptr().cast());
    let uni_radiance_map = gl::GetUniformLocation(prog, "radiance_map\0".as_ptr().cast());
    let uni_brdf_lut = gl::GetUniformLocation(prog, "brdf_lut\0".as_ptr().cast());

    gl::UseProgram(prog);
    gl::Uniform1i(uni_irradiance_map, 0);
    gl::Uniform1i(uni_radiance_map, 1);
    gl::Uniform1i(uni_brdf_lut, 2);

    // BRDF LUT
    let mut brdf_lut = 0;
    gl::GenTextures(1, &mut brdf_lut);
    gl::BindTexture(gl::TEXTURE_2D, brdf_lut);
    gl::TexImage2D(
      gl::TEXTURE_2D,
      0,
      gl::RG16F as gl::int,
      512, 512, 0,
      gl::RG,
      gl::FLOAT,
      0 as *const _
    );
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as gl::int);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as gl::int);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::int);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::int);

    let mut fbo = 0;
    gl::GenFramebuffers(1, &mut fbo);
    gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
    gl::FramebufferTexture2D(gl::FRAMEBUFFER,
      gl::COLOR_ATTACHMENT0,
      gl::TEXTURE_2D, brdf_lut, 0);

    let mut fb_vao = 0;
    gl::GenVertexArrays(1, &mut fb_vao);
    gl::BindVertexArray(fb_vao);
    let mut fb_vbo = 0;
    gl::GenBuffers(1, &mut fb_vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, fb_vbo);

    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(
      0,
      2, gl::FLOAT, gl::FALSE,
      (2 * size_of::<f32>()) as gl::int,
      0 as *const _,
    );

    let fb_verts: [f32; 12] = [
      -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,
      -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,
    ];
    gl::BufferData(
      gl::ARRAY_BUFFER,
      size_of_val(&fb_verts) as isize,
      fb_verts.as_ptr().cast(),
      gl::STATIC_DRAW,
    );

    // Program
    let lut_prog = crate::program(r"
      #version 330 core
      layout (location = 0) in vec2 v_pos;
      out vec2 f_tex_coord;

      void main() {
        gl_Position = vec4(v_pos, 0.0, 1.0);
        f_tex_coord = vec2((1 + v_pos.x) / 2, (1 - v_pos.y) / 2);
      }
    ", r"
#version 330 core
out vec2 FragColor;
in vec2 f_tex_coord;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits)
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;

	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	// from tangent-space H vector to world-space sample vector
	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    // note that we use a different k for IBL
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec2 IntegrateBRDF(float NdotV, float roughness)
{
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    vec3 N = vec3(0.0, 0.0, 1.0);

    const uint SAMPLE_COUNT = 1024u;
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // generates a sample vector that's biased towards the
        // preferred alignment direction (importance sampling).
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}
// ----------------------------------------------------------------------------
void main()
{
    vec2 integratedBRDF = IntegrateBRDF(f_tex_coord.x, f_tex_coord.y);
    FragColor = integratedBRDF;
}

    ");

    // Render LUT
    gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
    gl::Viewport(0, 0, 512, 512);
    gl::UseProgram(lut_prog);
    gl::Clear(gl::COLOR_BUFFER_BIT);
    gl::BindVertexArray(fb_vao);
    gl::BindBuffer(gl::ARRAY_BUFFER, fb_vbo);
    gl::DrawArrays(gl::TRIANGLES, 0, 6);

    gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

    Self {
      vao, vbo, prog,
      uni_vp, uni_cam_pos, uni_light_pos, uni_metallic, uni_roughness,
      num_vertices,

      skybox: load_hdr_cubemap(&["ibl/skybox_posx.hdr"]),
      irradiance_map: load_hdr_cubemap(&["ibl/irrad_posx.hdr"]),
      radiance_map: load_hdr_cubemap(&[
        "ibl/rad_posx_0_256x256.hdr",
        "ibl/rad_posx_1_128x128.hdr",
        "ibl/rad_posx_2_64x64.hdr",
        "ibl/rad_posx_3_32x32.hdr",
        "ibl/rad_posx_4_16x16.hdr",
        "ibl/rad_posx_5_8x8.hdr", // XXX: artefacts?
      ]),
      brdf_lut,
    }
  }

  pub fn skybox(&self) -> gl::uint { self.skybox }
  pub fn irradiance_map(&self) -> gl::uint { self.irradiance_map }
  pub fn radiance_map(&self) -> gl::uint { self.radiance_map }
  pub fn brdf_lut(&self) -> gl::uint { self.brdf_lut }

  pub fn draw(
    &self,
    vp: glm::Mat4,
    cam_pos: glm::Vec3,
    light_pos: glm::Vec3,
    metallic: f32,
    roughness: f32,
  ) {
    // Set uniforms
    gl::UseProgram(self.prog);
    gl::UniformMatrix4fv(self.uni_vp, 1, gl::FALSE, vp.as_array().as_ptr().cast());
    gl::Uniform3f(self.uni_cam_pos, cam_pos.x, cam_pos.y, cam_pos.z);
    gl::Uniform3f(self.uni_light_pos, light_pos.x, light_pos.y, light_pos.z);
    gl::Uniform1f(self.uni_metallic, metallic);
    gl::Uniform1f(self.uni_roughness, roughness);

    // Bind textures
    gl::ActiveTexture(gl::TEXTURE0);
    gl::BindTexture(gl::TEXTURE_CUBE_MAP, self.irradiance_map);
    gl::ActiveTexture(gl::TEXTURE1);
    gl::BindTexture(gl::TEXTURE_CUBE_MAP, self.radiance_map);
    gl::ActiveTexture(gl::TEXTURE2);
    gl::BindTexture(gl::TEXTURE_2D, self.brdf_lut);

    gl::BindVertexArray(self.vao);
    gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);
    gl::DrawArrays(gl::TRIANGLES, 0, self.num_vertices);
  }
}

fn load_hdr_cubemap(
  files: &[&str],
) -> gl::uint {
  let mut tex = 0;
  gl::GenTextures(1, &mut tex);
  gl::ActiveTexture(gl::TEXTURE0);
  gl::BindTexture(gl::TEXTURE_CUBE_MAP, tex);
  let face_replacement = [
    "posx", "negx", "posy", "negy", "posz", "negz"
  ];
  for (mipmap_level, mipmap_base_file) in files.iter().enumerate() {
    for (face_index, face_name) in face_replacement.iter().enumerate() {
      let file = mipmap_base_file.replacen("posx", face_name, 1);
      let decoder = image::codecs::hdr::HdrDecoder::new(
        std::io::BufReader::new(std::fs::File::open(&file)
          .expect(&format!("cannot open file {}", file)))
      ).expect(&format!("file {} is not a proper HDR file", file));
      let md = decoder.metadata();
      let (w, h) = (md.width, md.height);
      let buf = decoder.read_image_hdr()
        .expect(&format!("file {} is not a proper HDR file", file)).as_ptr();
      gl::TexImage2D(
        gl::TEXTURE_CUBE_MAP_POSITIVE_X + face_index as u32,
        mipmap_level as gl::int,
        gl::RGB16F as gl::int,
        w as gl::int, h as gl::int, 0,
        gl::RGB,
        gl::FLOAT,
        buf.cast()
      );
      println!("Loaded IBL cubemap (mipmap {}) {}", mipmap_level, file);
    }
    if mipmap_level == 0 && files.len() > 1 {
      gl::GenerateMipmap(gl::TEXTURE_CUBE_MAP);
    }
  }
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MIN_FILTER,
    if files.len() == 1 { gl::LINEAR as gl::int }
    else { gl::LINEAR_MIPMAP_LINEAR as gl::int });
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MAG_FILTER, gl::LINEAR as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as gl::int);
  tex
}
