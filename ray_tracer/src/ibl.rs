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
  num_vertices: gl::int,

  skybox: gl::uint,
  irradiance_map: gl::uint,
  radiance_map: gl::uint,
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
      in vec3 f_pos;
      in vec3 f_normal;
      out vec4 out_colour;

const vec3 light_colour = vec3(0.6) * 100;

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

      void main() {
        vec3 albedo = vec3(0.9, 0.8, 0.5);
        float metallic = 0.9;
        float roughness = 0.6;
        float ao = 0.5;

        vec3 N = f_normal;
        vec3 V = normalize(cam_pos - f_pos);

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

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * albedo * ao;

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

    // Irradiance map

    Self {
      vao, vbo, prog,
      uni_vp, uni_cam_pos, uni_light_pos,
      num_vertices,

      skybox: load_hdr_cubemap(&["ibl/skybox_posx.hdr"]),
      irradiance_map: load_hdr_cubemap(&["ibl/irrad_posx.hdr"]),
      radiance_map: load_hdr_cubemap(&[
        "ibl/rad_posx_0_256x256.hdr",
        "ibl/rad_posx_1_128x128.hdr",
        "ibl/rad_posx_2_64x64.hdr",
        "ibl/rad_posx_3_32x32.hdr",
        "ibl/rad_posx_4_16x16.hdr",
        "ibl/rad_posx_5_8x8.hdr",
      ]),
    }
  }

  pub fn skybox(&self) -> gl::uint { self.skybox }
  pub fn irradiance_map(&self) -> gl::uint { self.irradiance_map }
  pub fn radiance_map(&self) -> gl::uint { self.radiance_map }

  pub fn draw(
    &self,
    vp: glm::Mat4,
    cam_pos: glm::Vec3,
    light_pos: glm::Vec3,
  ) {
    // Set uniforms
    gl::UseProgram(self.prog);
    gl::UniformMatrix4fv(self.uni_vp, 1, gl::FALSE, vp.as_array().as_ptr().cast());
    gl::Uniform3f(self.uni_cam_pos, cam_pos.x, cam_pos.y, cam_pos.z);
    gl::Uniform3f(self.uni_light_pos, light_pos.x, light_pos.y, light_pos.z);

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
  }
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MIN_FILTER, gl::LINEAR as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MAG_FILTER, gl::LINEAR as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::int);
  gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as gl::int);
  tex
}
