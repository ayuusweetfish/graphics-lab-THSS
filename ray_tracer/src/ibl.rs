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
    ",
      include_str!("ibl.lighting.frag")
    );

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
    ",
      include_str!("ibl.brdf_lut.frag")
    );

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
