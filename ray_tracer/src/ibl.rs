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

      void main() {
        out_colour = vec4(0.9, 0.8, 0.5, 1);
      }
    ");

    let uni_vp = gl::GetUniformLocation(prog, "VP\0".as_ptr().cast());
    let uni_cam_pos = gl::GetUniformLocation(prog, "cam_pos\0".as_ptr().cast());
    let uni_light_pos = gl::GetUniformLocation(prog, "light_pos\0".as_ptr().cast());

    Self {
      vao, vbo, prog,
      uni_vp, uni_cam_pos, uni_light_pos,
      num_vertices,
    }
  }

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
