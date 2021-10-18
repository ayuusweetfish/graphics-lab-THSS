mod gl;

use glfw::Context;

use core::mem::{size_of, size_of_val};

fn check_gl_errors() {
  let err = gl::GetError();
  if err != 0 {
    panic!("OpenGL error: {}", err);
  }
}

fn main() {
  let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

  glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
  glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
  glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

  let (mut window, events) = glfw.create_window(
    960, 540,
    "Window",
    glfw::WindowMode::Windowed,
  )
    .expect("Cannot open window -- check graphics driver");

  gl::load_with(|s| window.get_proc_address(s) as *const _);
  glfw.set_swap_interval(glfw::SwapInterval::Sync(1));

  window.set_key_polling(true);
  window.make_current();

  let mut vao = 0;
  gl::GenVertexArrays(1, &mut vao);
  gl::BindVertexArray(vao);
  let mut vbo = 0;
  gl::GenBuffers(1, &mut vbo);
  gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

  let mut vertices: [[f32; 6]; 3] = [
    [-0.4, -0.4, 0.0, 1.0, 0.6, 0.0],
    [ 0.4, -0.4, 0.0, 0.8, 0.9, 0.1],
    [ 0.0,  0.5, 0.0, 0.7, 0.4, 1.0],
  ];
  gl::VertexAttribPointer(
    0,
    vertices.len() as gl::int, gl::FLOAT, gl::FALSE,
    size_of_val(&vertices[0]) as gl::int,
    0 as *const _,
  );
  gl::VertexAttribPointer(
    1,
    vertices.len() as gl::int, gl::FLOAT, gl::FALSE,
    size_of_val(&vertices[0]) as gl::int,
    (3 * size_of::<f32>()) as *const _,
  );
  gl::EnableVertexAttribArray(0);
  gl::EnableVertexAttribArray(1);

  let vs = gl::CreateShader(gl::VERTEX_SHADER);
  const VERTEX_SHADER: &str = r"
#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 v_colour_i;
out vec3 v_colour;
void main() {
  gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
  v_colour = v_colour_i;
}
";
  gl::ShaderSource(
    vs, 1,
    &(VERTEX_SHADER.as_bytes().as_ptr().cast()),
    &(VERTEX_SHADER.len() as gl::int),
  );
  gl::CompileShader(vs);

  let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
  const FRAGMENT_SHADER: &str = r"
#version 330 core
in vec3 v_colour;
out vec4 colour;

void main() {
  colour = vec4(v_colour, 1.0);
}
";
  gl::ShaderSource(
    fs, 1,
    &(FRAGMENT_SHADER.as_bytes().as_ptr().cast()),
    &(FRAGMENT_SHADER.len() as gl::int),
  );
  gl::CompileShader(fs);

  let prog = gl::CreateProgram();
  gl::AttachShader(prog, vs);
  gl::AttachShader(prog, fs);
  gl::LinkProgram(prog);
  gl::DeleteShader(vs);
  gl::DeleteShader(fs);

  gl::UseProgram(prog);

  check_gl_errors();

  while !window.should_close() {
    window.swap_buffers();

    glfw.poll_events();
    for (_, event) in glfw::flush_messages(&events) {
      match event {
        glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) => {
          window.set_should_close(true)
        }
        _ => {}
      }
    }

    gl::ClearColor(1.0, 0.99, 0.99, 1.0);
    gl::Clear(gl::COLOR_BUFFER_BIT);

    vertices[1][1] = (-0.4 + glfw.get_time().sin() * 0.2) as f32;
    gl::BufferData(
      gl::ARRAY_BUFFER,
      size_of_val(&vertices) as isize,
      vertices.as_ptr().cast(),
      gl::STREAM_DRAW,
    );

    gl::DrawArrays(gl::TRIANGLES, 0, 3);
  }
}
