mod gl;
mod scene_loader;

use glfw::Context;
use wavefront_obj::obj;

use core::mem::{size_of, size_of_val};

fn check_gl_errors() {
  let err = gl::GetError();
  if err != 0 {
    panic!("OpenGL error: {}", err);
  }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

  // Load frame
  let frame = scene_loader::load("1a/1a_000001.obj")?;
  gl::VertexAttribPointer(
    0,
    3, gl::FLOAT, gl::FALSE,
    size_of_val(&frame.vertices[0]) as gl::int,
    0 as *const _,
  );
  gl::EnableVertexAttribArray(0);
  gl::VertexAttribPointer(
    1,
    3, gl::FLOAT, gl::FALSE,
    size_of_val(&frame.vertices[0]) as gl::int,
    (3 * size_of::<f32>()) as *const _,
  );
  gl::EnableVertexAttribArray(1);
  gl::BufferData(
    gl::ARRAY_BUFFER,
    size_of_val(&*frame.vertices) as isize,
    frame.vertices.as_ptr().cast(),
    gl::STREAM_DRAW,
  );

  let vs = gl::CreateShader(gl::VERTEX_SHADER);
  const VERTEX_SHADER: &str = r"
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
uniform vec3 light_pos;
uniform vec3 cam_pos;
in vec3 f_pos;
in vec3 f_normal;
out vec4 out_colour;

void main() {
  vec3 ambient_colour = vec3(0.1, 0.05, 0.0);
  vec3 light_colour = vec3(0.9, 0.8, 0.7);

  vec3 n = normalize(f_normal);

  vec3 light_dir = normalize(light_pos - f_pos);
  float diff = 0.7 * max(dot(n, light_dir), 0);

  vec3 view_dir = normalize(cam_pos - f_pos);
  vec3 refl_dir = reflect(-light_dir, n);
  float spec = 0.3 * pow(max(dot(view_dir, refl_dir), 0.0), 16);

  out_colour = vec4(ambient_colour + (diff + spec) * light_colour, 1.0);
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

  // Camera
  let cam_pos = (9.02922, -8.50027, 7.65063);
  let cam_right = (
    8.72799 - cam_pos.0,
    -8.21633 - cam_pos.1,
    8.48306 - cam_pos.2,
  );
  let cam_look = (4.01535, -3.77411, 4.22417);

  let uni_vp = gl::GetUniformLocation(prog, "VP\0".as_ptr().cast());
  let v = glm::ext::look_at(
    glm::vec3(cam_pos.0, cam_pos.1, cam_pos.2),
    glm::vec3(cam_look.0, cam_look.1, cam_look.2),
    glm::vec3(cam_right.0, cam_right.1, cam_right.2),
  );
  let p = glm::ext::perspective(
    0.5236,
    16.0 / 9.0,
    0.1,
    100.0,
  );
  let vp = p * v;
  gl::UniformMatrix4fv(uni_vp, 1, gl::FALSE, vp.as_array().as_ptr().cast());

  // Light
  let light_pos = (6.0, -3.0, 6.0);
  let light_pos = (3.7, -6.1, 5.0);
  let uni_light_pos = gl::GetUniformLocation(prog, "light_pos\0".as_ptr().cast());
  gl::Uniform3f(uni_light_pos, light_pos.0, light_pos.1, light_pos.2);
  let uni_cam_pos = gl::GetUniformLocation(prog, "cam_pos\0".as_ptr().cast());
  gl::Uniform3f(uni_cam_pos, cam_pos.0, cam_pos.1, cam_pos.2);

  check_gl_errors();

  gl::Enable(gl::DEPTH_TEST);
  gl::DepthFunc(gl::LESS);

  gl::Enable(gl::CULL_FACE);

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
    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

    gl::DrawArrays(gl::TRIANGLES, 0, frame.vertices.len() as gl::int);
    check_gl_errors();
  }

  Ok(())
}
