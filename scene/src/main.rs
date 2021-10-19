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

const W: u32 = 960;
const H: u32 = 540;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

  glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
  glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
  glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

  let (mut window, events) = glfw.create_window(
    W, H,
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

  // Load frames
  let mut frames = vec![];
  for i in 1..=1 {
    frames.push(scene_loader::load(format!("1a/1a_{:0>6}.obj", i))?);
  }
  // let max_num_vertices =
  //   frames.iter().map(|frame| frame.vertices.len()).max().unwrap_or(0);

  gl::EnableVertexAttribArray(0);
  gl::EnableVertexAttribArray(1);

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

  // Uniform locations
  let uni_vp = gl::GetUniformLocation(prog, "VP\0".as_ptr().cast());
  let uni_light_pos = gl::GetUniformLocation(prog, "light_pos\0".as_ptr().cast());
  let uni_cam_pos = gl::GetUniformLocation(prog, "cam_pos\0".as_ptr().cast());

  gl::Enable(gl::DEPTH_TEST);
  gl::DepthFunc(gl::LESS);

  gl::Enable(gl::CULL_FACE);

  check_gl_errors();

  let frame_len = 1.0 / 48.0;
  let mut last_time = glfw.get_time() as f32;
  let mut accum_time = 0.0;
  let mut frame_num = 0;

  let mut cam_pos = glm::vec3(9.02922, -8.50027, 7.65063);
  let mut cam_up = glm::normalize(glm::vec3(8.72799, -8.21633, 8.48306) - cam_pos);
  let mut cam_ori = glm::normalize(glm::vec3(4.01535, -3.77411, 4.22417) - cam_pos);

  let p_mat = glm::ext::perspective(
    0.5236,
    16.0 / 9.0,
    0.1,
    100.0,
  );

  // Hide cursor
  window.set_cursor_mode(glfw::CursorMode::Disabled);
  let mut last_cursor = window.get_cursor_pos();

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

    // Update with time
    let cur_time = glfw.get_time() as f32;
    let delta_time = cur_time - last_time;
    accum_time += delta_time;
    last_time = cur_time;
    while accum_time >= frame_len {
      accum_time -= frame_len;
      frame_num = (frame_num + 1) % frames.len();
    }

    // Frame data
    let frame = &frames[frame_num];
    gl::VertexAttribPointer(
      0,
      3, gl::FLOAT, gl::FALSE,
      size_of_val(&frame.vertices[0]) as gl::int,
      0 as *const _,
    );
    gl::VertexAttribPointer(
      1,
      3, gl::FLOAT, gl::FALSE,
      size_of_val(&frame.vertices[0]) as gl::int,
      (3 * size_of::<f32>()) as *const _,
    );
    gl::BufferData(
      gl::ARRAY_BUFFER,
      size_of_val(&*frame.vertices) as isize,
      frame.vertices.as_ptr().cast(),
      gl::STREAM_DRAW,
    );

    // Camera panning
    let move_dist = delta_time * 10.0;
    if window.get_key(glfw::Key::W) == glfw::Action::Press {
      cam_pos = cam_pos + cam_ori * move_dist;
    }
    if window.get_key(glfw::Key::S) == glfw::Action::Press {
      cam_pos = cam_pos - cam_ori * move_dist;
    }
    if window.get_key(glfw::Key::A) == glfw::Action::Press {
      cam_pos = cam_pos - glm::cross(cam_ori, cam_up) * move_dist;
    }
    if window.get_key(glfw::Key::D) == glfw::Action::Press {
      cam_pos = cam_pos + glm::cross(cam_ori, cam_up) * move_dist;
    }
    if window.get_key(glfw::Key::Tab) == glfw::Action::Press {
      cam_pos = cam_pos + cam_up * move_dist;
    }
    if window.get_key(glfw::Key::LeftShift) == glfw::Action::Press {
      cam_pos = cam_pos - cam_up * move_dist;
    }

    // Camera rotation
    let (x, y) = window.get_cursor_pos();
    let (dx, dy) = (last_cursor.0 - x, last_cursor.1 - y);
    last_cursor = (x, y);
    if dx.abs() >= 0.25 || dy.abs() >= 0.25 {
      let rotate_speed = 1.0 / 480.0;
      let cam_right = glm::cross(cam_ori, cam_up);
      // X
      let angle = dx as f32 * rotate_speed;
      let (cos_a, sin_a) = (angle.cos(), angle.sin());
      // cross(up, ori) = -right
      cam_ori = cam_ori * cos_a - cam_right * sin_a;
      // Y
      let angle = dy as f32 * rotate_speed;
      let (cos_a, sin_a) = (angle.cos(), angle.sin());
      // cross(right, ori) = up
      cam_ori = cam_ori * cos_a + cam_up * sin_a;
      // In case drift happens
      cam_ori = glm::normalize(cam_ori);
    }

    // Camera matrix
    let v = glm::ext::look_at(cam_pos, cam_pos + cam_ori, cam_up);
    let vp = p_mat * v;
    gl::UniformMatrix4fv(uni_vp, 1, gl::FALSE, vp.as_array().as_ptr().cast());
    gl::Uniform3f(uni_cam_pos, cam_pos.x, cam_pos.y, cam_pos.z);

    // Light
    let light_pos = glm::vec3(6.0, -3.0, 6.0);
    gl::Uniform3f(uni_light_pos, light_pos.x, light_pos.y, light_pos.z);

    // Draw
    gl::ClearColor(1.0, 0.99, 0.99, 1.0);
    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

    gl::DrawArrays(gl::TRIANGLES, 0, frame.vertices.len() as gl::int);
    check_gl_errors();
  }

  Ok(())
}
