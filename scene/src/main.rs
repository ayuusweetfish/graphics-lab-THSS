mod gl;

use glfw::Context;

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

  window.set_key_polling(true);
  window.make_current();

  let mut vao = 0;
  gl::GenVertexArrays(1, &mut vao);
  println!("{}", vao);

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
  }
}
