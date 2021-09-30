use eframe::egui;
mod app;
mod geom;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
  let app = app::App::default();
  let options = eframe::NativeOptions {
    initial_window_size: Some(egui::Vec2::new(960.0, 640.0)),
    ..Default::default()
  };
  eframe::run_native(Box::new(app), options);
}
