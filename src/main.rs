use eframe::egui;
mod app;
mod intersect;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
  let app = app::App::default();
  let options = eframe::NativeOptions {
    initial_window_size: Some(egui::Vec2::new(960.0, 640.0)),
    ..Default::default()
  };
  eframe::run_native(Box::new(app), options);
}

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), eframe::wasm_bindgen::JsValue> {
  let app = app::App::default();
  eframe::start_web(canvas_id, Box::new(app))
}
