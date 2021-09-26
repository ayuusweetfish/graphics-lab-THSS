use eframe::{egui, epi};

#[derive(Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct App {
  x: f32,
}

impl Default for App {
  fn default() -> Self {
    Self {
      x: 0.0,
    }
  }
}

impl epi::App for App {
  fn name(&self) -> &str {
    "Polygon Intersection"
  }

  fn setup(
    &mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>,
    storage: Option<&dyn epi::Storage>,
  ) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
      "qisixihei".to_owned(),
      std::borrow::Cow::Borrowed(include_bytes!("../fonts/1570788235.ttf")));
    fonts.fonts_for_family.get_mut(&egui::FontFamily::Proportional).unwrap()
      .insert(0, "qisixihei".to_owned());
    fonts.fonts_for_family.get_mut(&egui::FontFamily::Monospace).unwrap()
      .insert(0, "qisixihei".to_owned());
    *fonts.family_and_size.get_mut(&egui::TextStyle::Small).unwrap()
      = (egui::FontFamily::Proportional, 13.0);
    *fonts.family_and_size.get_mut(&egui::TextStyle::Body).unwrap()
      = (egui::FontFamily::Proportional, 16.0);
    *fonts.family_and_size.get_mut(&egui::TextStyle::Button).unwrap()
      = (egui::FontFamily::Proportional, 16.0);
    *fonts.family_and_size.get_mut(&egui::TextStyle::Heading).unwrap()
      = (egui::FontFamily::Proportional, 22.0);
    *fonts.family_and_size.get_mut(&egui::TextStyle::Monospace).unwrap()
      = (egui::FontFamily::Proportional, 16.0);
    ctx.set_fonts(fonts);

    let mut style: egui::Style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 5.0);
    style.spacing.window_padding = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(5.0, 2.5);
    style.spacing.slider_width = 40.0;
    style.visuals.popup_shadow = egui::epaint::Shadow {
      extrusion: 6.0, color: egui::Color32::from_black_alpha(32)
    };
    ctx.set_style(style);

    #[cfg(feature = "persistence")]
    if let Some(storage) = storage {
      *self = epi::get_value(storage, epi::APP_KEY).unwrap_or_default()
    }
  }

  #[cfg(feature = "persistence")]
  fn save(&mut self, storage: &mut dyn epi::Storage) {
    epi::set_value(storage, epi::APP_KEY, self);
  }

  #[cfg(feature = "persistence")]
  fn auto_save_interval(&self) -> std::time::Duration {
    std::time::Duration::from_secs(10)
  }

  fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.add(egui::DragValue::new(&mut self.x).clamp_range(0.0..=10.0).speed(0.1));
    });
  }
}
