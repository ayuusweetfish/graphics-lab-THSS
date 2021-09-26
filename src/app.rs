use eframe::{egui, epi};

#[derive(Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct App {
  dark: bool,
  polygon_khroma: [f32; 3],

  last_rect_popup: egui::Rect,
  last_pt_down: (bool, bool),
  last_pt_held: (bool, bool),
}

impl Default for App {
  fn default() -> Self {
    Self {
      dark: false,
      polygon_khroma: [0f32; 3],

      last_rect_popup: egui::Rect::NOTHING,
      last_pt_down: (false, false),
      last_pt_held: (false, false),
    }
  }
}

impl App {
  fn update_theme(&self, ctx: &egui::CtxRef) {
    ctx.set_visuals(
      if self.dark { egui::style::Visuals::dark() }
      else { egui::style::Visuals::light() }
    );
  }

  fn process_canvas_interactions(
    &mut self,
    painter: egui::Painter,
    rect: egui::Rect,
    exclude: egui::Rect,
    resp: egui::Response,
    input: &egui::InputState,
  ) {
    let pt_pos = match input.pointer.hover_pos().or(input.pointer.interact_pos()) {
      Some(p) => p,
      None => return, // Return if no position information is available
    };

    // Button states: buttons themselves
    let (last_pt1, last_pt2) = self.last_pt_down;
    let pt1 = input.pointer.primary_down();
    let pt2 = input.pointer.secondary_down();
    self.last_pt_down = (pt1, pt2);

    // Whether buttons are pressed (up -> down)
    let pt1_press = !last_pt1 && pt1;
    let pt2_press = !last_pt2 && pt2;
    // Whether buttons are released (down -> up)
    let (last_pt1_held, last_pt2_held) = self.last_pt_held;
    let pt1_rel = last_pt1_held && !pt1;
    let pt2_rel = last_pt2_held && !pt2;

    // Hold states: valid button presses and drags in the region
    let excluded = exclude.contains(pt_pos) || !rect.contains(pt_pos);
    if !pt1 { self.last_pt_held.0 = false; }
    else if pt1_press && !excluded { self.last_pt_held.0 = true; }
    if !pt2 { self.last_pt_held.1 = false; }
    else if pt2_press && !excluded { self.last_pt_held.1 = true; }
    // Return if in the excluded range and not dragging
    if excluded && (!last_pt1_held && !last_pt2_held) { return; }

    let (pt1_held, pt2_held) = self.last_pt_held;

    // Process events
    if pt1_press { println!("press"); }
    if pt1_rel { println!("release"); }
    painter.add(egui::Shape::circle_filled(pt_pos, 10.0,
      if pt1_held { egui::Color32::from_rgb(255, 128, 128) }
      else { egui::Color32::from_rgb(255, 255, 128) }));
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
    style.spacing.interact_size.y =
      // ctx.fonts().layout_single_line(egui::TextStyle::Button, "A".into()).size.y +
      16.0 +
      style.spacing.button_padding.y * 2.0;
    style.visuals.popup_shadow = egui::epaint::Shadow {
      extrusion: 6.0, color: egui::Color32::from_black_alpha(32)
    };
    ctx.set_style(style);

    self.update_theme(ctx);

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
      let rect_painter = ui.available_rect_before_wrap();
      let painter = egui::Painter::new(
        ui.ctx().clone(),
        // egui::layers::LayerId::new(egui::layers::Order::Foreground, egui::Id::new("1111")),
        ui.layer_id(),
        rect_painter);
      painter.add(egui::Shape::rect_filled(
        rect_painter, 6.0, egui::Color32::from_rgba_premultiplied(8, 8, 8, 16)));
      painter.add(egui::Shape::circle_filled(
        rect_painter.center(), 480.0, egui::Color32::from_rgb(64, 128, 255)));

      self.process_canvas_interactions(
        painter,
        rect_painter,
        self.last_rect_popup,
        ui.allocate_rect(rect_painter, egui::Sense::hover()),
        ui.input(),
      );

      let rect_popup = egui::Rect {
        min: egui::pos2(16.0, 16.0),
        max: egui::pos2(16.0, 16.0),
      };
      let width = ui.available_width();
      ui.allocate_ui_at_rect(rect_popup, |ui| {
        egui::Frame::popup(ui.style())
          .fill(
            if self.dark { egui::Color32::from_rgba_premultiplied(32, 32, 32, 216) }
            else { egui::Color32::from_rgba_premultiplied(216, 216, 216, 216) }
          )
          .show(ui, |ui| {
            ui.set_max_width(width);
            if ui.button("dark/light").clicked() {
              self.dark = !self.dark;
              self.update_theme(ctx);
            }
            egui::CollapsingHeader::new("Polygons").default_open(true).show(ui, |ui| {
              for i in 0..3 {
                ui.horizontal(|ui| {
                  ui.add_sized(
                    egui::vec2(80.0, ui.style().spacing.interact_size.y),
                    egui::SelectableLabel::new(i == 2, format!("#{} (10)", i + 1)));
                  ui.color_edit_button_rgb(&mut self.polygon_khroma);
                  if i == 2 {
                    ui.selectable_label(false, "×");
                  }
                });
              }
              ui.add_sized(
                egui::vec2(158.0, ui.style().spacing.interact_size.y),
                egui::Button::new("☆ new"));
            });
          });
        self.last_rect_popup = ui.min_rect();
      });
    });
  }
}
