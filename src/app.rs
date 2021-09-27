use eframe::{egui, epi};

#[derive(Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct Polygon {
  khroma: [f32; 3],
  cycles: Vec<Vec<(f32, f32)>>,
}

#[derive(Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct App {
  dark: bool,

  show_intersection: bool,
  intersection_khroma: [f32; 4],

  polygons: Vec<Polygon>,
  sel_polygon: Option<usize>,
  polygons_collapsed: bool,

  cur_cycle: Option<Vec<(f32, f32)>>,

  last_added_polygon: bool,

  last_rect_popup: [egui::Rect; 2],
  last_pt_down: (bool, bool),
  last_pt_held: (bool, bool),
}

fn rand_khroma() -> [f32; 3] {
  [0.5, 0.5, 1.0]
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
  ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)).sqrt()
}

impl Default for App {
  fn default() -> Self {
    let mut result = Self {
      dark: false,

      show_intersection: false,
      intersection_khroma: {
        let k = rand_khroma();
        [k[0], k[1], k[2], 0.6]
      },

      polygons: vec![],
      sel_polygon: None,
      polygons_collapsed: false,

      cur_cycle: None,

      last_added_polygon: false,

      last_rect_popup: [egui::Rect::NOTHING; 2],
      last_pt_down: (false, false),
      last_pt_held: (false, false),
    };
    result.add_polygon();
    result
  }
}

impl App {
  fn update_theme(&self, ctx: &egui::CtxRef) {
    ctx.set_visuals(
      if self.dark { egui::style::Visuals::dark() }
      else { egui::style::Visuals::light() }
    );
  }

  fn add_polygon(&mut self) {
    self.polygons.push(Polygon {
      khroma: rand_khroma(),
      cycles: vec![],
    });
    self.sel_polygon = Some(self.polygons.len() - 1);
    self.cur_cycle = None;
  }

  fn process_canvas_interactions(
    &mut self,
    painter: egui::Painter,
    rect: egui::Rect,
    exclude: &[egui::Rect],
    resp: egui::Response,
    input: &egui::InputState,
  ) {
    // Draw polygons
    // TODO
    if self.sel_polygon.is_some() && !self.polygons_collapsed {
      painter.add(egui::Shape::circle_filled(
        rect.center(), 480.0, egui::Color32::from_rgb(64, 128, 255)));
    } else {
      painter.add(egui::Shape::circle_stroke(
        rect.center(), 480.0,
        egui::Stroke::new(6.0, egui::Color32::from_rgb(64, 128, 255))));
    }
    // Current cycle
    if let Some(cyc) = &self.cur_cycle {
      for vert in cyc {
        painter.circle_filled(
          vert.into(),
          6.0,
          egui::Color32::from_rgb(128, 255, 64),
        );
      }
    }

    // Pointer interactions
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
    let excluded = !rect.contains(pt_pos) ||
      exclude.iter().any(|rect| rect.contains(pt_pos));
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

    if pt1_press {
      // Adding new point or dragging?
      let mut dragging = None;
      if self.cur_cycle.is_none() && self.sel_polygon.is_some() {
        // Find a point in the polygon
        let poly = &self.polygons[self.sel_polygon.unwrap()];
        'outer: for (i, cyc) in poly.cycles.iter().enumerate() {
          for (j, vert) in cyc.iter().enumerate() {
            if dist(*vert, pt_pos.into()) <= 6.0 {
              dragging = Some((i, j));
              break 'outer;
            }
          }
        }
      }
      if let Some((i, j)) = dragging {
        // Dragging
      } else if self.sel_polygon.is_some() {
        // Adding new point
        let cyc = self.cur_cycle.get_or_insert(vec![]);
        cyc.push(pt_pos.into());
      }
    }
    if pt2_press {
      if let Some(cyc) = self.cur_cycle.take() {
        self.polygons[self.sel_polygon.unwrap()].cycles.push(cyc);
      }
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
      let width = ui.available_width();
      let height = ui.available_height();

      let rect_painter = ui.available_rect_before_wrap();
      let painter = egui::Painter::new(
        ui.ctx().clone(),
        // egui::layers::LayerId::new(egui::layers::Order::Foreground, egui::Id::new("1111")),
        ui.layer_id(),
        rect_painter);
      painter.add(egui::Shape::rect_filled(
        rect_painter, 6.0, egui::Color32::from_rgba_premultiplied(8, 8, 8, 16)));

      let exclude = self.last_rect_popup;
      self.process_canvas_interactions(
        painter,
        rect_painter,
        &exclude,
        ui.allocate_rect(rect_painter, egui::Sense::hover()),
        ui.input(),
      );

      let rect_popup = egui::Rect::from_min_size(
        egui::pos2(16.0, 16.0), egui::Vec2::ZERO);
      ui.allocate_ui_at_rect(rect_popup, |ui| {
        egui::Frame::popup(ui.style())
          .fill(
            if self.dark { egui::Color32::from_rgba_premultiplied(32, 32, 32, 216) }
            else { egui::Color32::from_rgba_premultiplied(216, 216, 216, 216) }
          )
          .show(ui, |ui| {
            ui.set_max_width(width);
            ui.horizontal(|ui| {
              ui.checkbox(&mut self.show_intersection, "Intersection");
              ui.color_edit_button_rgba_unmultiplied(&mut self.intersection_khroma);
            });
            if egui::CollapsingHeader::new("Polygons").default_open(true).show(ui, |ui| {
              let mut polygon_remove = None;
              ui.allocate_ui(egui::vec2(180.0, 240.0), |ui| {
                egui::ScrollArea::auto_sized().show(ui, |ui| {
                  for (index, poly) in self.polygons.iter_mut().enumerate() {
                    let sel = match self.sel_polygon {
                      Some(i) => i == index,
                      _ => false,
                    };
                    ui.horizontal(|ui| {
                      if ui.add_sized(
                        egui::vec2(160.0, ui.style().spacing.interact_size.y),
                        egui::SelectableLabel::new(sel,
                          format!("#{} ({} vert, {} cyc)",
                            index + 1,
                            poly.cycles.iter().map(|c| c.len()).sum::<usize>(),
                            poly.cycles.len(),
                          ))
                      ).clicked() {
                        self.sel_polygon = if sel { None } else { Some(index) };
                        self.cur_cycle = None;
                      }
                      // Recalculate selection state to avoid UI jumping
                      let sel = match self.sel_polygon {
                        Some(i) => i == index,
                        _ => false,
                      };
                      ui.color_edit_button_rgb(&mut poly.khroma);
                      if sel {
                        if ui.selectable_label(false, "×")
                             .on_hover_text("Remove").clicked() {
                          polygon_remove = Some(index);
                        }
                      }
                    });
                  }
                  if self.last_added_polygon {
                    ui.scroll_to_cursor(egui::Align::BOTTOM);
                  }
                });
              });
              if let Some(index) = polygon_remove {
                self.polygons.remove(index);
                if let Some(i) = self.sel_polygon {
                  if i > index {
                    self.sel_polygon = Some(i - 1);
                  } else if i == index {
                    self.sel_polygon = None;
                  }
                } else {
                  self.sel_polygon = None;
                }
              }
              if ui.add_sized(
                egui::vec2(211.0, ui.style().spacing.interact_size.y),
                egui::Button::new("☆ new")).clicked()
              {
                self.add_polygon();
                self.last_added_polygon = true;
              } else {
                self.last_added_polygon = false;
              }
            }).body_response.is_none() {
              self.polygons_collapsed = true;
            } else {
              self.polygons_collapsed = false;
            }
          });
        self.last_rect_popup[0] = ui.min_rect();
      });
      let rect_popup = egui::Rect::from_min_size(
        egui::pos2(12.0, height - ui.style().spacing.interact_size.y - 12.0),
        egui::Vec2::ZERO);
      ui.allocate_ui_at_rect(rect_popup, |ui| {
        egui::Frame::popup(ui.style())
          .fill(
            if self.dark { egui::Color32::from_rgba_premultiplied(32, 32, 32, 216) }
            else { egui::Color32::from_rgba_premultiplied(216, 216, 216, 216) }
          )
          .show(ui, |ui| {
            ui.set_max_width(width);
            if ui.selectable_label(false, "dark/light").clicked() {
              self.dark = !self.dark;
              self.update_theme(ctx);
            }
          });
        self.last_rect_popup[1] = ui.min_rect();
      });
    });
  }
}
