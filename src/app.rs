use eframe::{egui, epi};
use rand::Rng;

#[derive(Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct Polygon {
  khroma: [f32; 3],
  cycles: Vec<Vec<(f32, f32)>>,
}

#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
enum DraggingVert {
  None,
  PolygonCycle(usize, usize),
  CurCycle(usize),
}

#[derive(Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct App {
  dark: bool,

  intersection_khroma: [f32; 4],

  polygons: Vec<Polygon>,
  polygons_visible: Vec<bool>,
  sel_polygon: Option<usize>,
  polygons_collapsed: bool,

  cur_cycle: Option<Vec<(f32, f32)>>,
  dragging_vert: DraggingVert,
  drag_offset: (f32, f32),

  last_added_polygon: bool,

  last_rect_popup: [egui::Rect; 2],
  last_pt_down: (bool, bool),
  last_pt_held: (bool, bool),

  rng: rand::rngs::ThreadRng,
}

fn rand_khroma<T: rand::Rng>(rng: &mut T) -> [f32; 3] {
  let h = rng.gen_range(0.0..1.0);
  let s = rng.gen_range(0.4..0.8);
  let v = rng.gen_range(0.5..0.8);
  egui::color::rgb_from_hsv((h, s, v))
}

fn to_rgba32(k: [f32; 3]) -> egui::Color32 {
  egui::Color32::from_rgb(
    egui::color::gamma_u8_from_linear_f32(k[0]),
    egui::color::gamma_u8_from_linear_f32(k[1]),
    egui::color::gamma_u8_from_linear_f32(k[2]),
  )
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
  ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1))
}
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 { dist_sq(a, b).sqrt() }
fn diff(a: (f32, f32), b: (f32, f32)) -> (f32, f32) { (a.0 - b.0, a.1 - b.1) }
fn det(a: (f32, f32), b: (f32, f32)) -> f32 { a.0 * b.1 - a.1 * b.0 }
fn det3(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 { det(diff(b, a), diff(c, a)) }
fn dot(a: (f32, f32), b: (f32, f32)) -> f32 { a.0 * b.0 + a.1 * b.1 }
fn lerp(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
  (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}
fn project(a: (f32, f32), p: (f32, f32), q: (f32, f32)) -> (f32, f32) {
  let l_sq = dist_sq(p, q);
  let t = dot(diff(a, p), diff(q, p)) / l_sq;
  lerp(p, q, t)
}
fn dist_to_seg(a: (f32, f32), p: (f32, f32), q: (f32, f32)) -> f32 {
  let l_sq = dist_sq(p, q);
  if l_sq == 0.0 { return dist(a, p); }
  let t = dot(diff(a, p), diff(q, p)) / l_sq;
  let t = match t {
    f32::MIN..=0.0 => 0.0,
    1.0..=f32::MAX => 1.0,
    _ => t,
  };
  dist(a, lerp(p, q, t))
}

impl Default for App {
  fn default() -> Self {
    let mut rng = rand::thread_rng();
    let mut result = Self {
      dark: false,

      intersection_khroma: {
        let k = rand_khroma(&mut rng);
        [k[0], k[1], k[2], 0.6]
      },

      polygons: vec![],
      polygons_visible: vec![],
      sel_polygon: None,
      polygons_collapsed: false,

      cur_cycle: None,
      dragging_vert: DraggingVert::None,
      drag_offset: (0.0, 0.0),

      last_added_polygon: false,

      last_rect_popup: [egui::Rect::NOTHING; 2],
      last_pt_down: (false, false),
      last_pt_held: (false, false),

      rng,
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
      khroma: rand_khroma(&mut self.rng),
      cycles: vec![],
    });
    self.polygons_visible.push(true);
    self.sel_polygon = Some(self.polygons.len() - 1);
    self.cur_cycle = None;
    self.dragging_vert = DraggingVert::None;
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
    /*
    if self.sel_polygon.is_some() && !self.polygons_collapsed {
      painter.add(egui::Shape::circle_filled(
        rect.center(), 480.0, egui::Color32::from_rgb(64, 128, 255)));
    } else {
      painter.add(egui::Shape::circle_stroke(
        rect.center(), 480.0,
        egui::Stroke::new(6.0, egui::Color32::from_rgb(64, 128, 255))));
    }
    */
    for (poly_index, poly) in self.polygons.iter().enumerate() {
      let kh = to_rgba32(poly.khroma);
      let sel = (self.sel_polygon == Some(poly_index));
      for (i, cyc) in poly.cycles.iter().enumerate() {
        // Segments
        for j in 0..cyc.len() {
          painter.line_segment(
            [cyc[j].into(), cyc[(j + 1) % cyc.len()].into()],
            if sel {
              egui::Stroke::new(6.0, kh)
            } else {
              let [r, g, b, _] = kh.to_array();
              egui::Stroke::new(4.0, egui::Color32::from_rgba_unmultiplied(
                r, g, b,
                128))
            }
          );
        }
        // Vertices
        for j in 0..cyc.len() {
          painter.circle_filled(cyc[j].into(),
            if sel { 6.0 } else { 2.0 },
            if sel && self.dragging_vert == DraggingVert::PolygonCycle(i, j) {
              egui::Color32::from_rgb(255, 192, 128)
            } else {
              kh
            }
          );
        }
      }
    }
    // Current cycle
    if let Some(cyc) = &self.cur_cycle {
      for vert in cyc {
        painter.circle_filled(
          vert.into(),
          6.0,
          egui::Color32::from_rgb(128, 192, 64),
        );
      }
      for verts in cyc.windows(2) {
        if let [v1, v2] = verts {
          painter.line_segment(
            [v1.into(), v2.into()],
            egui::Stroke::new(6.0, egui::Color32::from_rgb(128, 192, 64)),
          );
        }
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
    /*
    painter.add(egui::Shape::circle_filled(pt_pos, 10.0,
      if pt1_held { egui::Color32::from_rgb(255, 128, 128) }
      else { egui::Color32::from_rgb(255, 255, 128) }));
    */

    let find_vertex = |poly: &mut Polygon| {
      // Find a point in the polygon
      for (i, cyc) in poly.cycles.iter().enumerate() {
        for (j, vert) in cyc.iter().enumerate() {
          if dist(*vert, pt_pos.into()) <= 6.0 {
            return Some((i, j));
          }
        }
      }
      None
    };
    if pt1_press {
      // Adding new point or dragging?
      let mut dragging = None;
      if self.sel_polygon.is_some() {
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        if self.cur_cycle.is_none() {
          dragging = find_vertex(poly);
        }
        if dragging.is_none() {
          // Or creating new point?
          'outer: for (i, cyc) in poly.cycles.iter_mut().enumerate() {
            for j in 0..cyc.len() {
              let p = cyc[j];
              let q = cyc[(j + 1) % cyc.len()];
              if dist_to_seg(pt_pos.into(), p, q) <= 6.0 {
                cyc.insert(j + 1, project(pt_pos.into(), p, q));
                dragging = Some((i, j + 1));
                break 'outer;
              }
            }
          }
        }
      }
      if let Some((i, j)) = dragging {
        // Dragging
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        self.dragging_vert = DraggingVert::PolygonCycle(i, j);
        self.drag_offset = diff(pt_pos.into(), poly.cycles[i][j]);
      } else if self.sel_polygon.is_some() {
        // Adding new point
        let cyc = self.cur_cycle.get_or_insert(vec![]);
        cyc.push(pt_pos.into());
        self.dragging_vert = DraggingVert::CurCycle(cyc.len() - 1);
        self.drag_offset = (0.0, 0.0);
      } else {
        self.dragging_vert = DraggingVert::None;
        self.drag_offset = (0.0, 0.0);
      }
    } else if pt1 {
      if let DraggingVert::PolygonCycle(i, j) = self.dragging_vert {
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        poly.cycles[i][j] = diff(pt_pos.into(), self.drag_offset);
      } else if let DraggingVert::CurCycle(j) = self.dragging_vert {
        self.cur_cycle.as_deref_mut().unwrap()[j] = diff(pt_pos.into(), self.drag_offset);
      }
    } else if pt1_rel {
      self.dragging_vert = DraggingVert::None;
    }
    if pt2_press {
      if let Some(cyc) = self.cur_cycle.take() {
        // Commit new cycle
        if cyc.len() >= 3 {
          self.polygons[self.sel_polygon.unwrap()].cycles.push(cyc);
          self.dragging_vert = DraggingVert::None;
        }
      } else if self.sel_polygon.is_some() {
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        if let Some((i, j)) = find_vertex(poly) {
          // Remove vertex
          poly.cycles[i].remove(j);
          if poly.cycles[i].is_empty() {
            poly.cycles.remove(i);
          }
        }
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
            if egui::CollapsingHeader::new("Polygons").default_open(true).show(ui, |ui| {
              let mut polygon_remove = None;
              ui.allocate_ui(egui::vec2(180.0, 240.0), |ui| {
                egui::ScrollArea::auto_sized().show(ui, |ui| {
                  ui.horizontal(|ui| {
                    ui.add_sized(
                      egui::vec2(160.0, ui.style().spacing.interact_size.y),
                      egui::Label::new("Intersection"),
                    );
                    ui.color_edit_button_rgba_unmultiplied(&mut self.intersection_khroma);
                    let any_visible = self.polygons_visible.iter().any(|x| *x);
                    if ui.selectable_label(any_visible, "\u{25cb}")
                         .on_hover_text("Visibility").clicked() {
                      for value in self.polygons_visible.iter_mut() {
                        *value = !any_visible;
                      }
                    }
                  });
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
                        self.dragging_vert = DraggingVert::None;
                      }
                      // Recalculate selection state to avoid UI jumping
                      let sel = match self.sel_polygon {
                        Some(i) => i == index,
                        _ => false,
                      };
                      ui.color_edit_button_rgb(&mut poly.khroma);
                      if ui.selectable_label(self.polygons_visible[index], "\u{25cb}")
                           .on_hover_text("Visibility").clicked() {
                        self.polygons_visible[index] = !self.polygons_visible[index];
                      }
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
                self.polygons_visible.remove(index);
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
