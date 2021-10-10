use eframe::{egui, epi};
use crate::geom::*;

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
  last_pt_pos: (f32, f32),

  rng: (rand::rngs::ThreadRng, f32),

  canvas_shapes_cache: Vec<egui::Shape>,
  canvas_shapes_obsolete: bool,

  guide: usize,
}

fn rand_khroma<T: rand::Rng>(rng: &mut (T, f32)) -> [f32; 3] {
  let (rng, last) = rng;
  let h = rng.gen_range((*last + 0.2)..(*last + 0.8)) % 1.0;
  let s = rng.gen_range(0.4..0.8);
  let v = rng.gen_range(0.5..0.8);
  *last = h;
  egui::color::rgb_from_hsv((h, s, v))
}

fn to_rgba32<const N: usize>(k: [f32; N]) -> egui::Color32 {
  if N == 3 {
    egui::Color32::from_rgb(
      egui::color::gamma_u8_from_linear_f32(k[0]),
      egui::color::gamma_u8_from_linear_f32(k[1]),
      egui::color::gamma_u8_from_linear_f32(k[2]),
    )
  } else if N == 4 {
    egui::Color32::from_rgba_unmultiplied(
      egui::color::gamma_u8_from_linear_f32(k[0]),
      egui::color::gamma_u8_from_linear_f32(k[1]),
      egui::color::gamma_u8_from_linear_f32(k[2]),
      egui::color::linear_u8_from_linear_f32(k[3]),
    )
  } else {
    unimplemented!()
  }
}

fn fill_polygon(shapes: &mut Vec<egui::Shape>, polygon: &[Vec<(f32, f32)>], kh: egui::Color32) {
  // Split the polygon into disjoint components
  let comps = normalize_polygon(polygon);
  for comp in comps {
    let v: Vec<_> =
      comp.iter().map(|v| v.iter().map(|&(a, b)| vec![a, b]).collect()).collect();
    if v.is_empty() { continue; }
    let (verts, holes, dims) = earcutr::flatten(&v);
    let tris = earcutr::earcut(&verts, &holes, dims);
    for tri in tris.chunks_exact(3) {
      assert!(tri.len() == 3);
      // Fill a triangle
      shapes.push(egui::Shape::convex_polygon(
        vec![
          (verts[tri[0] * 2], verts[tri[0] * 2 + 1]).into(),
          (verts[tri[1] * 2], verts[tri[1] * 2 + 1]).into(),
          (verts[tri[2] * 2], verts[tri[2] * 2 + 1]).into(),
        ],
        kh,
        egui::Stroke::none(),
      ));
    }
  }
}

impl Default for App {
  fn default() -> Self {
    let mut rng = (rand::thread_rng(), 0.0);
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
      last_pt_pos: (f32::NAN, f32::NAN),

      rng,

      canvas_shapes_cache: vec![],
      canvas_shapes_obsolete: true,

      guide: 0,
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
    self.canvas_shapes_obsolete = true;
  }

  fn calc_canvas_shapes(&mut self) {
    if !self.canvas_shapes_obsolete { return; }
    self.canvas_shapes_obsolete = false;
    let shapes = &mut self.canvas_shapes_cache;
    shapes.clear();

    let mut self_intxns_any = false;
    // Draw polygons
    for (poly_index, poly) in self.polygons.iter().enumerate() {
      if !self.polygons_visible[poly_index] && self.sel_polygon != Some(poly_index) {
        continue;
      }

      // Check for self-intersections
      // Find out all segments
      let mut segs = vec![];
      for cyc in &poly.cycles {
        for j in 0..cyc.len() {
          segs.push((cyc[j].into(), cyc[(j + 1) % cyc.len()].into()));
        }
      }
      let self_intxns = all_segment_intersections(&segs);
      let self_intxns_cur = self_intxns.iter().any(|x| !x.is_empty());
      self_intxns_any |= self_intxns_cur;

      // Draw
      let kh = to_rgba32(poly.khroma);
      let sel = (self.sel_polygon == Some(poly_index)) && !self.polygons_collapsed;
      // Fill if currently selected
      if !self_intxns_cur && sel {
        fill_polygon(shapes, &poly.cycles,
          to_rgba32([poly.khroma[0], poly.khroma[1], poly.khroma[2], 0.6]));
      }
      for (i, cyc) in poly.cycles.iter().enumerate() {
        // Segments
        for j in 0..cyc.len() {
          shapes.push(egui::Shape::line_segment(
            [cyc[j].into(), cyc[(j + 1) % cyc.len()].into()],
            if sel {
              egui::Stroke::new(6.0, kh)
            } else {
              let [r, g, b, _] = kh.to_array();
              egui::Stroke::new(4.0, egui::Color32::from_rgba_unmultiplied(
                r, g, b,
                128))
            }
          ));
        }
        // Vertices
        for j in 0..cyc.len() {
          shapes.push(egui::Shape::circle_filled(cyc[j].into(),
            if sel { 6.0 } else { 2.0 },
            if sel && self.dragging_vert == DraggingVert::PolygonCycle(i, j) {
              egui::Color32::from_rgb(255, 192, 128)
            } else {
              kh
            }
          ));
        }
      }

      // Self-intersections highlighting
      if self_intxns_cur {
        let kh = egui::Color32::from_rgb(255, 144, 128);
        for (i, with_i) in self_intxns.iter().enumerate() {
          for &(j, p) in with_i {
            if j > i {
              shapes.push(egui::Shape::circle_filled(p.into(), 6.0, kh));
              shapes.push(egui::Shape::circle_stroke(p.into(), 9.0, (2.0, kh)));
            }
          }
        }
      }
    }
    // Intersection
    if !self_intxns_any {
      let intersection_polygons: Vec<&Vec<Vec<(f32, f32)>>> =
        self.polygons.iter().zip(self.polygons_visible.iter())
          .filter(|(_, &visible)| visible)
          .map(|(polygon, _)| &polygon.cycles)
          .collect();
      // println!("{:?}", intersection_polygons);
      let intersection = intersection(&intersection_polygons);
      let intersection_kh = to_rgba32(self.intersection_khroma);
      let intersection_kh_opaque = intersection_kh.to_opaque();
      // Fill
      fill_polygon(shapes, &intersection, intersection_kh);
      // Outline
      for cyc in intersection {
        for j in 0..cyc.len() {
          shapes.push(egui::Shape::line_segment(
            [cyc[j].into(), cyc[(j + 1) % cyc.len()].into()],
            egui::Stroke::new(3.0, intersection_kh_opaque),
          ));
        }
        for j in 0..cyc.len() {
          shapes.push(egui::Shape::circle_filled(cyc[j].into(),
            3.0, intersection_kh_opaque,
          ));
        }
      }
    }
    // Current cycle
    if let Some(cyc) = &self.cur_cycle {
      for vert in cyc {
        shapes.push(egui::Shape::circle_filled(
          vert.into(),
          6.0,
          egui::Color32::from_rgb(128, 192, 64),
        ));
      }
      for verts in cyc.windows(2) {
        if let [v1, v2] = verts {
          shapes.push(egui::Shape::line_segment(
            [v1.into(), v2.into()],
            egui::Stroke::new(6.0, egui::Color32::from_rgb(128, 192, 64)),
          ));
        }
      }
    }
  }

  fn process_canvas_interactions(
    &mut self,
    painter: egui::Painter,
    rect: egui::Rect,
    exclude: &[egui::Rect],
    _resp: egui::Response,
    input: &egui::InputState,
    popup: bool,
  ) {
    self.calc_canvas_shapes();
    painter.extend(self.canvas_shapes_cache.clone());

    const GUIDE_STRINGS: [&'static str; 4] = [
      "Left click to add at least 3 points",
      "Right click anywhere to close the ring",
      "Similarly add another ring for the same polygon",
      "In the top-left panel, create a new polygon and see their intersection",
    ];
    if self.guide < 4 {
      painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        GUIDE_STRINGS[self.guide],
        egui::TextStyle::Heading,
        egui::Color32::from_rgba_unmultiplied(128, 128, 128, 216),
      );
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

    // Ignore presses if popup is open
    let pt1_press = pt1_press && !popup;
    let pt2_press = pt2_press && !popup;

    // Pointer moved?
    let pt_moved = dist_sq(pt_pos.into(), self.last_pt_pos) >= 1e-6;
    self.last_pt_pos = pt_pos.into();

    // Process events
    let find_vertex_cycle = |cyc: &Vec<(f32, f32)>| {
      for (j, vert) in cyc.iter().enumerate() {
        if dist(*vert, pt_pos.into()) <= 6.0 {
          return Some(j);
        }
      }
      None
    };
    let find_vertex = |cycles: &Vec<Vec<(f32, f32)>>| {
      // Find a point in the polygon
      for (i, cyc) in cycles.iter().enumerate() {
        if let Some(j) = find_vertex_cycle(&cyc) {
          return Some((i, j));
        }
      }
      None
    };
    if pt1_press && !self.polygons_collapsed {
      // Adding new point or dragging?
      self.dragging_vert = DraggingVert::None;
      if self.sel_polygon.is_some() {
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        // XXX: Reduce duplication?
        if let Some(cyc) = &mut self.cur_cycle {
          // Editing the temporary cycle
          // Dragging
          if let Some(j) = find_vertex_cycle(&cyc) {
            self.dragging_vert = DraggingVert::CurCycle(j);
          }
          if self.dragging_vert == DraggingVert::None {
            // Creating a new point
            for j in 0..cyc.len() {
              let p = cyc[j];
              let q = cyc[(j + 1) % cyc.len()];
              if dist_to_seg(pt_pos.into(), p, q) <= 6.0 {
                cyc.insert(j + 1, project(pt_pos.into(), p, q));
                self.dragging_vert = DraggingVert::CurCycle(j + 1);
                break;
              }
            }
          }
        } else {
          // Editing the existing polygon
          // Dragging
          if let Some((i, j)) = find_vertex(&mut poly.cycles) {
            self.dragging_vert = DraggingVert::PolygonCycle(i, j);
          }
          if self.dragging_vert == DraggingVert::None {
            // Creating a new point
            'outer: for (i, cyc) in poly.cycles.iter_mut().enumerate() {
              for j in 0..cyc.len() {
                let p = cyc[j];
                let q = cyc[(j + 1) % cyc.len()];
                if dist_to_seg(pt_pos.into(), p, q) <= 6.0 {
                  cyc.insert(j + 1, project(pt_pos.into(), p, q));
                  self.dragging_vert = DraggingVert::PolygonCycle(i, j + 1);
                  break 'outer;
                }
              }
            }
          }
        }
      }
      if let DraggingVert::PolygonCycle(i, j) = self.dragging_vert {
        // Dragging
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        self.drag_offset = diff(pt_pos.into(), poly.cycles[i][j]);
      } else if let DraggingVert::CurCycle(j) = self.dragging_vert {
        self.dragging_vert = DraggingVert::CurCycle(j);
        self.drag_offset = diff(pt_pos.into(), self.cur_cycle.as_ref().unwrap()[j]);
      } else if self.sel_polygon.is_some() {
        // Adding new point
        let cyc = self.cur_cycle.get_or_insert(vec![]);
        cyc.push(pt_pos.into());
        self.dragging_vert = DraggingVert::CurCycle(cyc.len() - 1);
        self.drag_offset = (0.0, 0.0);
        if self.guide == 0 && cyc.len() >= 3 { self.guide = 1; }
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
          if self.guide == 1 {
            self.guide = 2;
          } else if self.guide == 2 &&
              self.polygons[self.sel_polygon.unwrap()].cycles.len() >= 2 {
            self.guide = 3;
          }
        }
      } else if self.sel_polygon.is_some() {
        let poly = &mut self.polygons[self.sel_polygon.unwrap()];
        if let Some((i, j)) = find_vertex(&mut poly.cycles) {
          // Remove vertex
          poly.cycles[i].remove(j);
          if poly.cycles[i].is_empty() {
            poly.cycles.remove(i);
          }
        }
      }
    }

    if ((pt1_held || pt2_held) && pt_moved) ||
       (pt1_press || pt2_press || pt1_rel || pt2_rel) {
      self.canvas_shapes_obsolete = true;
    }
  }
}

impl epi::App for App {
  fn name(&self) -> &str {
    "Polygon Intersection"
  }

  fn setup(
    &mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>,
    _storage: Option<&dyn epi::Storage>,
  ) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
      "qisixihei".to_owned(),
      std::borrow::Cow::Borrowed(include_bytes!("../fonts/1570788235-subset.ttf")));
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
    if let Some(storage) = _storage {
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
        ui.memory().is_any_popup_open(),
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
                egui::ScrollArea::vertical().show(ui, |ui| {
                  ui.horizontal(|ui| {
                    ui.add_sized(
                      egui::vec2(160.0, ui.style().spacing.interact_size.y),
                      egui::Label::new("Intersection"),
                    );
                    ui.color_edit_button_rgba_unmultiplied(&mut self.intersection_khroma);
                    let any_visible = self.polygons_visible.iter().any(|x| *x);
                    if ui.selectable_label(any_visible, "○")
                         .on_hover_text("Visibility").clicked() {
                      for value in self.polygons_visible.iter_mut() {
                        *value = !any_visible;
                      }
                      self.canvas_shapes_obsolete = true;
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
                          format!("#{} ({} vert, {} ring)",
                            index + 1,
                            poly.cycles.iter().map(|c| c.len()).sum::<usize>(),
                            poly.cycles.len(),
                          ))
                      ).clicked() {
                        self.sel_polygon = if sel { None } else { Some(index) };
                        self.cur_cycle = None;
                        self.dragging_vert = DraggingVert::None;
                        self.canvas_shapes_obsolete = true;
                      }
                      // Recalculate selection state to avoid UI jumping
                      let sel = match self.sel_polygon {
                        Some(i) => i == index,
                        _ => false,
                      };
                      ui.color_edit_button_rgb(&mut poly.khroma);
                      if ui.selectable_label(self.polygons_visible[index], "○")
                           .on_hover_text("Visibility").clicked() {
                        self.polygons_visible[index] = !self.polygons_visible[index];
                        self.canvas_shapes_obsolete = true;
                      }
                      if sel {
                        if ui.selectable_label(false, "×")
                             .on_hover_text("Remove").clicked() {
                          polygon_remove = Some(index);
                          self.canvas_shapes_obsolete = true;
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
                if self.guide == 3 { self.guide = 4; }
              } else {
                self.last_added_polygon = false;
              }
            }).body_response.is_none() {
              if !self.polygons_collapsed {
                self.polygons_collapsed = true;
                self.canvas_shapes_obsolete = true;
              }
            } else {
              if self.polygons_collapsed {
                self.polygons_collapsed = false;
                self.canvas_shapes_obsolete = true;
              }
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
