use crate::scene_loader;
use rand::{Rng, seq::SliceRandom};

pub struct RayTracer<'a> {
  w: u32,
  h: u32,
  frame: &'a scene_loader::Frame,

  cam_pos: glm::Vec3,
  cam_ori: glm::Vec3,
  cam_up: glm::Vec3,
  vfov: f32,

  cam_corner: glm::Vec3,
  cam_horz_span: glm::Vec3,
  cam_vert_span: glm::Vec3,

  tris: Vec<TriangleIndex>,
  bvh: bvh::bvh::BVH,

  sample_order: Vec<(u32, u32)>,
  sample_order_pos: usize,

  cbuf: Vec<[f32; 3]>,
  sample_count: u32,

  ibuf: Vec<u8>,

  debug: bool,
}

struct TriangleIndex {
  index: usize,
  aabb: bvh::aabb::AABB,
  node_index: usize,
}

fn minmax(a: f32, b: f32, c: f32) -> (f32, f32) {
  let (mut min, mut max) = (a, a);
  if b < min { min = b; } else if b > max { max = b; }
  if c < min { min = c; } else if c > max { max = c; }
  (min, max)
}

impl TriangleIndex {
  fn new(frame: &scene_loader::Frame, index: usize) -> Self {
    let (min_x, max_x) = minmax(
      frame.vertices[index * 3].pos.0,
      frame.vertices[index * 3 + 1].pos.0,
      frame.vertices[index * 3 + 2].pos.0);
    let (min_y, max_y) = minmax(
      frame.vertices[index * 3].pos.1,
      frame.vertices[index * 3 + 1].pos.1,
      frame.vertices[index * 3 + 2].pos.1);
    let (min_z, max_z) = minmax(
      frame.vertices[index * 3].pos.2,
      frame.vertices[index * 3 + 1].pos.2,
      frame.vertices[index * 3 + 2].pos.2);
    let aabb = bvh::aabb::AABB::with_bounds(
      (min_x, min_y, min_z).into(),
      (max_x, max_y, max_z).into(),
    );
    Self { aabb, index, node_index: 0 }
  }
}

impl bvh::aabb::Bounded for TriangleIndex {
  fn aabb(&self) -> bvh::aabb::AABB { self.aabb }
}

impl bvh::bounding_hierarchy::BHShape for TriangleIndex {
  fn set_bh_node_index(&mut self, index: usize) { self.node_index = index; }
  fn bh_node_index(&self) -> usize { self.node_index }
}

impl<'a> RayTracer<'a> {
  pub fn new(
    w: u32,
    h: u32,
    frame: &'a scene_loader::Frame,
  ) -> Self {
    let mut tris = (0..frame.vertices.len() / 3)
      .map(|i| TriangleIndex::new(frame, i)).collect::<Vec<_>>();
    let bvh = bvh::bvh::BVH::build(&mut tris);

    Self {
      w, h, frame,

      cam_pos: glm::vec3(0.0, 0.0, 0.0),
      cam_ori: glm::vec3(0.0, 0.0, -1.0),
      cam_up: glm::vec3(0.0, 1.0, 0.0),
      vfov: 0.5236,

      cam_corner: glm::vec3(0.0, 0.0, 0.0),
      cam_horz_span: glm::vec3(0.0, 0.0, 0.0),
      cam_vert_span: glm::vec3(0.0, 0.0, 0.0),

      tris,
      bvh,

      sample_order: (0..w).into_iter()
        .flat_map(|x| (0..h).into_iter().map(move |y| (x, y)))
        .collect(),
      sample_order_pos: 0,

      cbuf: vec![[0.0; 3]; (w * h) as usize],
      sample_count: 0,
      ibuf: vec![0; (w * h * 4) as usize],

      debug: false,
    }
  }

  pub fn reset(&mut self,
    cam_pos: glm::Vec3,
    cam_ori: glm::Vec3,
    cam_up: glm::Vec3,
    vfov: f32,
    foc_len: f32,
  ) {
    self.cam_pos = cam_pos;
    self.cam_ori = cam_ori;
    self.cam_up = cam_up;
    self.vfov = vfov;

    let viewport_h = (vfov / 2.0).tan() * 2.0;
    let viewport_w = (self.w as f32 / self.h as f32) * viewport_h;
    let w = glm::normalize(-self.cam_ori);
    let u = glm::normalize(glm::cross(cam_up, w));
    let v = glm::cross(w, u);
    self.cam_horz_span = u * foc_len * viewport_w;
    self.cam_vert_span = v * foc_len * viewport_h;
    self.cam_corner = self.cam_pos
      - self.cam_horz_span / 2.0
      - self.cam_vert_span / 2.0
      - w * foc_len;

    self.cbuf.iter_mut().map(|x| *x = [0.0; 3]).for_each(drop);
    self.sample_count = 0;

    self.sample_order.shuffle(&mut rand::thread_rng());
    self.sample_order_pos = 0;
  }

  pub fn frame_filled(&self) -> bool { self.sample_count > 0 }

  pub fn render(&mut self) {
    let start = std::time::SystemTime::now();
    loop {
      for _rept in 0..200 {
        let (x, y) = self.sample_order[self.sample_order_pos];
        // if y != self.h / 2 || x != self.w / 2 { continue; }
        // Cast a ray
        let rx = (x as f32 + rand::random::<f32>()) / self.w as f32;
        let ry = (y as f32 + rand::random::<f32>()) / self.h as f32;
        let ray_ori = self.cam_corner
          + self.cam_horz_span * rx
          + self.cam_vert_span * ry
          - self.cam_pos;
        let ray_ori = glm::normalize(ray_ori);

        self.debug = x == self.w / 2 && y == self.h / 2;
        let k = self.ray_colour(self.cam_pos, ray_ori, 1.0);

        let i = (y * self.w + x) as usize;
        let s = self.sample_count as f32;
        self.cbuf[i][0] = (self.cbuf[i][0] * s + k[0]) / (s + 1.0);
        self.cbuf[i][1] = (self.cbuf[i][1] * s + k[1]) / (s + 1.0);
        self.cbuf[i][2] = (self.cbuf[i][2] * s + k[2]) / (s + 1.0);
        if self.debug {
          println!("{} {} {:?} {:?}", x, y, ray_ori, k[0]);
          self.debug = false;
        }

        self.sample_order_pos += 1;
        if self.sample_order_pos as u32 == self.w * self.h {
          self.sample_order.shuffle(&mut rand::thread_rng());
          self.sample_order_pos = 0;
          self.sample_count += 1;
        }
      }

      match start.elapsed() {
        Ok(dur) if dur >= std::time::Duration::from_millis(40) => break,
        Err(_) => break,
        _ => continue,
      }
    }

    println!("{}", self.sample_count);
  }

  pub fn image(&mut self) -> *const u8 {
    for y in 0..self.h {
      for x in 0..self.w {
        let i = (y * self.w + x) as usize;
        for ch in 0..3 {
          self.ibuf[i * 4 + ch] =
            (self.cbuf[i][ch].powf(1.0/2.2) * 255.99) as u8;
        }
        self.ibuf[i * 4 + 3] = 255;
      }
    }
    if self.sample_count == 0 {
      for (x, y) in &self.sample_order[self.sample_order_pos..] {
        let i = (*y * self.w + x) as usize;
        self.ibuf[i * 4 + 0] = 0;
        self.ibuf[i * 4 + 1] = 0;
        self.ibuf[i * 4 + 2] = 0;
        self.ibuf[i * 4 + 3] = 0;
      }
    }
    self.ibuf.as_ptr()
  }

  fn ray_colour(
    &self,
    cen: glm::Vec3, dir: glm::Vec3,
    w: f32,
  ) -> glm::Vec3 {
    if w <= 1e-4 {
      return glm::vec3(0.0, 0.0, 0.0);
    }
    if self.debug {
      println!("{:?} - {:?}", cen.as_array(), glm::normalize(dir).as_array());
    }
    if let Some((tri_idx, intxn)) = self.collide(cen, dir) {
      let v = &self.frame.vertices[tri_idx * 3];
      let norm = tuple_vec3(v.norm);
      let side_in = glm::dot(dir, norm) < 0.0;
      // The normal may need to be flipped
      let norm = if side_in { norm } else { -norm };
      // Reflect or refract?
      let refr_index = v.refr;
      let refr_index = if side_in { 1.0 / refr_index } else { refr_index };
      let refract = if refr_index != 0.0 {
        let cos_theta = glm::dot(-dir, norm).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let no_refr = refr_index * sin_theta > 1.0;
        if self.debug {
          println!("incident angle: {}\nrefr index: {}", sin_theta, refr_index);
        }
        !no_refr && reflectance(cos_theta, refr_index) <= rand::random::<f32>()
      } else {
        false
      };
      // Scattered direction
      let refl = if refract {
        // Refrcation
        glm::refract(
          glm::normalize(dir),
          glm::normalize(norm),
          refr_index)
      } else if self.frame.vertices[tri_idx * 3].mirror {
        // Specular reflection
        glm::reflect(dir, norm)
      } else {
        // Lambertian refection
        lambertian(norm)
      };
      if self.debug {
        println!("  tri = \n    {:?}\n    {:?}\n    {:?}\n  intxn = {:?}",
          self.frame.vertices[tri_idx * 3].pos,
          self.frame.vertices[tri_idx * 3 + 1].pos,
          self.frame.vertices[tri_idx * 3 + 2].pos,
          intxn);
      }
      let mut rate = 0.5;
      let albedo =
        if self.frame.vertices[tri_idx * 3].texid != 255 {
          // glm::vec3(0.8, 0.5, 0.4)
          tex_sample(
            &self.frame.textures[v.texid as usize],
            tuple_vec3(self.frame.vertices[tri_idx * 3 + 0].pos),
            tuple_vec3(self.frame.vertices[tri_idx * 3 + 1].pos),
            tuple_vec3(self.frame.vertices[tri_idx * 3 + 2].pos),
            tuple_vec2(self.frame.vertices[tri_idx * 3 + 0].texc),
            tuple_vec2(self.frame.vertices[tri_idx * 3 + 1].texc),
            tuple_vec2(self.frame.vertices[tri_idx * 3 + 2].texc),
            intxn)
        } else {
          if v.texc == (0.0, 0.0) {
            // Mirror
            if v.mirror { rate = 0.8; }
            // White stuff
            glm::vec3(1.0, 1.0, 1.0)
          } else if v.texc.0 < 0.0 {
            // Glass
            rate = 0.9;
            glm::vec3(v.texc.1, 1.0 - (1.0 - v.texc.1) / 2.0, 1.0)
          } else {
            // Coloured
            glm::vec3(v.texc.0, v.texc.1, (v.texc.0 + v.texc.1) * 0.3)
          }
        };
      albedo * self.ray_colour(intxn, refl, w * rate)
    } else {
      let angle = glm::normalize(dir).y * 0.5 + 0.5;
      glm::vec3(0.6, 0.6 + angle * 0.1, 0.6 + angle * 0.2) * w
    }
  }

  // Finds the nearest colliding triangle
  // Returns the index and point of intersection
  fn collide(&self, cen: glm::Vec3, dir: glm::Vec3) -> Option<(usize, glm::Vec3)> {
    let ray = bvh::ray::Ray::new(
      (cen[0], cen[1], cen[2]).into(),
      (dir[0], dir[1], dir[2]).into());
    let mut best = (f32::INFINITY, None);
    for tri in self.bvh.traverse_iterator(&ray, &self.tris) {
      if let Some((p, t)) = ray_tri_intersect(
        cen, dir,
        tuple_vec3(self.frame.vertices[tri.index * 3 + 0].pos),
        tuple_vec3(self.frame.vertices[tri.index * 3 + 1].pos),
        tuple_vec3(self.frame.vertices[tri.index * 3 + 2].pos),
      ) {
        if t < best.0 {
          best = (t, Some((tri.index, p)));
        }
      }
    }
    best.1
  }
}

fn tuple_vec2(p: (f32, f32)) -> glm::Vec2 { glm::vec2(p.0, p.1) }
fn tuple_vec3(p: (f32, f32, f32)) -> glm::Vec3 { glm::vec3(p.0, p.1, p.2) }

// Möller–Trumbore
fn ray_tri_intersect(
  cen: glm::Vec3, dir: glm::Vec3,
  p0: glm::Vec3, p1: glm::Vec3, p2: glm::Vec3,
) -> Option<(glm::Vec3, f32)> {
  let eps = 1e-5;
  let e1 = p1 - p0;
  let e2 = p2 - p0;
  let h = glm::cross(dir, e2);
  let a = glm::dot(e1, h);
  if a > -eps && a < -eps { return None; }
  let f = 1.0 / a;
  let s = cen - p0;
  let u = f * glm::dot(s, h);
  if u < 0.0 || u > 1.0 { return None; }
  let q = glm::cross(s, e1);
  let v = f * glm::dot(dir, q);
  if v < 0.0 || u + v > 1.0 { return None; }
  let t = f * glm::dot(e2, q);
  if t > eps {
    Some((cen + dir * t, t))
  } else {
    None
  }
}

fn lambertian(n: glm::Vec3) -> glm::Vec3 {
  let sin_theta = rand::random::<f32>().sqrt();
  let cos_theta = (1.0 - sin_theta * sin_theta).sqrt();
  let psi = rand::random::<f32>() * std::f32::consts::TAU;
  let x = sin_theta * psi.cos();
  let y = sin_theta * psi.sin();
  let z = cos_theta;
  // (x, y, z) is sampled from a Lambertian distribution
  // for the normal (0, 0, 1)
  let p = if n.x.abs() <= 1e-6 && n.y.abs() <= 1e-6 {
    glm::normalize(glm::vec3(-n.z, 0.0, n.x))
  } else {
    glm::normalize(glm::vec3(-n.y, n.x, 0.0))
  };
  let q = glm::cross(n, p);
  p * x + q * y + n * z
}

fn tex_sample(
  tex: &(u32, u32, Vec<u8>),
  p0: glm::Vec3, p1: glm::Vec3, p2: glm::Vec3,
  t0: glm::Vec2, t1: glm::Vec2, t2: glm::Vec2,
  q: glm::Vec3,
) -> glm::Vec3 {
  let i0 = q - p0;
  let i1 = q - p1;
  let i2 = q - p2;
  let a0 = glm::ext::sqlength(glm::cross(i1, i2));
  let a1 = glm::ext::sqlength(glm::cross(i2, i0));
  let a2 = glm::ext::sqlength(glm::cross(i0, i1));
  let texc = (t0 * a0 + t1 * a1 + t2 * a2) / (a0 + a1 + a2);
  let texc = texc * glm::vec2(tex.0 as f32, tex.1 as f32);

  let buf = &tex.2;

  // Nearest filter
  let x = (texc.x as u32).min(tex.0 - 1);
  let y = (texc.y as u32).min(tex.1 - 1);
  let k = glm::vec3(
    buf[(y * tex.0 + x) as usize * 3 + 0] as f32,
    buf[(y * tex.0 + x) as usize * 3 + 1] as f32,
    buf[(y * tex.0 + x) as usize * 3 + 2] as f32,
  ) / 255.0;
  k
}

// Schlick's approximation
fn reflectance(cosine: f32, refr_index: f32) -> f32 {
  let r0 = ((1.0 - refr_index) / (1.0 + refr_index)).powi(2);
  r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
