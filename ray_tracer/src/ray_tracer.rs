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

        //if y == self.h / 2 - 1 && (x as i32 - self.w as i32 / 2).abs() <= 6 {
        /*if y == 39 && x == 224 {
          self.debug = true;
        }*/
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
        Ok(dur) if dur >= std::time::Duration::from_millis(10) => break,
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
        /*if self.ibuf[i * 4] <= 20 {
          self.ibuf[i * 4] = 255;
          println!("{} {}", y, x);
        }*/
      }
    }
    if self.sample_count == 0 {
      for (x, y) in &self.sample_order[self.sample_order_pos..] {
        let i = (*y * self.w + x) as usize;
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
      println!("{:?} - {:?}", cen.as_array(), dir.as_array());
    }
    if let Some((tri_idx, intxn)) = self.collide(cen, dir) {
      let norm = tuple_vec3(self.frame.vertices[tri_idx * 3].norm);
      // The normal may need to be flipped
      let norm = if glm::dot(dir, norm) < 0.0 { norm } else { -norm };
      let refl = lambertian(norm);
      if self.debug {
        println!("  tri = \n    {:?}\n    {:?}\n    {:?}\n  intxn = {:?}",
          self.frame.vertices[tri_idx * 3].pos,
          self.frame.vertices[tri_idx * 3 + 1].pos,
          self.frame.vertices[tri_idx * 3 + 2].pos,
          intxn);
      }
      self.ray_colour(intxn, refl, w * 0.5)
    } else {
      glm::vec3(0.3, 0.3, 0.2) * w
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
