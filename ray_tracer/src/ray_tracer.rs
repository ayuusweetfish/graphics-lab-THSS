use crate::scene_loader;

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

  cbuf: Vec<[f32; 3]>,
  sample_count: u32,

  ibuf: Vec<u8>,
}

impl<'a> RayTracer<'a> {
  pub fn new(
    w: u32,
    h: u32,
    frame: &'a scene_loader::Frame,
  ) -> Self {
    Self {
      w, h, frame,

      cam_pos: glm::vec3(0.0, 0.0, 0.0),
      cam_ori: glm::vec3(0.0, 0.0, -1.0),
      cam_up: glm::vec3(0.0, 1.0, 0.0),
      vfov: 0.5236,

      cam_corner: glm::vec3(0.0, 0.0, 0.0),
      cam_horz_span: glm::vec3(0.0, 0.0, 0.0),
      cam_vert_span: glm::vec3(0.0, 0.0, 0.0),

      cbuf: vec![[0.0; 3]; (w * h) as usize],
      sample_count: 0,
      ibuf: vec![0; (w * h * 3) as usize],
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
  }

  pub fn render(&mut self) {
    for y in 0..self.h {
      for x in 0..self.w {
        // if y != self.h / 2 || x != self.w / 2 { continue; }
        // Cast a ray
        let rx = (x as f32 + 0.5) / self.w as f32;
        let ry = (y as f32 + 0.5) / self.h as f32;
        let ray_ori = self.cam_corner
          + self.cam_horz_span * rx
          + self.cam_vert_span * ry
          - self.cam_pos;
        let ray_ori = glm::normalize(ray_ori);

        let k = self.ray_colour(self.cam_pos, ray_ori, 1.0);

        let i = (y * self.w + x) as usize;
        self.cbuf[i][0] += k[0];
        self.cbuf[i][1] += k[1];
        self.cbuf[i][2] += k[2];
      }
    }

    self.sample_count += 1;
    println!("{}", self.sample_count);
  }

  pub fn image(&mut self) -> *const u8 {
    let scale = 255.99 / (self.sample_count as f32);
    for y in 0..self.h {
      for x in 0..self.w {
        let i = (y * self.w + x) as usize;
        for ch in 0..3 {
          self.ibuf[i * 3 + ch] =
            (self.cbuf[i][ch] as f32 * scale) as u8;
        }
      }
    }
    self.ibuf.as_ptr()
  }

  fn ray_colour(
    &self,
    cen: glm::Vec3, dir: glm::Vec3,
    w: f32,
  ) -> glm::Vec3 {
    // println!("{:?} - {:?}", cen, dir);
    let tri_index = self.collide(cen, dir);
    if let Some(_) = tri_index {
      glm::vec3(0.9, 1.0, 0.9)
    } else {
      glm::vec3(0.3, 0.3, 0.2)
    }
  }

  // Finds the nearest colliding triangle
  // Returns the index
  fn collide(&self, cen: glm::Vec3, dir: glm::Vec3) -> Option<usize> {
    for (i, tri) in self.frame.vertices.chunks_exact(3).enumerate() {
      if let Some(p) = ray_tri_intersect(
        cen, dir,
        tuple_vec3(tri[0].pos),
        tuple_vec3(tri[1].pos),
        tuple_vec3(tri[2].pos),
      ) {
        return Some(i);
      }
    }
    None
  }
}

fn tuple_vec3(p: (f32, f32, f32)) -> glm::Vec3 { glm::vec3(p.0, p.1, p.2) }

// Möller–Trumbore
fn ray_tri_intersect(
  cen: glm::Vec3, dir: glm::Vec3,
  p0: glm::Vec3, p1: glm::Vec3, p2: glm::Vec3,
) -> Option<glm::Vec3> {
  let eps = 1e-7;
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
    Some(cen + dir * t)
  } else {
    None
  }
}
