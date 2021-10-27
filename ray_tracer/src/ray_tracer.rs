use crate::scene_loader;

pub struct RayTracer<'a> {
  w: u32,
  h: u32,
  frame: &'a scene_loader::Frame,

  cbuf: Vec<[f32; 3]>,
  sample_count: u32,

  ibuf: Vec<u8>,
}

impl<'a> RayTracer<'a> {
  pub fn new(w: u32, h: u32, frame: &'a scene_loader::Frame) -> Self {
    Self {
      w, h, frame,
      cbuf: vec![[0.0; 3]; (w * h) as usize],
      sample_count: 0,
      ibuf: vec![0; (w * h * 3) as usize],
    }
  }

  pub fn reset(&mut self) {
    self.cbuf.iter_mut().map(|x| *x = [0.0; 3]).for_each(drop);
    self.sample_count = 0;
  }

  pub fn render(&mut self) {
    for y in 0..self.h {
      for x in 0..self.w {
        let i = (y * self.w + x) as usize;
        self.cbuf[i][0] += 0.2;
        self.cbuf[i][1] += 0.3;
        self.cbuf[i][2] += 0.2;
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
}
