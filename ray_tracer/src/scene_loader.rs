use crate::gl;
use wavefront_obj::obj;

pub struct Frame {
  pub vertices: Vec<[f32; 7]>,
}

// (x_min, x_max, y_min, y_max, z_min, z_max)
fn aabb_overlap(
  a: (f32, f32, f32, f32, f32, f32),
  b: (f32, f32, f32, f32, f32, f32),
) -> bool {
  let eps = 1e-6;
  a.0.max(b.0) < a.1.min(b.1) + eps &&
  a.2.max(b.2) < a.3.min(b.3) + eps &&
  a.4.max(b.4) < a.5.min(b.5) + eps
}

pub fn load<P: AsRef<std::path::Path>>(p: P)
-> Result<Frame, Box<dyn std::error::Error>> {
  let contents = std::fs::read_to_string(p)?;
  let objects = obj::parse(contents)?.objects;

  let mut vertices = vec![];
  let mut aabb: Vec<((f32, f32, f32, f32, f32, f32), (usize, usize))> = vec![];

  for object in &objects {
    let vertices_start = vertices.len();
    let (mut x_min, mut x_max) = (f32::INFINITY, -f32::INFINITY);
    let (mut y_min, mut y_max) = (f32::INFINITY, -f32::INFINITY);
    let (mut z_min, mut z_max) = (f32::INFINITY, -f32::INFINITY);
    for geom in &object.geometry {
      for shape in &geom.shapes {
        if let obj::Primitive::Triangle(
          (vi0, _, Some(ni0)),
          (vi1, _, Some(ni1)),
          (vi2, _, Some(ni2)),
        ) = shape.primitive {
          for (vi, ni) in [
            (vi0, ni0),
            (vi1, ni1),
            (vi2, ni2),
          ] {
            let v = object.vertices[vi];
            let n = object.normals[ni];
            vertices.push([
              v.x as f32, v.y as f32, v.z as f32,
              n.x as f32, n.y as f32, n.z as f32,
              0.0,
            ]);
            x_min = x_min.min(v.x as f32);
            x_max = x_max.max(v.x as f32);
            y_min = y_min.min(v.y as f32);
            y_max = y_max.max(v.y as f32);
            z_min = z_min.min(v.z as f32);
            z_max = z_max.max(v.z as f32);
          }
        }
      }
    }
    let bb = (x_min, x_max, y_min, y_max, z_min, z_max);
    for &(bb_other, (start, end)) in &aabb {
      if aabb_overlap(bb, bb_other) {
        // Mark vertices in the other object as overlapping
        for i in start..end {
          vertices[i][6] = 1.0;
        }
        for i in vertices_start..vertices.len() {
          vertices[i][6] = 1.0;
        }
      }
    }
    aabb.push((bb, (vertices_start, vertices.len())));
  }

  Ok(Frame {
    vertices,
  })
}
