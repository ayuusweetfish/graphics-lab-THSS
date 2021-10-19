use crate::gl;
use wavefront_obj::obj;

pub struct Frame {
  pub vertices: Vec<[f32; 3]>,
}

pub fn load<P: AsRef<std::path::Path>>(p: P)
-> Result<Frame, Box<dyn std::error::Error>> {
  let contents = std::fs::read_to_string(p)?;
  let objects = obj::parse(contents)?.objects;

  let mut vertices = vec![];

  for object in objects {
    println!("{}", object.name);
    for geom in object.geometry {
      for shape in geom.shapes {
        if let obj::Primitive::Triangle(
          (vi0, _, _),
          (vi1, _, _),
          (vi2, _, _),
        ) = shape.primitive {
          for vi in [vi0, vi1, vi2] {
            let v = object.vertices[vi];
            vertices.push([v.x as f32 / 10.0, v.y as f32 / 10.0, v.z as f32 / 10.0]);
          }
        }
      }
    }
  }

  Ok(Frame {
    vertices,
  })
}
