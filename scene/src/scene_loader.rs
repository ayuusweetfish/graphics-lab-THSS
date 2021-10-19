use crate::gl;
use wavefront_obj::obj;

pub struct Frame {
  pub vertices: Vec<[f32; 6]>,
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
            ]);
          }
        }
      }
    }
  }

  Ok(Frame {
    vertices,
  })
}
