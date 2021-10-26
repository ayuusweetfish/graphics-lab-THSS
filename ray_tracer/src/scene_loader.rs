use wavefront_obj::{obj, mtl};
use relative_path::RelativePath;
use image::GenericImageView;

#[repr(C)]
pub struct Vertex {
  pos: (f32, f32, f32),
  norm: (f32, f32, f32),
  texc: (f32, f32),
  texid: u8,  // 255 denotes no texture
}

pub struct Frame {
  pub vertices: Vec<Vertex>,
  pub textures: Vec<(u32, u32, Vec<u8>)>,
}

pub fn load<P: AsRef<std::path::Path>>(p: P)
-> Result<Frame, Box<dyn std::error::Error>> {
  let contents = std::fs::read_to_string(&p)?;
  let obj = obj::parse(contents)?;
  println!("Loaded object set {}", p.as_ref().to_str().unwrap_or(""));

  let mtl_path = obj.material_library.ok_or("no MTL specified")?;
  let mtl_path = RelativePath::from_path(&p)?
    .parent().ok_or("empty path")?.join(RelativePath::new(&mtl_path));
  let contents = std::fs::read_to_string(mtl_path.to_path("."))?;
  let mtl = mtl::parse(contents)?.materials;
  println!("Loaded material library {}", mtl_path.to_path(".").to_str().unwrap_or(""));

  // Consume material records and insert into a lookup table
  // HashMap<name: String, (index: usize, texid: usize)>
  let mut mtl_lookup = std::collections::HashMap::new();

  // Load textures
  let mut textures = vec![];
  for (i, mat) in mtl.iter().enumerate() {
    let mut texid = 255;

    if let Some(tex_path) = &mat.diffuse_map {
      let tex_path = mtl_path
        .parent().ok_or("empty path")?.join(RelativePath::new(&tex_path));
      let img = image::io::Reader::open(tex_path.to_path("."))?.decode()?;
      let (w, h) = img.dimensions();
      let buf = img.into_rgb8().into_raw();
      println!("Loaded texture {}", tex_path.to_path(".").to_str().unwrap_or(""));
      texid = textures.len();
      textures.push((w, h, buf));
    } else {
    }

    mtl_lookup.insert(mat.name.clone(), (i, texid));
  }

  // Read geometry
  let objects = obj.objects;
  let mut vertices = vec![];
  for object in &objects {
    for geom in &object.geometry {
      let (mat_idx, texid) = *mtl_lookup.get(
        geom.material_name.as_ref().ok_or("no material")?
      ).ok_or("unknown material")?;
      let mat = &mtl[mat_idx];
      for shape in &geom.shapes {
        if let obj::Primitive::Triangle(
          (vi0, Some(ti0), Some(ni0)),
          (vi1, Some(ti1), Some(ni1)),
          (vi2, Some(ti2), Some(ni2)),
        ) = shape.primitive {
          for (vi, ti, ni) in [
            (vi0, ti0, ni0),
            (vi1, ti1, ni1),
            (vi2, ti2, ni2),
          ] {
            let v = object.vertices[vi];
            let t = object.tex_vertices[ti];
            let n = object.normals[ni];
            vertices.push(Vertex {
              pos:  (v.x as f32, v.y as f32, v.z as f32),
              norm: (n.x as f32, n.y as f32, n.z as f32),
              texc: (t.u as f32, 1.0 - t.v as f32),
              texid: texid as u8,
            });
          }
        }
      }
    }
  }

  Ok(Frame {
    vertices,
    textures,
  })
}
