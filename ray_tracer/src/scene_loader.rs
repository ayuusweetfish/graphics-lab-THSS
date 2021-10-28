use wavefront_obj::{obj, mtl};
use relative_path::RelativePath;
use image::GenericImageView;

#[repr(C)]
pub struct Vertex {
  pub pos: (f32, f32, f32),
  pub norm: (f32, f32, f32),
  pub texc: (f32, f32),
  pub texid: u8,  // 255 denotes no texture

  pub mirror: bool,
  pub refr: f32,
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
            let mut v = Vertex {
              pos:  (v.x as f32, v.y as f32, v.z as f32),
              norm: (n.x as f32, n.y as f32, n.z as f32),
              texc: (t.u as f32, 1.0 - t.v as f32),
              texid: texid as u8,
              mirror: object.name.contains("Mirror"),
              refr: if object.name.contains("Glass") { 1.4 } else { 0.0 },
            };
            if texid == 255 {
              // Assign colours to models
              if object.name.contains("Crown_Cone") {
                v.texc = (0.4, 0.7);
              } else if object.name.contains("Crown_Sphere") {
                v.texc = (0.4, 0.85);
              } else if object.name.contains("Trunk") {
                v.texc = (0.5, 0.35);
              } else if object.name.contains("Glass") {
                v.texc = (-1.0, 0.9);
              } else if object.name.contains("tea") {
                v.texc = (0.6, 0.3);
              } else {
                v.texc = (0.0, 0.0);
              }
            }
            vertices.push(v);
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
