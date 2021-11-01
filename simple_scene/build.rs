#![feature(iter_intersperse)]

use gl_generator::{Registry, Api, Profile, Fallbacks, GlobalGenerator};
use std::env;
use std::io::Write;
use std::fs::File;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("cargo:rerun-if-changed=build.rs");

  let mut buf = std::io::BufWriter::new(Vec::new());
  Registry::new(Api::Gl, (3, 3), Profile::Core, Fallbacks::All, [])
    .write_bindings(GlobalGenerator, &mut buf)?;

  let s = String::from_utf8(buf.into_inner()?)?;

  let s_written = s.replacen(
    "mut loadfn: &mut FnMut(&str)",
    "loadfn: &mut dyn FnMut(&str)",
    1,
  ).replacen(
    "mod __gl_imports",
    "pub mod __gl_imports",
    1,
  ).replace(
    "pub unsafe fn",
    "pub(super) unsafe fn",
  );

  let dest = env::var("OUT_DIR")?;
  let mut file = File::create(&Path::new(&dest).join("bindings.rs"))?;
  file.write_all(s_written.as_bytes())?;

  // Pretend things are safe
  let mut file = File::create(&Path::new(&dest).join("safe_wrapper.rs"))?;
  let prefix = "pub unsafe fn ";
  for (index, _) in s.match_indices(prefix) {
    let start = index + prefix.len();
    let left_bracket = start + s[start..].find('(').unwrap();
    let right_bracket = left_bracket + s[left_bracket..].find(')').unwrap();
    let left_brace = right_bracket + s[right_bracket..].find(" {").unwrap();
    file.write_all(format!(
r"
#[allow(non_snake_case, dead_code)]
pub fn {} {{
  unsafe {{ generated::{}({}) }}
}}
",
      s[start..left_brace].replace("types::", "generated::types::"),
      &s[start..left_bracket],
      s[left_bracket + 1..right_bracket]
        .split(", ").map(|arg| arg.split(':').next().unwrap())
        .intersperse(", ").collect::<String>(),
    ).as_bytes())?;
  }

  Ok(())
}
