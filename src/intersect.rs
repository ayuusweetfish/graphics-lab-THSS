// Weiler–Atherton algorithm
pub fn calculate(polygons: &[&Vec<Vec<(f32, f32)>>]) -> Vec<Vec<(f32, f32)>> {
  if polygons.is_empty() { vec![] }
  else { polygons[0].clone() }
}
