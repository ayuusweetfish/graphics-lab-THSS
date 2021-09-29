pub fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
  ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1))
}
pub fn dist(a: (f32, f32), b: (f32, f32)) -> f32 { dist_sq(a, b).sqrt() }
pub fn diff(a: (f32, f32), b: (f32, f32)) -> (f32, f32) { (a.0 - b.0, a.1 - b.1) }
pub fn det(a: (f32, f32), b: (f32, f32)) -> f32 { a.0 * b.1 - a.1 * b.0 }
pub fn det3(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 { det(diff(b, a), diff(c, a)) }
pub fn dot(a: (f32, f32), b: (f32, f32)) -> f32 { a.0 * b.0 + a.1 * b.1 }
pub fn lerp(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
  (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}
pub fn project(a: (f32, f32), p: (f32, f32), q: (f32, f32)) -> (f32, f32) {
  let l_sq = dist_sq(p, q);
  let t = dot(diff(a, p), diff(q, p)) / l_sq;
  lerp(p, q, t)
}
pub fn dist_to_seg(a: (f32, f32), p: (f32, f32), q: (f32, f32)) -> f32 {
  let l_sq = dist_sq(p, q);
  if l_sq == 0.0 { return dist(a, p); }
  let t = dot(diff(a, p), diff(q, p)) / l_sq;
  let t = match t {
    f32::MIN..=0.0 => 0.0,
    1.0..=f32::MAX => 1.0,
    _ => t,
  };
  dist(a, lerp(p, q, t))
}

pub fn point_in_simple_polygon(p: (f32, f32), a: &[(f32, f32)]) -> bool {
  // Winding number
  // Optimization by W. Randolph Franklin
  // https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
  let (x0, y0) = p;
  let mut c = false;
  for i in 0..a.len() {
    let (x1, y1) = a[i];
    let (x2, y2) = a[(i + 1) % a.len()];
    if ((y2 > y0) != (y1 > y0)) &&
       x0 < (x1 - x2) * (y0 - y2) / (y1 - y2) + x2 {
      c = !c;
    }
  }
  c
}

// Direction is in the right-handed coordinate system
fn normalize_dir(cycle: &[(f32, f32)], ccw: bool) -> Vec<(f32, f32)> {
  if (det3(cycle[0], cycle[1], cycle[2]) > 0.0) != ccw {
    cycle.iter().map(|&x| x).rev().collect()
  } else {
    cycle.iter().map(|&x| x).collect()
  }
}

// Split polygons into disjoint components,
// each with an outer boundary (coming first) and zero or more inner holes
// Outer boundaries are counter-clockwise; inner boundaries are clockwise
pub fn normalize_polygon(cycles: &[Vec<(f32, f32)>])
  -> Vec<Vec<Vec<(f32, f32)>>>
{
  let n = cycles.len();
  let mut contain_count = vec![0; n];
  let mut outer = vec![vec![]; n];
  for i in 0..n {
    for j in 0..n {
      if i != j && point_in_simple_polygon(cycles[i][0], &cycles[j]) {
        contain_count[j] += 1;
        outer[i].push(j);
      }
    }
  }
  let mut comps = vec![];
  let mut comp_index = vec![0; n];
  for i in 0..n {
    if outer[i].len() % 2 == 0 {
      // Outer boundary
      comps.push(vec![normalize_dir(&cycles[i], true)]);
      comp_index[i] = comps.len() - 1;
    }
  }
  for i in 0..n {
    if outer[i].len() % 2 == 1 {
      // Inner boundary
      // Find out the smallest outer cycle
      let (_, innermost) = outer[i].iter()
        .filter(|&&j| outer[j].len() % 2 == 0)  // Can be removed if no intersection
        .map(|&j| (contain_count[j], j)).min().unwrap();
      comps[comp_index[innermost]].push(normalize_dir(&cycles[i], false));
    }
  }
  comps
}

// Weiler–Atherton algorithm
pub fn intersection(polygons: &[&Vec<Vec<(f32, f32)>>]) -> Vec<Vec<(f32, f32)>> {
  if polygons.is_empty() { vec![] }
  else { polygons[0].clone() }
}
