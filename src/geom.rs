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
pub fn seg_intxn(
  a: ((f32, f32), (f32, f32)),
  b: ((f32, f32), (f32, f32)),
) -> Option<(f32, f32)> {
  let ((x1, y1), (x2, y2)) = a;
  let ((x3, y3), (x4, y4)) = b;
  let d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if d == 0.0 { return None; }
  let ta = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d;
  let tb = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d;
  if ta >= 0.0 && ta <= 1.0 && tb >= 0.0 && tb <= 1.0 {
    Some(lerp((x1, y1), (x2, y2), ta))
  } else {
    None
  }
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

// Direction is in the left-handed coordinate system
fn normalize_dir(cycle: &[(f32, f32)], ccw: bool) -> Vec<(f32, f32)> {
  // if (det3(cycle[0], cycle[1], cycle[2]) > 0.0) != ccw {
  let mut area = 0.0;
  for i in 0..cycle.len() {
    let (x1, y1) = cycle[i];
    let (x2, y2) = cycle[(i + 1) % cycle.len()];
    area += (x2 - x1) * (y1 + y2);
  }
  if (area <= 0.0) != ccw {
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

// Each vector returned corresponds to one segment in the input
// in the input order
fn all_segment_intersections(segs: &[((f32, f32), (f32, f32))])
-> Vec<Vec<(usize, (f32, f32))>> {
  // TODO: Optimize
  let mut result = vec![];
  for i in 0..segs.len() {
    let mut with_cur = vec![];
    for j in 0..segs.len() {
      if i != j {
        if let Some(p) = seg_intxn(segs[i], segs[j]) {
          with_cur.push((j, p));
        }
      }
    }
    result.push(with_cur);
  }
  result
}

pub fn intersection(polygons: &[&Vec<Vec<(f32, f32)>>]) -> Vec<Vec<(f32, f32)>> {
  if polygons.is_empty() {
    vec![]
  } else {
    polygons.iter().skip(1).fold(
      polygons[0].clone(),
      |a, &b| intersection_two([&a, b])
    )
  }
}

// Weiler–Atherton algorithm
// Intersection of two polygons, each represented by a collection of cycles
fn intersection_two(polygons: [&[Vec<(f32, f32)>]; 2]) -> Vec<Vec<(f32, f32)>> {
  // Collect segments of polygons
  let mut segs = vec![];
  let mut segs_origin = vec![];
  let mut normalized_polygons = vec![];
  for (i, poly) in polygons.iter().enumerate() {
    // Into Vec<components: Vec<cycles: Vec<points>>>
    let poly = normalize_polygon(poly);
    // Into Vec<cycles: Vec<points>>
    let poly = poly.iter().flatten().map(|x| x.clone()).collect::<Vec<_>>();
    for (j, cyc) in poly.iter().enumerate() {
      for k in 0..cyc.len() {
        // Tag: (polygon index (0/1), cycle index, vertex index)
        segs.push((cyc[k], cyc[(k + 1) % cyc.len()]));
        segs_origin.push((i, j, k));
      }
    }
    normalized_polygons.push(poly);
  }

  // Find out all intersections and build vertex lists
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  enum Vertex {
    Polygon(usize, usize, usize), // (polygon index, cycle index, vertex index)
    Intersection(usize, usize),   // (smaller segment index, larger segment index)
  };
  use Vertex::*;
  let intxns = all_segment_intersections(&segs);
  let mut intxns_coords = std::collections::HashMap::new();
  // intxns_is_enter is true means that
  // directed segment of polygon 1 is coming into the interior of polygon 0
  // This may be reversed since right-handed and left-handed coordinates are
  // mixed here, depending on the interpretation of coordinates.
  let mut intxns_is_enter = std::collections::HashMap::new();
  let mut links = [std::collections::HashMap::new(), std::collections::HashMap::new()];
  let mut real_intxns_unvisited = std::collections::HashSet::new();
  for (seg1_index, with_seg1) in intxns.iter().enumerate() {
    let (i1, j1, k1) = segs_origin[seg1_index];
    let mut from_other_with_seg1 = vec![];
    for &(seg2_index, p) in with_seg1 {
      let (i2, j2, k2) = segs_origin[seg2_index];
      if i1 != i2 {
        // Intersection between two polygons
        let min_index = seg1_index.min(seg2_index);
        let max_index = seg1_index.max(seg2_index);
        from_other_with_seg1.push((min_index, max_index, p));
        intxns_coords.insert((min_index, max_index), p);
        intxns_is_enter.insert((min_index, max_index),
          (det3(segs[seg1_index].0, p, segs[seg2_index].1) >= 0.0)
          ^ (i1 == 0));
        real_intxns_unvisited.insert((min_index, max_index));
      }
    }
    from_other_with_seg1.sort_by(
      |&(_, _, p1), &(_, _, p2)|
        dist_sq(p1, segs[seg1_index].0).partial_cmp(
       &dist_sq(p2, segs[seg1_index].0)).unwrap_or(std::cmp::Ordering::Equal)
    );
    let list =
      std::iter::once(Polygon(i1, j1, k1))
        .chain(from_other_with_seg1.iter()
                .map(|&(idx1, idx2, _)| Intersection(idx1, idx2)))
        .chain(std::iter::once(Polygon(i1, j1, (k1 + 1) % normalized_polygons[i1][j1].len())))
        .collect::<Vec<_>>();
    // println!("{:?}", list);
    for i in 0..(list.len() - 1) {
      links[i1].insert(list[i], list[i + 1]);
    }
  }
  // println!("0 - {:?}", links[0]);
  // println!("1 - {:?}", links[1]);

  let mut result_cycles = vec![];
  while !real_intxns_unvisited.is_empty() {
    let mut cur_cycle = vec![];
    // Remove any tuple (segment index, segment index) from
    // the collection of intersections
    let intxn = real_intxns_unvisited.iter().next().unwrap().clone();
    let start = Intersection(intxn.0, intxn.1);
    let mut polygon_index =
      if *intxns_is_enter.get(&intxn).unwrap() == true { 1 } else { 0 };
    let mut current = start;
    loop {
      cur_cycle.push(match current {
        Polygon(i, j, k) => normalized_polygons[i][j][k],
        Intersection(idx1, idx2) => *intxns_coords.get(&(idx1, idx2)).unwrap(),
      });
      if let Intersection(idx1, idx2) = current {
        real_intxns_unvisited.remove(&(idx1, idx2));
        polygon_index = 1 - polygon_index;
      }
      // Traverse down
      current = links[polygon_index].get(&current).unwrap().clone();
      // println!("{:?}", current);
      if current == start { break; }
    }
    // println!("-- {}", real_intxns_unvisited.len());
    result_cycles.push(cur_cycle);
  }

  result_cycles
}
