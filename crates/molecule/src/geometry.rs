//! Measurements between atoms and groups of atoms: distances, angles,
//! dihedrals, contacts (H-bonds, clashes), solvent-accessible surface and
//! van der Waals volume.
//!
//! Everything works on world-space [`Site`]s rather than a [`crate::Structure`],
//! because the app moves entities rigidly and measures between several of
//! them: the caller applies each entity's transform and hands the points here.

use crate::Element;
use glam::Vec3;
use std::collections::HashMap;

/// An atom in world space.
#[derive(Copy, Clone, Debug)]
pub struct Site {
    pub pos: Vec3,
    pub element: Element,
    /// Caller's identifier (usually the atom index in its structure).
    pub id: usize,
}

pub fn distance(a: Vec3, b: Vec3) -> f32 {
    a.distance(b)
}

/// Angle at `b` between `a` and `c`, degrees.
pub fn angle(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let u = (a - b).normalize_or_zero();
    let v = (c - b).normalize_or_zero();
    u.dot(v).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Torsion about the b–c axis, degrees in (-180, 180] (IUPAC sign).
pub fn dihedral(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    let b1 = b - a;
    let b2 = c - b;
    let b3 = d - c;
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);
    let m1 = n1.cross(b2.normalize_or_zero());
    let x = n1.dot(n2);
    let y = m1.dot(n2);
    (-y).atan2(x).to_degrees()
}

/// Centroid of a set of points.
pub fn centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    points.iter().sum::<Vec3>() / points.len() as f32
}

/// Least-squares plane through `points`: `(centroid, unit normal)`.
/// Needs at least three non-collinear points.
pub fn best_fit_plane(points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    if points.len() < 3 {
        return None;
    }
    let c = centroid(points);
    let mut m = nalgebra::Matrix3::<f32>::zeros();
    for p in points {
        let d = nalgebra::Vector3::new(p.x - c.x, p.y - c.y, p.z - c.z);
        m += d * d.transpose();
    }
    let eig = nalgebra::SymmetricEigen::new(m);
    let smallest = (0..3).min_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]))?;
    let n = eig.eigenvectors.column(smallest);
    let normal = Vec3::new(n[0], n[1], n[2]).normalize_or_zero();
    (normal != Vec3::ZERO).then_some((c, normal))
}

/// Uniform grid for neighbour queries.
struct Grid {
    cell: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl Grid {
    fn new(points: impl Iterator<Item = Vec3>, cell: f32) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, p) in points.enumerate() {
            cells.entry(Self::key(p, cell)).or_default().push(i);
        }
        Grid { cell, cells }
    }

    fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
        ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32)
    }

    /// Indices in the cells around `p` (a superset of everything within one cell size).
    fn near(&self, p: Vec3, reach: f32) -> impl Iterator<Item = usize> + '_ {
        let (cx, cy, cz) = Self::key(p, self.cell);
        let n = (reach / self.cell).ceil() as i32;
        (-n..=n).flat_map(move |dx| {
            (-n..=n).flat_map(move |dy| {
                (-n..=n).flat_map(move |dz| {
                    self.cells
                        .get(&(cx + dx, cy + dy, cz + dz))
                        .into_iter()
                        .flat_map(|v| v.iter().copied())
                })
            })
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactKind {
    /// N/O pair close enough to hydrogen-bond (donor/acceptor roles are not
    /// assigned: hydrogens are usually absent).
    HBond,
    /// Any other pair within the cutoff.
    Contact,
    /// Closer than 75% of the summed van der Waals radii.
    Clash,
}

#[derive(Copy, Clone, Debug)]
pub struct Contact {
    /// `Site::id` of the atom from the first and second group.
    pub a: usize,
    pub b: usize,
    pub distance: f32,
    pub kind: ContactKind,
}

fn is_polar(e: Element) -> bool {
    matches!(e.symbol(), "N" | "O")
}

/// Contacts between two groups of atoms, closest first.
///
/// `cutoff` is the largest distance reported (4.0 Å is typical). N···O pairs
/// within 3.5 Å are H-bonds, anything under 0.75·(rA+rB) is a clash (checked
/// first, so an N···O pair at 2.2 Å is a clash, not a bond).
pub fn contacts(a: &[Site], b: &[Site], cutoff: f32) -> Vec<Contact> {
    const HBOND: f32 = 3.5;
    let reach = cutoff.max(HBOND);
    let grid = Grid::new(b.iter().map(|s| s.pos), reach.max(1.0));
    let mut out = Vec::new();
    for sa in a {
        for j in grid.near(sa.pos, reach) {
            let sb = &b[j];
            let d = sa.pos.distance(sb.pos);
            if d > reach {
                continue;
            }
            let clash = d < 0.75 * (sa.element.vdw_radius() + sb.element.vdw_radius());
            let kind = if clash {
                ContactKind::Clash
            } else if is_polar(sa.element) && is_polar(sb.element) && d <= HBOND {
                ContactKind::HBond
            } else if d <= cutoff {
                ContactKind::Contact
            } else {
                continue;
            };
            out.push(Contact { a: sa.id, b: sb.id, distance: d, kind });
        }
    }
    out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
    out
}

/// Fraction of `points` within `cutoff` of any of `targets` (e.g. ligand atoms
/// against the probe points of a cavity).
pub fn fraction_within(points: &[Vec3], targets: &[Vec3], cutoff: f32) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    let grid = Grid::new(targets.iter().copied(), cutoff.max(0.5));
    let inside = points
        .iter()
        .filter(|p| grid.near(**p, cutoff).any(|i| targets[i].distance(**p) <= cutoff))
        .count();
    inside as f32 / points.len() as f32
}

/// Evenly spread points on the unit sphere (Fibonacci lattice).
fn sphere_points(n: usize) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let t = golden * i as f32;
            Vec3::new(r * t.cos(), y, r * t.sin())
        })
        .collect()
}

/// Solvent-accessible surface area per atom, Å² (Shrake–Rupley).
///
/// Every atom in `sites` occludes; areas are computed only for the indices in
/// `of` (or all when `None`) and are zero for the rest. This is how the area
/// of a ligand *in its pocket* is measured: pass the complex, ask for the ligand.
pub fn sasa(sites: &[Site], probe: f32, points: usize, of: Option<&[usize]>) -> Vec<f32> {
    let radii: Vec<f32> = sites.iter().map(|s| s.element.vdw_radius() + probe).collect();
    let max_r = radii.iter().cloned().fold(0.0, f32::max);
    let grid = Grid::new(sites.iter().map(|s| s.pos), (2.0 * max_r).max(1.0));
    let unit = sphere_points(points.max(10));
    let mut out = vec![0.0; sites.len()];
    let all: Vec<usize>;
    let indices: &[usize] = match of {
        Some(v) => v,
        None => {
            all = (0..sites.len()).collect();
            &all
        }
    };
    for &i in indices {
        let (c, r) = (sites[i].pos, radii[i]);
        let neighbours: Vec<usize> = grid
            .near(c, r + max_r)
            .filter(|&j| j != i && sites[j].pos.distance(c) < r + radii[j])
            .collect();
        let exposed = unit
            .iter()
            .filter(|u| {
                let p = c + **u * r;
                neighbours.iter().all(|&j| sites[j].pos.distance_squared(p) >= radii[j] * radii[j])
            })
            .count();
        out[i] = 4.0 * std::f32::consts::PI * r * r * exposed as f32 / unit.len() as f32;
    }
    out
}

/// Volume of the union of the atoms' van der Waals spheres, Å³, by counting
/// grid cells (`spacing` Å; 0.5 is within a percent for protein-sized sets).
pub fn vdw_volume(sites: &[Site], spacing: f32) -> f32 {
    if sites.is_empty() {
        return 0.0;
    }
    let spacing = spacing.max(0.1);
    let max_r = sites.iter().map(|s| s.element.vdw_radius()).fold(0.0, f32::max);
    let grid = Grid::new(sites.iter().map(|s| s.pos), (2.0 * max_r).max(1.0));
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for s in sites {
        let r = Vec3::splat(s.element.vdw_radius());
        lo = lo.min(s.pos - r);
        hi = hi.max(s.pos + r);
    }
    let n = ((hi - lo) / spacing).ceil().as_ivec3().max(glam::IVec3::ONE);
    let mut inside = 0u64;
    for ix in 0..n.x {
        for iy in 0..n.y {
            for iz in 0..n.z {
                let p = lo + (Vec3::new(ix as f32, iy as f32, iz as f32) + 0.5) * spacing;
                let hit = grid.near(p, max_r).any(|j| {
                    let r = sites[j].element.vdw_radius();
                    sites[j].pos.distance_squared(p) <= r * r
                });
                if hit {
                    inside += 1;
                }
            }
        }
    }
    inside as f32 * spacing.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(x: f32, y: f32, z: f32, symbol: &str, id: usize) -> Site {
        Site { pos: Vec3::new(x, y, z), element: Element::from_symbol(symbol), id }
    }

    #[test]
    fn angle_and_dihedral() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::ZERO;
        let c = Vec3::new(0.0, 1.0, 0.0);
        assert!((angle(a, b, c) - 90.0).abs() < 1e-4);
        // Trans arrangement: 180°, cis: 0°, and a quarter turn: ±90°.
        let (p0, p1) = (Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO);
        let p2 = Vec3::new(1.0, 0.0, 0.0);
        assert!((dihedral(p0, p1, p2, Vec3::new(1.0, -1.0, 0.0)).abs() - 180.0).abs() < 1e-3);
        assert!(dihedral(p0, p1, p2, Vec3::new(1.0, 1.0, 0.0)).abs() < 1e-3);
        assert!((dihedral(p0, p1, p2, Vec3::new(1.0, 0.0, 1.0)).abs() - 90.0).abs() < 1e-3);
    }

    #[test]
    fn plane_normal() {
        let pts = [
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(1.0, 0.0, 2.0),
            Vec3::new(0.0, 1.0, 2.0),
            Vec3::new(1.0, 1.0, 2.0),
        ];
        let (c, n) = best_fit_plane(&pts).unwrap();
        assert!((c.z - 2.0).abs() < 1e-5);
        assert!(n.z.abs() > 0.999);
    }

    #[test]
    fn contact_kinds() {
        let a = [site(0.0, 0.0, 0.0, "N", 1), site(10.0, 0.0, 0.0, "C", 2)];
        let b = [site(2.9, 0.0, 0.0, "O", 7), site(10.0, 3.8, 0.0, "C", 8), site(0.0, 1.5, 0.0, "O", 9)];
        let found = contacts(&a, &b, 4.0);
        let kind = |x, y| found.iter().find(|c| c.a == x && c.b == y).map(|c| c.kind);
        assert_eq!(kind(1, 7), Some(ContactKind::HBond));
        assert_eq!(kind(2, 8), Some(ContactKind::Contact));
        assert_eq!(kind(1, 9), Some(ContactKind::Clash));
        assert!(found.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn sasa_of_one_and_two_atoms() {
        let one = [site(0.0, 0.0, 0.0, "C", 0)];
        let r = one[0].element.vdw_radius() + 1.4;
        let expected = 4.0 * std::f32::consts::PI * r * r;
        let got = sasa(&one, 1.4, 400, None)[0];
        assert!((got - expected).abs() / expected < 0.02, "{got} vs {expected}");
        // A neighbour touching it hides part of the surface.
        let two = [one[0], site(1.5, 0.0, 0.0, "C", 1)];
        assert!(sasa(&two, 1.4, 400, None)[0] < got * 0.9);
        // Only the requested atoms are computed.
        let only = sasa(&two, 1.4, 400, Some(&[1]));
        assert_eq!(only[0], 0.0);
        assert!(only[1] > 0.0);
    }

    #[test]
    fn volume_of_a_sphere() {
        let s = [site(0.0, 0.0, 0.0, "C", 0)];
        let r = s[0].element.vdw_radius();
        let expected = 4.0 / 3.0 * std::f32::consts::PI * r.powi(3);
        let got = vdw_volume(&s, 0.2);
        assert!((got - expected).abs() / expected < 0.04, "{got} vs {expected}");
    }

    #[test]
    fn fraction_inside() {
        let pts = [Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let targets = [Vec3::new(0.5, 0.0, 0.0)];
        assert_eq!(fraction_within(&pts, &targets, 1.0), 0.5);
    }
}
