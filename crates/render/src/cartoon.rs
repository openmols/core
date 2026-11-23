//! Cartoon representation: a smooth spline through the Cα (or phosphate)
//! trace, swept with an elliptical profile that widens into ribbons for
//! helices and flat arrows for β-strands.

use crate::scene::{Coloring, Mesh, MeshVertex, Palette};
use glam::Vec3;
use molecule::{SecondaryStructure as Ss, Structure};

/// Spline samples per residue.
const SUBDIVISIONS: usize = 8;
/// Points around the profile.
const PROFILE: usize = 16;

/// (half-width, half-thickness) in Å.
const COIL: (f32, f32) = (0.22, 0.22);
const HELIX: (f32, f32) = (1.15, 0.22);
const SHEET: (f32, f32) = (0.95, 0.28);
const ARROW_HALF_WIDTH: f32 = 1.6;
const NUCLEIC: (f32, f32) = (0.45, 0.45);

struct TraceResidue {
    residue: usize,
    atom: usize,
    position: Vec3,
    /// Approximate in-plane direction (Cα→O for proteins).
    orientation: Vec3,
    ss: Ss,
    nucleic: bool,
}

pub(crate) fn build(structure: &Structure, selected: &[bool], palette: &Palette, coloring: Coloring) -> Mesh {
    let mut mesh = Mesh::default();
    for segment in segments(structure, selected) {
        if segment.len() >= 2 {
            sweep(structure, &segment, palette, coloring, &mut mesh);
        }
    }
    mesh
}

fn segments(structure: &Structure, selected: &[bool]) -> Vec<Vec<TraceResidue>> {
    let pos = |i: usize| structure.atoms[i].position;
    let mut segments: Vec<Vec<TraceResidue>> = Vec::new();
    let mut current: Vec<TraceResidue> = Vec::new();

    for (ri, residue) in structure.residues.iter().enumerate() {
        let ca = structure.residue_atom(ri, "CA").filter(|&a| selected[a]);
        let p = structure.residue_atom(ri, "P").filter(|&a| selected[a]);
        let entry = if let Some(ca) = ca {
            let orientation = match (structure.residue_atom(ri, "C"), structure.residue_atom(ri, "O")) {
                (Some(c), Some(o)) => pos(o) - pos(c),
                _ => Vec3::ZERO,
            };
            Some(TraceResidue {
                residue: ri,
                atom: ca,
                position: pos(ca),
                orientation,
                ss: structure.secondary.get(ri).copied().unwrap_or_default(),
                nucleic: false,
            })
        } else if let Some(p) = p {
            let toward = ["C1'", "C4'", "C3'"]
                .iter()
                .find_map(|n| structure.residue_atom(ri, n))
                .map(|a| pos(a) - pos(p))
                .unwrap_or(Vec3::ZERO);
            Some(TraceResidue {
                residue: ri,
                atom: p,
                position: pos(p),
                orientation: toward,
                ss: Ss::Coil,
                nucleic: true,
            })
        } else {
            None
        };

        let Some(entry) = entry else {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            continue;
        };
        if let Some(prev) = current.last() {
            let max = if entry.nucleic { 8.0 } else { 4.3 };
            let same_chain = structure.residues[prev.residue].chain == residue.chain;
            if !same_chain || prev.nucleic != entry.nucleic || prev.position.distance(entry.position) > max {
                segments.push(std::mem::take(&mut current));
            }
        }
        current.push(entry);
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

fn profile(r: &TraceResidue) -> (f32, f32) {
    if r.nucleic {
        return NUCLEIC;
    }
    match r.ss {
        Ss::Helix => HELIX,
        Ss::Sheet => SHEET,
        Ss::Coil => COIL,
    }
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> (Vec3, Vec3) {
    let t2 = t * t;
    let t3 = t2 * t;
    let pos = 0.5
        * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
    let tangent = 0.5
        * ((-p0 + p2)
            + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t
            + 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2);
    (pos, tangent)
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn sweep(structure: &Structure, seg: &[TraceResidue], palette: &Palette, coloring: Coloring, mesh: &mut Mesh) {
    let n = seg.len();

    // Control points; strands are smoothed so they don't zig-zag.
    let mut points: Vec<Vec3> = seg.iter().map(|r| r.position).collect();
    for i in 1..n.saturating_sub(1) {
        if seg[i].ss == Ss::Sheet && seg[i - 1].ss == Ss::Sheet && seg[i + 1].ss == Ss::Sheet {
            points[i] = (seg[i - 1].position + 2.0 * seg[i].position + seg[i + 1].position) * 0.25;
        }
    }
    let control = |i: isize| -> Vec3 {
        if i < 0 {
            2.0 * points[0] - points[1]
        } else if i as usize >= n {
            2.0 * points[n - 1] - points[n - 2]
        } else {
            points[i as usize]
        }
    };

    // Orientation vectors, made continuous (no 180° flips between residues).
    let mut orient: Vec<Vec3> = Vec::with_capacity(n);
    for (i, r) in seg.iter().enumerate() {
        let tangent = (control(i as isize + 1) - control(i as isize - 1)).normalize_or_zero();
        let mut o = r.orientation - tangent * r.orientation.dot(tangent);
        if o.length_squared() < 1e-6 {
            o = orient.last().copied().unwrap_or_else(|| tangent.any_orthonormal_vector());
        }
        let mut o = o.normalize();
        if let Some(prev) = orient.last()
            && o.dot(*prev) < 0.0
        {
            o = -o;
        }
        orient.push(o);
    }

    let mut rings = 0usize;
    let base = mesh.vertices.len() as u32;
    let mut first_center = None;
    let mut last = (Vec3::ZERO, Vec3::ZERO, [0u8; 4], 0u32);

    for i in 0..n - 1 {
        let samples = if i == n - 2 { SUBDIVISIONS + 1 } else { SUBDIVISIONS };
        let (w0, h0) = profile(&seg[i]);
        let (w1, h1) = profile(&seg[i + 1]);
        let arrow = seg[i].ss == Ss::Sheet && seg[i + 1].ss != Ss::Sheet;
        for s in 0..samples {
            let t = s as f32 / SUBDIVISIONS as f32;
            let (pos, tangent) = catmull_rom(
                control(i as isize - 1),
                control(i as isize),
                control(i as isize + 1),
                control(i as isize + 2),
                t,
            );
            let tangent = tangent.normalize_or_zero();
            let side = orient[i].lerp(orient[i + 1], t);
            let side = (side - tangent * side.dot(tangent)).normalize_or(tangent.any_orthonormal_vector());
            let up = tangent.cross(side).normalize_or_zero();

            let (w, h) = if arrow {
                (ARROW_HALF_WIDTH + (COIL.0 - ARROW_HALF_WIDTH) * t, h0 + (h1 - h0) * smoothstep(t))
            } else {
                let k = smoothstep(t);
                (w0 + (w1 - w0) * k, h0 + (h1 - h0) * k)
            };
            let owner = if t < 0.5 { &seg[i] } else { &seg[i + 1] };
            let color = palette.residue_color(structure, owner.residue, coloring);
            let id = owner.atom as u32;

            for k in 0..PROFILE {
                let a = k as f32 / PROFILE as f32 * std::f32::consts::TAU;
                let (sin, cos) = a.sin_cos();
                let offset = side * (cos * w) + up * (sin * h);
                let normal = (side * (cos / w.max(1e-3)) + up * (sin / h.max(1e-3))).normalize_or_zero();
                mesh.vertices.push(MeshVertex {
                    position: (pos + offset).to_array(),
                    normal: normal.to_array(),
                    color,
                    id,
                });
            }
            if first_center.is_none() {
                first_center = Some((pos, tangent, color, id));
            }
            last = (pos, tangent, color, id);
            rings += 1;
        }
    }

    // Tube walls.
    for r in 0..rings - 1 {
        let a = base + (r * PROFILE) as u32;
        let b = a + PROFILE as u32;
        for k in 0..PROFILE as u32 {
            let k1 = (k + 1) % PROFILE as u32;
            mesh.indices.extend_from_slice(&[a + k, b + k, b + k1, a + k, b + k1, a + k1]);
        }
    }

    // End caps.
    let mut cap = |center: (Vec3, Vec3, [u8; 4], u32), ring: u32, outward: f32| {
        let c = mesh.vertices.len() as u32;
        mesh.vertices.push(MeshVertex {
            position: center.0.to_array(),
            normal: (center.1 * outward).to_array(),
            color: center.2,
            id: center.3,
        });
        let ring = base + ring * PROFILE as u32;
        for k in 0..PROFILE as u32 {
            let k1 = (k + 1) % PROFILE as u32;
            mesh.indices.extend_from_slice(&[c, ring + k, ring + k1]);
        }
    };
    if let Some(first) = first_center {
        cap(first, 0, -1.0);
    }
    cap(last, (rings - 1) as u32, 1.0);
}
