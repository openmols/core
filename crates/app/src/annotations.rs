//! Measurements and helper objects drawn over the viewport: distances, angles,
//! dihedrals, contact sets, the docking box and spheres.
//!
//! Annotations refer to atoms by *entity name + atom index*, never by
//! coordinates, so they follow an entity when it is posed (G/R) and when an
//! ensemble plays. [`overlay`] turns them into 2D segments and labels for the
//! viewer to paint; there is no depth test (a line behind the protein still
//! shows), which keeps the first version independent of the renderer.

use crate::viewer::ViewerEntity;
use glam::{Mat4, Vec3, Vec4};
use molecule::geometry::{self, ContactKind, Site};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtomRef {
    pub entity: String,
    pub atom: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairKind {
    HBond,
    Contact,
    Clash,
}

impl From<ContactKind> for PairKind {
    fn from(k: ContactKind) -> Self {
        match k {
            ContactKind::HBond => PairKind::HBond,
            ContactKind::Contact => PairKind::Contact,
            ContactKind::Clash => PairKind::Clash,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Annotation {
    Distance { a: AtomRef, b: AtomRef },
    Angle { a: AtomRef, b: AtomRef, c: AtomRef },
    Dihedral { a: AtomRef, b: AtomRef, c: AtomRef, d: AtomRef },
    /// Contacts between two entities, found when the annotation was made.
    Contacts {
        a: String,
        b: String,
        cutoff: f32,
        /// `(atom in a, atom in b, kind)`.
        pairs: Vec<(usize, usize, PairKind)>,
    },
    /// Axis-aligned search box (world space), the grid for docking.
    DockBox { center: [f32; 3], size: [f32; 3] },
    Sphere { center: [f32; 3], radius: f32, label: String },
}

/// What a click in measure mode builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MeasureMode {
    Distance,
    Angle,
    Dihedral,
}

impl MeasureMode {
    pub fn atoms(self) -> usize {
        match self {
            MeasureMode::Distance => 2,
            MeasureMode::Angle => 3,
            MeasureMode::Dihedral => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MeasureMode::Distance => "Distance",
            MeasureMode::Angle => "Angle",
            MeasureMode::Dihedral => "Dihedral",
        }
    }

    /// The annotation these atoms make, if there are enough of them.
    pub fn build(self, atoms: &[AtomRef]) -> Option<Annotation> {
        let at = |i: usize| atoms.get(i).cloned();
        Some(match self {
            MeasureMode::Distance => Annotation::Distance { a: at(0)?, b: at(1)? },
            MeasureMode::Angle => Annotation::Angle { a: at(0)?, b: at(1)?, c: at(2)? },
            MeasureMode::Dihedral => Annotation::Dihedral { a: at(0)?, b: at(1)?, c: at(2)?, d: at(3)? },
        })
    }
}

fn entity<'a>(entities: &'a [ViewerEntity], name: &str) -> Option<&'a ViewerEntity> {
    entities.iter().find(|e| e.name == name)
}

/// World position of an atom, if its entity and the atom still exist.
pub fn world(entities: &[ViewerEntity], r: &AtomRef) -> Option<Vec3> {
    let e = entity(entities, &r.entity)?;
    // A hidden entity takes its measurements with it.
    if !e.visible {
        return None;
    }
    let atom = e.structure.atoms.get(r.atom)?;
    Some(e.transform.transform_point3(atom.position))
}

/// `1.2 Å`-style text for an annotation, and the label shown on screen.
pub fn value_text(entities: &[ViewerEntity], a: &Annotation) -> Option<String> {
    let w = |r: &AtomRef| world(entities, r);
    Some(match a {
        Annotation::Distance { a, b } => format!("{:.2} Å", geometry::distance(w(a)?, w(b)?)),
        Annotation::Angle { a, b, c } => format!("{:.1}°", geometry::angle(w(a)?, w(b)?, w(c)?)),
        Annotation::Dihedral { a, b, c, d } => format!("{:.1}°", geometry::dihedral(w(a)?, w(b)?, w(c)?, w(d)?)),
        Annotation::Contacts { pairs, .. } => {
            let n = |k| pairs.iter().filter(|p| p.2 == k).count();
            format!("{} H-bonds · {} contacts · {} clashes", n(PairKind::HBond), n(PairKind::Contact), n(PairKind::Clash))
        }
        Annotation::DockBox { size, .. } => format!("{:.1} × {:.1} × {:.1} Å", size[0], size[1], size[2]),
        Annotation::Sphere { radius, label, .. } => format!("{label} r={radius:.1} Å"),
    })
}

/// Short kind name for lists.
pub fn kind_name(a: &Annotation) -> &'static str {
    match a {
        Annotation::Distance { .. } => "Distance",
        Annotation::Angle { .. } => "Angle",
        Annotation::Dihedral { .. } => "Dihedral",
        Annotation::Contacts { .. } => "Contacts",
        Annotation::DockBox { .. } => "Dock box",
        Annotation::Sphere { .. } => "Sphere",
    }
}

fn sites(e: &ViewerEntity, only: Option<&[usize]>) -> Vec<Site> {
    let make = |i: usize| Site {
        pos: e.transform.transform_point3(e.structure.atoms[i].position),
        element: e.structure.atoms[i].element,
        id: i,
    };
    match only {
        Some(list) => list.iter().filter(|&&i| i < e.structure.atoms.len()).map(|&i| make(i)).collect(),
        None => (0..e.structure.atoms.len()).map(make).collect(),
    }
}

/// Contacts between two entities (or a selection of the first against the
/// second), as an annotation. Waters are left out unless they are the point.
pub fn contacts_between(
    entities: &[ViewerEntity],
    a: &str,
    b: &str,
    a_atoms: Option<&[usize]>,
    cutoff: f32,
) -> Result<Annotation, String> {
    let ea = entity(entities, a).ok_or_else(|| format!("no entity '{a}'"))?;
    let eb = entity(entities, b).ok_or_else(|| format!("no entity '{b}'"))?;
    let dry = |e: &ViewerEntity, list: Vec<Site>| -> Vec<Site> {
        list.into_iter()
            .filter(|s| !e.structure.residue_of(s.id).is_water())
            .collect()
    };
    let (sa, sb) = (dry(ea, sites(ea, a_atoms)), dry(eb, sites(eb, None)));
    if sa.is_empty() || sb.is_empty() {
        return Err("nothing to compare (empty selection or only water)".into());
    }
    let mut pairs: Vec<(usize, usize, PairKind)> = geometry::contacts(&sa, &sb, cutoff)
        .into_iter()
        .map(|c| (c.a, c.b, c.kind.into()))
        .collect();
    pairs.truncate(600);
    Ok(Annotation::Contacts { a: a.into(), b: b.into(), cutoff, pairs })
}

/// A box around a set of world points with `padding` on every side.
pub fn box_around(points: &[Vec3], padding: f32) -> Option<Annotation> {
    let first = *points.first()?;
    let (lo, hi) = points.iter().fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let center = (lo + hi) * 0.5;
    let size = (hi - lo) + Vec3::splat(2.0 * padding);
    Some(Annotation::DockBox { center: center.to_array(), size: size.to_array() })
}

// ------------------------------------------------------------------ overlay

/// A text label anchored at a point in the world.
pub struct Label {
    pub at: Vec3,
    pub text: String,
    pub color: u32,
}

const DISTANCE: u32 = 0xf2c14e;
const ANGLE: u32 = 0x9be37d;
const DIHEDRAL: u32 = 0xe08df0;
const BOX: u32 = 0x5fb4ff;
const HANDLE: u32 = 0xffffff;
const HANDLE_CENTRE: u32 = 0xff9f43;
const HBOND: u32 = 0x4dd0e1;
const CONTACT: u32 = 0x8a8f99;
const CLASH: u32 = 0xff6b6b;

/// Overlay pick ids: the six box faces (+x, -x, +y, -y, +z, -z), then its centre.
pub const HANDLE_CENTER_ID: u32 = 7;

/// Axis and sign of a face handle id (1..=6).
pub fn handle_axis(id: u32) -> Option<(usize, f32)> {
    match id {
        1..=6 => Some(((id as usize - 1) / 2, if id % 2 == 1 { 1.0 } else { -1.0 })),
        _ => None,
    }
}

fn rgba(color: u32, alpha: u8) -> [u8; 4] {
    [(color >> 16) as u8, (color >> 8) as u8, color as u8, alpha]
}

struct Builder {
    out: render::Overlay,
    labels: Vec<Label>,
    /// World length of one screen pixel at the scene's depth, so lines and
    /// handles keep the same apparent thickness at any zoom.
    unit: f32,
}

impl Builder {
    fn line(&mut self, a: Vec3, b: Vec3, color: u32, thickness: f32, id: u32) {
        self.out.cylinders.push(render::OverlayCylinder {
            p0: a.to_array(),
            p1: b.to_array(),
            radius: (thickness * self.unit).max(0.02),
            color: rgba(color, 255),
            id,
        });
    }

    fn dashed(&mut self, a: Vec3, b: Vec3, color: u32, thickness: f32) {
        let length = a.distance(b);
        let dash = (7.0 * self.unit).max(0.3);
        let step = dash * 1.7;
        if length <= step {
            return self.line(a, b, color, thickness, 0);
        }
        let dir = (b - a) / length;
        let mut at = 0.0;
        while at < length {
            let end = (at + dash).min(length);
            self.line(a + dir * at, a + dir * end, color, thickness, 0);
            at += step;
        }
    }

    fn marker(&mut self, at: Vec3, color: u32, size: f32, id: u32) {
        self.out.spheres.push(render::OverlaySphere {
            center: at.to_array(),
            radius: (size * self.unit).max(0.05),
            color: rgba(color, 255),
            id,
        });
    }

    /// A drag handle: drawn over everything and pickable through the molecule.
    fn handle(&mut self, at: Vec3, color: u32, size: f32, id: u32) {
        self.out.handles.push(render::OverlaySphere {
            center: at.to_array(),
            radius: (size * self.unit).max(0.05),
            color: rgba(color, 255),
            id,
        });
    }

    fn label(&mut self, at: Vec3, text: String, color: u32) {
        self.labels.push(Label { at, text, color });
    }
}

/// The 3D shapes and labels for `annotations` plus the atoms picked so far.
///
/// `unit` is the world size of a screen pixel; `edit_box` adds drag handles to
/// the docking box.
pub fn shapes(
    entities: &[ViewerEntity],
    annotations: &[Annotation],
    hidden: &[bool],
    pending: &[AtomRef],
    unit: f32,
    edit_box: bool,
) -> (render::Overlay, Vec<Label>) {
    let mut b = Builder { out: render::Overlay::default(), labels: Vec::new(), unit };
    let w = |r: &AtomRef| world(entities, r);
    for (index, a) in annotations.iter().enumerate() {
        if hidden.get(index).copied().unwrap_or(false) {
            continue;
        }
        let text = value_text(entities, a).unwrap_or_default();
        match a {
            Annotation::Distance { a, b: end } => {
                if let (Some(p), Some(q)) = (w(a), w(end)) {
                    b.dashed(p, q, DISTANCE, 1.6);
                    b.marker(p, DISTANCE, 3.0, 0);
                    b.marker(q, DISTANCE, 3.0, 0);
                    b.label((p + q) * 0.5, text, DISTANCE);
                }
            }
            Annotation::Angle { a, b: mid, c } => {
                if let (Some(p), Some(m), Some(q)) = (w(a), w(mid), w(c)) {
                    b.dashed(p, m, ANGLE, 1.6);
                    b.dashed(m, q, ANGLE, 1.6);
                    b.marker(m, ANGLE, 3.0, 0);
                    b.label(m, text, ANGLE);
                }
            }
            Annotation::Dihedral { a, b: p2, c, d } => {
                if let (Some(p), Some(q), Some(r), Some(s)) = (w(a), w(p2), w(c), w(d)) {
                    b.dashed(p, q, DIHEDRAL, 1.6);
                    b.line(q, r, DIHEDRAL, 2.2, 0);
                    b.dashed(r, s, DIHEDRAL, 1.6);
                    b.label((q + r) * 0.5, text, DIHEDRAL);
                }
            }
            Annotation::Contacts { a: na, b: nb, pairs, .. } => {
                for &(i, j, kind) in pairs {
                    let (color, thickness) = match kind {
                        PairKind::HBond => (HBOND, 1.5),
                        PairKind::Contact => (CONTACT, 0.9),
                        PairKind::Clash => (CLASH, 1.8),
                    };
                    let ra = AtomRef { entity: na.clone(), atom: i };
                    let rb = AtomRef { entity: nb.clone(), atom: j };
                    if let (Some(p), Some(q)) = (w(&ra), w(&rb)) {
                        if kind == PairKind::Contact {
                            b.line(p, q, color, thickness, 0);
                        } else {
                            b.dashed(p, q, color, thickness);
                        }
                    }
                }
            }
            Annotation::DockBox { center, size } => {
                let (c, h) = (Vec3::from(*center), Vec3::from(*size) * 0.5);
                let corner = |i: usize| {
                    c + Vec3::new(
                        if i & 1 == 0 { -h.x } else { h.x },
                        if i & 2 == 0 { -h.y } else { h.y },
                        if i & 4 == 0 { -h.z } else { h.z },
                    )
                };
                for i in 0..8usize {
                    for bit in [1usize, 2, 4] {
                        if i & bit == 0 {
                            b.line(corner(i), corner(i | bit), BOX, 1.8, 0);
                        }
                    }
                }
                // Faint faces so the box reads as a volume.
                for (axis, sign) in [(0usize, -1.0f32), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)] {
                    let mut u = [0.0; 3];
                    let (a1, a2) = ((axis + 1) % 3, (axis + 2) % 3);
                    let mut quad = [Vec3::ZERO; 4];
                    for (k, (s1, s2)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
                        u[axis] = sign * h[axis];
                        u[a1] = s1 * h[a1];
                        u[a2] = s2 * h[a2];
                        quad[k] = c + Vec3::from(u);
                    }
                    b.out.faces.push(render::OverlayFace {
                        corners: quad.map(|p| p.to_array()),
                        color: rgba(BOX, 28),
                    });
                }
                if edit_box {
                    for id in 1..=6u32 {
                        if let Some((axis, sign)) = handle_axis(id) {
                            let mut at = c;
                            at[axis] += sign * h[axis];
                            b.handle(at, HANDLE, 8.0, id);
                        }
                    }
                    b.handle(c, HANDLE_CENTRE, 10.0, HANDLE_CENTER_ID);
                }
                b.label(corner(7), text, BOX);
            }
            Annotation::Sphere { center, radius, .. } => {
                let (c, r) = (Vec3::from(*center), *radius);
                const STEPS: usize = 48;
                for axis in 0..3 {
                    let at = |k: usize| {
                        let t = k as f32 / STEPS as f32 * std::f32::consts::TAU;
                        let (s, co) = (t.sin() * r, t.cos() * r);
                        c + match axis {
                            0 => Vec3::new(co, s, 0.0),
                            1 => Vec3::new(0.0, co, s),
                            _ => Vec3::new(s, 0.0, co),
                        }
                    };
                    for k in 0..STEPS {
                        b.line(at(k), at(k + 1), BOX, 1.4, 0);
                    }
                }
                b.label(c, text, BOX);
            }
        }
    }
    // Atoms picked for a measurement still in progress.
    for r in pending {
        if let Some(pos) = w(r) {
            b.marker(pos, HANDLE, 4.5, 0);
        }
    }
    (b.out, b.labels)
}

/// Projects world points to viewport pixels.
pub struct Projector {
    pub matrix: Mat4,
    pub width: f32,
    pub height: f32,
}

impl Projector {
    /// `None` for points behind the camera.
    pub fn point(&self, p: Vec3) -> Option<(f32, f32)> {
        let c = self.matrix * Vec4::new(p.x, p.y, p.z, 1.0);
        if c.w <= 1e-4 {
            return None;
        }
        let n = c.truncate() / c.w;
        Some(((n.x * 0.5 + 0.5) * self.width, (1.0 - (n.y * 0.5 + 0.5)) * self.height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_covers_points_with_padding() {
        let pts = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 2.0, 6.0)];
        let Annotation::DockBox { center, size } = box_around(&pts, 5.0).unwrap() else { panic!() };
        assert_eq!(center, [2.0, 1.0, 3.0]);
        assert_eq!(size, [14.0, 12.0, 16.0]);
    }

    #[test]
    fn modes_need_their_atoms() {
        let r = |n: usize| AtomRef { entity: "x".into(), atom: n };
        assert!(MeasureMode::Angle.build(&[r(0), r(1)]).is_none());
        assert!(matches!(MeasureMode::Distance.build(&[r(0), r(1)]), Some(Annotation::Distance { .. })));
    }

    #[test]
    fn hidden_annotations_draw_nothing() {
        let boxed = [Annotation::DockBox { center: [0.0; 3], size: [10.0; 3] }];
        let (shown, labels) = shapes(&[], &boxed, &[], &[], 0.01, true);
        assert!(!shown.cylinders.is_empty() && labels.len() == 1);
        let (hidden, labels) = shapes(&[], &boxed, &[true], &[], 0.01, true);
        assert!(hidden.cylinders.is_empty() && hidden.faces.is_empty() && hidden.handles.is_empty() && labels.is_empty());
    }

    #[test]
    fn handles_cover_each_axis_once() {
        let axes: Vec<_> = (1..=6).filter_map(handle_axis).collect();
        assert_eq!(axes, vec![(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)]);
        assert_eq!(handle_axis(HANDLE_CENTER_ID), None);
    }

    #[test]
    fn a_box_draws_twelve_edges_and_six_faces_and_handles_when_editable() {
        let boxed = [Annotation::DockBox { center: [1.0, 2.0, 3.0], size: [10.0, 10.0, 10.0] }];
        let (plain, labels) = shapes(&[], &boxed, &[], &[], 0.01, false);
        assert_eq!(plain.cylinders.len(), 12);
        assert_eq!(plain.faces.len(), 6);
        assert!(plain.spheres.is_empty() && plain.handles.is_empty());
        assert_eq!(labels.len(), 1);
        let (editable, _) = shapes(&[], &boxed, &[], &[], 0.01, true);
        assert_eq!(editable.handles.len(), 7);
        let ids: Vec<u32> = editable.handles.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn projection_centres_the_origin() {
        let p = Projector { matrix: Mat4::IDENTITY, width: 200.0, height: 100.0 };
        let (x, y) = p.point(Vec3::new(0.0, 0.0, 0.0)).unwrap();
        assert_eq!((x, y), (100.0, 50.0));
    }
}
