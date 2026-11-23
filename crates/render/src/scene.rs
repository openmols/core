//! Turn a [`Structure`] plus a list of [`Component`]s into GPU data.
//!
//! A component is "representation × target × colouring", the same model as
//! Mol*'s presets (e.g. polymer → cartoon, ligand → ball & stick).

use crate::{cartoon, surface};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use molecule::{AtomKind, Element, SecondaryStructure, Structure};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    BallAndStick,
    Spacefill,
    /// Sticks only: atoms are bond-thick spheres.
    Licorice,
    /// Cα / phosphate trace.
    Backbone,
    Cartoon,
    GaussianSurface,
}

impl Representation {
    pub const ALL: [Representation; 6] = [
        Representation::Cartoon,
        Representation::BallAndStick,
        Representation::Spacefill,
        Representation::Licorice,
        Representation::Backbone,
        Representation::GaussianSurface,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Representation::BallAndStick => "Ball & stick",
            Representation::Spacefill => "Spacefill",
            Representation::Licorice => "Licorice",
            Representation::Backbone => "Backbone",
            Representation::Cartoon => "Cartoon",
            Representation::GaussianSurface => "Gaussian surface",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    All,
    Polymer,
    Ligand,
    Water,
    /// The current selection passed to [`crate::Renderer::set_scene`].
    Selection,
}

impl Target {
    pub fn label(self) -> &'static str {
        match self {
            Target::All => "All",
            Target::Polymer => "Polymer",
            Target::Ligand => "Ligand",
            Target::Water => "Water",
            Target::Selection => "Selection",
        }
    }

    pub(crate) fn matches(self, kind: AtomKind, selected: bool) -> bool {
        match self {
            Target::All => true,
            Target::Polymer => kind == AtomKind::Polymer,
            Target::Ligand => kind == AtomKind::Ligand,
            Target::Water => kind == AtomKind::Water,
            Target::Selection => selected,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coloring {
    /// CPK element colours; carbons coloured by chain (ligands: green).
    Element,
    Chain,
    Secondary,
    /// Every residue its own colour (e.g. one colour per detected cavity).
    Residue,
}

impl Coloring {
    pub fn label(self) -> &'static str {
        match self {
            Coloring::Element => "Element",
            Coloring::Chain => "Chain",
            Coloring::Secondary => "Secondary structure",
            Coloring::Residue => "Residue",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Component {
    pub target: Target,
    pub representation: Representation,
    pub coloring: Coloring,
}

impl Component {
    pub const fn new(target: Target, representation: Representation, coloring: Coloring) -> Self {
        Self { target, representation, coloring }
    }
}

/// Mol*-like default: cartoon polymer, ball & stick ligands, no water.
pub fn default_preset(structure: &Structure) -> Vec<Component> {
    let has_polymer_trace = structure
        .residues
        .iter()
        .enumerate()
        .any(|(i, _)| structure.residue_atom(i, "CA").is_some() || structure.residue_atom(i, "P").is_some());
    let mut components = Vec::new();
    if has_polymer_trace {
        components.push(Component::new(Target::Polymer, Representation::Cartoon, Coloring::Chain));
    } else {
        components.push(Component::new(Target::Polymer, Representation::BallAndStick, Coloring::Element));
    }
    components.push(Component::new(Target::Ligand, Representation::BallAndStick, Coloring::Element));
    components
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SphereInstance {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [u8; 4],
    pub id: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct CylinderInstance {
    pub p0: [f32; 3],
    pub radius: f32,
    pub p1: [f32; 3],
    pub color0: [u8; 4],
    pub color1: [u8; 4],
    pub id0: u32,
    pub id1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [u8; 4],
    pub id: u32,
}

#[derive(Default)]
pub(crate) struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn append(&mut self, other: Mesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.indices.extend(other.indices.into_iter().map(|i| i + base));
    }
}

#[derive(Default)]
pub(crate) struct SceneData {
    pub spheres: Vec<SphereInstance>,
    pub cylinders: Vec<CylinderInstance>,
    pub mesh: Mesh,
}

/// Carbon / chain colours (soft, Mol*-like palette).
const CHAIN_PALETTE: &[[u8; 3]] = &[
    [0x8F, 0xB8, 0xDE],
    [0xE8, 0xA3, 0x8C],
    [0x9C, 0xD0, 0x9A],
    [0xC9, 0xA7, 0xE0],
    [0xE6, 0xCF, 0x84],
    [0x86, 0xCF, 0xC8],
    [0xF2, 0xA6, 0xC6],
    [0xB5, 0xB8, 0x8F],
];
const LIGAND_CARBON: [u8; 3] = [0x3D, 0xD6, 0x8C];
const HELIX_COLOR: [u8; 3] = [0xE0, 0x5A, 0x9B];
const SHEET_COLOR: [u8; 3] = [0xF2, 0xC1, 0x4E];
const COIL_COLOR: [u8; 3] = [0xD8, 0xD8, 0xD8];

pub(crate) const BOND_RADIUS: f32 = 0.16;

/// Per-structure colour lookups shared by all representations.
pub(crate) struct Palette {
    chain_index: Vec<u32>,
    /// Shifts the chain palette so superposed entities are distinguishable.
    offset: usize,
}

impl Palette {
    pub fn new(structure: &Structure, offset: usize) -> Self {
        let mut chains: Vec<&str> = Vec::new();
        let chain_index = structure
            .residues
            .iter()
            .map(|r| {
                (match chains.iter().position(|c| *c == r.chain) {
                    Some(i) => i,
                    None => {
                        chains.push(&r.chain);
                        chains.len() - 1
                    }
                }) as u32
            })
            .collect();
        Self { chain_index, offset }
    }

    pub fn chain(&self, residue: usize) -> [u8; 3] {
        CHAIN_PALETTE[(self.chain_index[residue] as usize + self.offset) % CHAIN_PALETTE.len()]
    }

    pub fn residue_color(&self, structure: &Structure, residue: usize, coloring: Coloring) -> [u8; 4] {
        let rgb = match coloring {
            Coloring::Secondary => match structure.secondary.get(residue) {
                Some(SecondaryStructure::Helix) => HELIX_COLOR,
                Some(SecondaryStructure::Sheet) => SHEET_COLOR,
                _ => COIL_COLOR,
            },
            Coloring::Residue => CHAIN_PALETTE[(residue + self.offset) % CHAIN_PALETTE.len()],
            Coloring::Chain | Coloring::Element => self.chain(residue),
        };
        [rgb[0], rgb[1], rgb[2], 255]
    }

    pub fn atom_color(&self, structure: &Structure, atom: usize, coloring: Coloring) -> [u8; 4] {
        let a = &structure.atoms[atom];
        let residue = a.residue as usize;
        match coloring {
            Coloring::Element => {
                let rgb = if a.element == Element::C {
                    if structure.kind(atom) == AtomKind::Ligand {
                        LIGAND_CARBON
                    } else {
                        self.chain(residue)
                    }
                } else {
                    a.element.color().map(|c| (c * 255.0) as u8)
                };
                [rgb[0], rgb[1], rgb[2], 255]
            }
            _ => self.residue_color(structure, residue, coloring),
        }
    }
}

/// `palette_offset` rotates chain colours (use the entity index so that
/// superposed structures get different colours).
pub(crate) fn build(structure: &Structure, components: &[Component], selection: &[usize], palette_offset: usize) -> SceneData {
    let palette = Palette::new(structure, palette_offset);
    let mut scene = SceneData::default();
    let mut in_selection = vec![false; structure.atoms.len()];
    for &i in selection {
        if let Some(s) = in_selection.get_mut(i) {
            *s = true;
        }
    }
    for component in components {
        let selected: Vec<bool> = (0..structure.atoms.len())
            .map(|i| component.target.matches(structure.kind(i), in_selection[i]))
            .collect();
        if !selected.iter().any(|s| *s) {
            continue;
        }
        let color = |i: usize| palette.atom_color(structure, i, component.coloring);
        match component.representation {
            Representation::BallAndStick | Representation::Spacefill | Representation::Licorice => {
                atoms_and_bonds(structure, component.representation, &selected, &color, &mut scene);
            }
            Representation::Backbone => backbone(structure, &selected, &palette, component.coloring, &mut scene),
            Representation::Cartoon => {
                scene
                    .mesh
                    .append(cartoon::build(structure, &selected, &palette, component.coloring));
            }
            Representation::GaussianSurface => {
                scene.mesh.append(surface::gaussian(structure, &selected, &color));
            }
        }
    }
    scene
}

fn atoms_and_bonds(
    structure: &Structure,
    representation: Representation,
    selected: &[bool],
    color: &(dyn Fn(usize) -> [u8; 4] + Sync),
    scene: &mut SceneData,
) {
    for (i, atom) in structure.atoms.iter().enumerate() {
        if !selected[i] {
            continue;
        }
        let vdw = atom.element.vdw_radius();
        let radius = match representation {
            Representation::Spacefill => vdw,
            Representation::Licorice => BOND_RADIUS,
            _ => (vdw * 0.24).max(BOND_RADIUS * 1.3),
        };
        scene.spheres.push(SphereInstance {
            center: atom.position.to_array(),
            radius,
            color: color(i),
            id: i as u32,
        });
    }
    if representation == Representation::Spacefill {
        return;
    }
    for bond in &structure.bonds {
        let (a, b) = (bond.a as usize, bond.b as usize);
        if !(selected[a] && selected[b]) {
            continue;
        }
        scene.cylinders.push(CylinderInstance {
            p0: structure.atoms[a].position.to_array(),
            radius: BOND_RADIUS,
            p1: structure.atoms[b].position.to_array(),
            color0: color(a),
            color1: color(b),
            id0: a as u32,
            id1: b as u32,
        });
    }
}

fn backbone(
    structure: &Structure,
    selected: &[bool],
    palette: &Palette,
    coloring: Coloring,
    scene: &mut SceneData,
) {
    const RADIUS: f32 = 0.3;
    let mut prev: Option<(usize, Vec3, usize)> = None;
    for (ri, residue) in structure.residues.iter().enumerate() {
        let trace = structure
            .residue_atom(ri, "CA")
            .or_else(|| structure.residue_atom(ri, "P"))
            .filter(|&a| selected[a]);
        let Some(atom) = trace else {
            prev = None;
            continue;
        };
        let p = structure.atoms[atom].position;
        let color = palette.residue_color(structure, ri, coloring);
        scene.spheres.push(SphereInstance {
            center: p.to_array(),
            radius: RADIUS,
            color,
            id: atom as u32,
        });
        if let Some((pa, pp, pr)) = prev {
            let max = if structure.residue_atom(ri, "CA").is_some() { 4.3 } else { 8.0 };
            if structure.residues[pr].chain == residue.chain && pp.distance(p) < max {
                scene.cylinders.push(CylinderInstance {
                    p0: pp.to_array(),
                    radius: RADIUS,
                    p1: p.to_array(),
                    color0: palette.residue_color(structure, pr, coloring),
                    color1: color,
                    id0: pa as u32,
                    id1: atom as u32,
                });
            }
        }
        prev = Some((atom, p, ri));
    }
}
