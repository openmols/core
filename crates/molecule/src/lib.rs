//! Molecular data model shared by the renderer and the application.
//!
//! A [`Structure`] is a flat list of atoms plus residue/chain metadata and a
//! bond list. It is deliberately simple and engine-agnostic so that docking
//! (Vina) and MD (OpenMM/GROMACS) wrappers can produce and consume it later.

mod bonds;
pub mod element;
mod pdb;
mod sdf;
mod write;
pub mod align;
pub mod secondary;

use anyhow::{Context, Result, bail};
use glam::Vec3;
use std::path::Path;

pub use bonds::perceive_bonds;
pub use element::Element;
pub use secondary::SecondaryStructure;
pub use write::write_pdb;

#[derive(Clone, Debug)]
pub struct Atom {
    /// Atom name as written in the file, e.g. `CA`, `O1`.
    pub name: String,
    pub element: Element,
    /// Position in Ångström.
    pub position: Vec3,
    /// Index into [`Structure::residues`].
    pub residue: u32,
    /// `HETATM` record (ligands, waters, ions) as opposed to polymer `ATOM`.
    pub hetero: bool,
    pub serial: usize,
    /// Temperature factor (or any per-atom property stored in that column).
    pub b_factor: f32,
}

#[derive(Clone, Debug)]
pub struct Residue {
    pub name: String,
    pub number: isize,
    pub insertion_code: Option<String>,
    pub chain: String,
    /// Indices of this residue's atoms in [`Structure::atoms`] (contiguous).
    pub atoms: std::ops::Range<u32>,
}

impl Residue {
    pub fn is_water(&self) -> bool {
        matches!(self.name.as_str(), "HOH" | "WAT" | "H2O" | "DOD")
    }
}

/// Coarse classification used to target representations, like Mol* presets.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AtomKind {
    /// Protein / nucleic acid chains (`ATOM` records).
    Polymer,
    /// Hetero groups other than water: ligands, cofactors, ions.
    Ligand,
    Water,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bond {
    pub a: u32,
    pub b: u32,
    /// 1 = single, 2 = double, 3 = triple, 4 = aromatic (SDF convention).
    pub order: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Structure {
    pub name: String,
    pub atoms: Vec<Atom>,
    pub residues: Vec<Residue>,
    pub bonds: Vec<Bond>,
    /// Per-residue secondary structure (same length as `residues`).
    pub secondary: Vec<SecondaryStructure>,
    /// Coordinate sets sharing this topology (ensemble models, trajectory
    /// frames, docking poses). Empty for a single-frame structure; otherwise
    /// `atoms[].position` holds the frame selected with [`Structure::set_frame`].
    pub frames: Vec<Vec<Vec3>>,
}

impl Structure {
    /// Load a structure, choosing the parser from the file extension.
    /// Supported: `.pdb`, `.ent`, `.cif`, `.mmcif` (optionally `.gz`), `.sdf`, `.mol`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let lower = path.to_string_lossy().to_lowercase();
        let lower = lower.strip_suffix(".gz").unwrap_or(&lower);
        let mut structure = if lower.ends_with(".sdf") || lower.ends_with(".mol") {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            sdf::parse(&text)?
        } else if [".pdb", ".ent", ".cif", ".mmcif"]
            .iter()
            .any(|ext| lower.ends_with(ext))
        {
            pdb::load(path)?
        } else {
            bail!("unsupported file type: {}", path.display());
        };
        structure.secondary = secondary::assign(&structure);
        if structure.name.is_empty() {
            structure.name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
        }
        Ok(structure)
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len().max(1)
    }

    /// Copy frame `index` into the atom positions. Out-of-range indices are clamped.
    pub fn set_frame(&mut self, index: usize) {
        if self.frames.is_empty() {
            return;
        }
        let frame = &self.frames[index.min(self.frames.len() - 1)];
        for (atom, p) in self.atoms.iter_mut().zip(frame) {
            atom.position = *p;
        }
    }

    /// Axis-aligned bounding box `(min, max)`; `None` for an empty structure.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let first = self.atoms.first()?.position;
        Some(self.atoms.iter().fold((first, first), |(lo, hi), a| {
            (lo.min(a.position), hi.max(a.position))
        }))
    }

    /// Centre of the bounding box and radius of the enclosing sphere around it.
    /// Waters are ignored (unless there is nothing else) so stray solvent
    /// molecules don't shrink the view of the protein.
    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        let non_water: Vec<Vec3> = self
            .atoms
            .iter()
            .filter(|a| !self.residues[a.residue as usize].is_water())
            .map(|a| a.position)
            .collect();
        let points = if non_water.is_empty() {
            self.atoms.iter().map(|a| a.position).collect()
        } else {
            non_water
        };
        let Some(&first) = points.first() else {
            return (Vec3::ZERO, 1.0);
        };
        let (lo, hi) = points
            .iter()
            .fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let center = (lo + hi) * 0.5;
        let radius = points.iter().map(|p| p.distance(center)).fold(0.0f32, f32::max);
        (center, radius.max(1.0))
    }

    pub fn kind(&self, atom: usize) -> AtomKind {
        let a = &self.atoms[atom];
        if self.residues[a.residue as usize].is_water() {
            AtomKind::Water
        } else if a.hetero {
            AtomKind::Ligand
        } else {
            AtomKind::Polymer
        }
    }

    /// Find an atom of a residue by name (e.g. `"CA"`).
    pub fn residue_atom(&self, residue: usize, name: &str) -> Option<usize> {
        let r = &self.residues[residue];
        (r.atoms.start..r.atoms.end)
            .map(|i| i as usize)
            .find(|&i| self.atoms[i].name == name)
    }

    pub fn residue_of(&self, atom: usize) -> &Residue {
        &self.residues[self.atoms[atom].residue as usize]
    }

    /// Human-readable label such as `A/ASP 25/OD1 (O)`.
    pub fn describe_atom(&self, atom: usize) -> String {
        let a = &self.atoms[atom];
        let r = self.residue_of(atom);
        let ins = r.insertion_code.as_deref().unwrap_or("");
        let chain = if r.chain.is_empty() { String::new() } else { format!("{}/", r.chain) };
        format!(
            "{chain}{} {}{ins}/{} ({}) #{}",
            r.name,
            r.number,
            a.name,
            a.element.symbol(),
            a.serial
        )
    }

    pub fn chain_count(&self) -> usize {
        let mut chains: Vec<&str> = self.residues.iter().map(|r| r.chain.as_str()).collect();
        chains.sort_unstable();
        chains.dedup();
        chains.len()
    }
}
