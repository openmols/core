//! PDB / mmCIF loading through `pdbtbx`.

use crate::{Atom, Element, Residue, Structure, perceive_bonds};
use anyhow::{Result, anyhow};
use glam::Vec3;
use pdbtbx::{ReadOptions, StrictnessLevel};
use std::path::Path;

pub fn load(path: &Path) -> Result<Structure> {
    let path_str = path.to_string_lossy();
    // pdbtbx can panic on unusual input (e.g. negative occupancies); never let
    // a bad file take the application down.
    let read = std::panic::catch_unwind(|| {
        ReadOptions::new()
            .guess_format(&path_str)
            .set_level(StrictnessLevel::Loose)
            .read(&path_str)
    })
    .map_err(|panic| {
        let reason = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "parser panicked".into());
        anyhow!("failed to read {}: {reason}", path.display())
    })?;
    let (pdb, _warnings) = read.map_err(|errors| {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        anyhow!("failed to read {}:\n{msg}", path.display())
    })?;

    let mut structure = Structure {
        name: pdb.identifier.clone().unwrap_or_default(),
        ..Default::default()
    };

    let Some(model) = pdb.model(0) else {
        return Ok(structure);
    };

    // Traversal shared by the topology pass and the extra-model pass, so atom
    // order is identical across models.
    fn model_atoms(model: &pdbtbx::Model) -> impl Iterator<Item = &pdbtbx::Atom> {
        model.chains().flat_map(|chain| {
            chain
                .residues()
                .filter_map(|r| r.conformers().next())
                .flat_map(|c| c.atoms())
        })
    }

    for chain in model.chains() {
        for residue in chain.residues() {
            // Keep only the first alternate conformer of each residue.
            let Some(conformer) = residue.conformers().next() else {
                continue;
            };
            let (number, insertion_code) = residue.id();
            let residue_index = structure.residues.len() as u32;
            let first_atom = structure.atoms.len() as u32;
            structure.residues.push(Residue {
                name: conformer.name().to_string(),
                number,
                insertion_code: insertion_code.map(str::to_string),
                chain: chain.id().to_string(),
                atoms: first_atom..first_atom,
            });
            for atom in conformer.atoms() {
                let (x, y, z) = atom.pos();
                let element = atom
                    .element()
                    .map(|e| Element::from_symbol(e.symbol()))
                    .filter(|e| *e != Element::UNKNOWN)
                    .unwrap_or_else(|| Element::guess_from_atom_name(atom.name()));
                structure.atoms.push(Atom {
                    name: atom.name().to_string(),
                    element,
                    position: Vec3::new(x as f32, y as f32, z as f32),
                    residue: residue_index,
                    hetero: atom.hetero(),
                    serial: atom.serial_number(),
                    b_factor: atom.b_factor() as f32,
                });
            }
            structure.residues[residue_index as usize].atoms.end = structure.atoms.len() as u32;
        }
    }

    // Further models (NMR ensembles, multi-model trajectories) become frames
    // when they have the same atoms as the first model.
    if pdb.model_count() > 1 {
        let mut frames = vec![structure.atoms.iter().map(|a| a.position).collect::<Vec<_>>()];
        for extra in pdb.models().skip(1) {
            let positions: Vec<Vec3> = model_atoms(extra)
                .map(|a| {
                    let (x, y, z) = a.pos();
                    Vec3::new(x as f32, y as f32, z as f32)
                })
                .collect();
            if positions.len() == structure.atoms.len() {
                frames.push(positions);
            }
        }
        if frames.len() > 1 {
            structure.frames = frames;
        }
    }

    structure.bonds = perceive_bonds(&structure.atoms);
    Ok(structure)
}
