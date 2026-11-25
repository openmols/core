//! Ligand RMSD of docking poses against a reference, without superposing —
//! the measure that says whether a docked pose is near-native.
//!
//! cargo run -p jobs --example pose_rmsd -- <reference.pdb> <pose.pdb>...
//!
//! Atoms are paired by (chain, residue number, atom name), so bound and
//! unbound forms with slightly different atom counts still compare.

use anyhow::{Context, Result};
use std::collections::HashMap;

type Key = (String, isize, String);

fn atoms(path: &str) -> Result<HashMap<Key, glam::Vec3>> {
    let s = molecule::Structure::load(path).with_context(|| format!("loading {path}"))?;
    Ok(s.atoms
        .iter()
        .map(|a| {
            let r = &s.residues[a.residue as usize];
            ((r.chain.clone(), r.number, a.name.clone()), a.position)
        })
        .collect())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let reference = args.next().context("usage: pose_rmsd <reference.pdb> <pose.pdb>...")?;
    let native = atoms(&reference)?;
    println!("reference {reference}: {} atoms\n", native.len());

    let mut best: Option<(String, f32)> = None;
    for path in args {
        let pose = atoms(&path)?;
        let shared: Vec<(glam::Vec3, glam::Vec3)> = pose
            .iter()
            .filter_map(|(k, p)| native.get(k).map(|n| (*p, *n)))
            .collect();
        if shared.is_empty() {
            println!("{path}: no atoms in common with the reference");
            continue;
        }
        let sum: f32 = shared.iter().map(|(p, n)| p.distance_squared(*n)).sum();
        let rmsd = (sum / shared.len() as f32).sqrt();
        println!("{path:<28} RMSD {rmsd:>7.2} A over {} atoms", shared.len());
        if best.as_ref().is_none_or(|(_, b)| rmsd < *b) {
            best = Some((path, rmsd));
        }
    }
    if let Some((path, rmsd)) = best {
        println!("\nbest: {path} at {rmsd:.2} A");
    }
    Ok(())
}
