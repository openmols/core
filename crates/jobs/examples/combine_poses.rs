//! Combine FRODOCK pose PDBs into the multi-model file openmols loads as
//! frames, then read it back to confirm the frames survive the round trip.
//!
//! cargo run -p jobs --example combine_poses -- <out.pdb> <pose.pdb>...
//!
//! With OPENMOLS_RECEPTOR=<receptor.pdb> the receptor is written into every
//! model, so each frame is the whole complex.

use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = args.next().context("usage: combine_poses <out.pdb> <pose.pdb>...")?;
    let poses: Vec<PathBuf> = args.map(PathBuf::from).collect();
    anyhow::ensure!(!poses.is_empty(), "give at least one pose");

    let receptor = std::env::var("OPENMOLS_RECEPTOR").ok().map(PathBuf::from);
    let text = jobs::frodock::combine_poses(&poses, receptor.as_deref())?;
    std::fs::write(&out, &text)?;
    println!("wrote {out}: {} models, {} bytes", text.matches("\nMODEL").count() + 1, text.len());

    let structure = molecule::Structure::load(&out)?;
    println!(
        "loaded back: {} atoms, {} residues, {} frames",
        structure.atoms.len(),
        structure.residues.len(),
        structure.frame_count()
    );
    anyhow::ensure!(
        structure.frame_count() == poses.len(),
        "expected {} frames, got {}",
        poses.len(),
        structure.frame_count()
    );
    Ok(())
}
