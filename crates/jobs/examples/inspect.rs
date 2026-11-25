//! Load a structure and report what openmols sees in it.
//! cargo run -p jobs --example inspect --release -- <file>
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: inspect <file>")?;
    let t = std::time::Instant::now();
    let s = molecule::Structure::load(&path)?;
    println!(
        "{path}\n  {} atoms, {} residues, {} chains, {} bonds, {} frames  ({:.1}s)",
        s.atoms.len(),
        s.residues.len(),
        s.chain_count(),
        s.bonds.len(),
        s.frame_count(),
        t.elapsed().as_secs_f64()
    );
    let (c, r) = s.bounding_sphere();
    println!("  centre {c:.1?}, radius {r:.1} A");
    Ok(())
}
