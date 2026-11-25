//! Re-filter a finished ensemble pocket job, outside the app.
//!
//! cargo run -p jobs --example ensemble -- <job-dir> [filters-file]
//!
//! `<job-dir>` holds `out/f00000/frame.KVFinder.results.toml`, as produced by
//! `kvfinder::ensemble_command`. Without a filters file the paper's defaults
//! are used; `-` reads the filters from stdin. The filter text is the same
//! one rule per line as the Ensemble pockets panel:
//!
//! ```text
//! residues >= 6
//! hydropathy <= 0
//! area 80 .. 2480
//! volume >= 120
//! overlap 0.6
//! ```
//!
//! With `OPENMOLS_ENSEMBLE=<file.pdb>` the per-residue RMSF of that ensemble
//! is computed and `rmsf` becomes available as a filter metric.

use anyhow::{Context, Result};
use jobs::kvfinder::{self, EnsembleFilter};
use std::io::Read;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: ensemble <job-dir> [filters]")?);

    let filter = match args.next() {
        None => EnsembleFilter::default(),
        Some(source) => {
            let text = if source == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                std::fs::read_to_string(&source)?
            };
            let (filter, problems) = EnsembleFilter::parse_text(&text);
            for p in &problems {
                eprintln!("warning: {p}");
            }
            filter
        }
    };

    let t = Instant::now();
    let (found, frames) = kvfinder::read_ensemble(&dir)?;
    println!(
        "read {} cavities from {frames} frames in {:.2?}\n",
        found.len(),
        t.elapsed()
    );

    let properties = residue_properties()?;
    let t = Instant::now();
    let stats = kvfinder::ensemble_stats(&found, frames, &filter, &properties);
    let stats_time = t.elapsed();

    println!(
        "{} cavities · {} of {} frames held one · up to {} per frame · {:.1} per frame on average",
        stats.cavities, stats.frames_with_cavities, stats.frames, stats.max_per_frame, stats.mean_per_frame
    );
    println!("\n{:<12} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "metric", "n", "min", "p10", "median", "p90", "max", "missing");
    for m in &stats.metrics {
        println!(
            "{:<12} {:>7} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8}",
            m.name, m.n, m.min, m.percentiles[0], m.percentiles[2], m.percentiles[4], m.max, m.missing
        );
    }

    println!("\nfilters: {}", filter.describe());
    for c in &stats.criteria {
        println!(
            "  {:<24} keeps {:>6} of {} ({:.0}%)",
            c.rule,
            c.passing,
            stats.cavities,
            100.0 * c.passing as f64 / stats.cavities.max(1) as f64
        );
    }
    println!(
        "  {:<24} keeps {:>6} of {} ({:.0}%)   [stats in {stats_time:.2?}]",
        "all together",
        stats.passing_all,
        stats.cavities,
        100.0 * stats.passing_all as f64 / stats.cavities.max(1) as f64
    );

    let t = Instant::now();
    let pockets = kvfinder::consensus_pockets_with(&found, frames, &filter, &properties);
    println!(
        "\n-> {} consensus pockets (overlap {}) in {:.2?}\n",
        pockets.len(),
        filter.residue_overlap,
        t.elapsed()
    );
    for p in pockets.iter().take(15) {
        println!(
            "#{:<3} persistence {:>4.0}% ({:>4}/{frames} frames)  max {:>6.0} A^3  mean {:>6.0} A^3  hyd {:>6.2}  best frame {}",
            p.rank,
            p.persistence * 100.0,
            p.frames.len(),
            p.max_volume,
            p.mean_volume,
            p.mean_hydropathy.unwrap_or(0.0),
            p.best_frame,
        );
        println!("     lining: {}", p.residue_labels(12));
    }
    Ok(())
}

/// Per-residue RMSF of the ensemble named by `OPENMOLS_ENSEMBLE`, so the
/// MD-derived filter path can be exercised from the command line.
fn residue_properties() -> Result<kvfinder::ResidueProperties> {
    let mut properties = kvfinder::ResidueProperties::new();
    let Ok(path) = std::env::var("OPENMOLS_ENSEMBLE") else {
        return Ok(properties);
    };
    let mut structure = molecule::Structure::load(&path)?;
    fit_frames(&mut structure);
    let rmsf = molecule::analysis::residue_rmsf(&structure);
    let values: std::collections::HashMap<(String, String), f64> = structure
        .residues
        .iter()
        .zip(&rmsf)
        .map(|(r, v)| ((r.chain.clone(), r.number.to_string()), *v as f64))
        .collect();
    println!(
        "rmsf from {} frames over {} residues (max {:.2} A)",
        structure.frame_count(),
        values.len(),
        rmsf.iter().copied().fold(0.0f32, f32::max)
    );
    properties.insert("rmsf".into(), values);
    Ok(properties)
}

/// Superpose every frame onto the first on Cα atoms, so RMSF measures
/// internal motion rather than tumbling.
fn fit_frames(structure: &mut molecule::Structure) {
    let ca: Vec<usize> = (0..structure.residues.len())
        .filter_map(|r| structure.residue_atom(r, "CA"))
        .collect();
    if ca.len() < 3 || structure.frames.len() < 2 {
        return;
    }
    let target: Vec<glam::Vec3> = ca.iter().map(|&a| structure.frames[0][a]).collect();
    for frame in &mut structure.frames {
        let mobile: Vec<glam::Vec3> = ca.iter().map(|&a| frame[a]).collect();
        let (transform, _) = molecule::align::superpose(&mobile, &target);
        for p in frame.iter_mut() {
            *p = transform.transform_point3(*p);
        }
    }
}
