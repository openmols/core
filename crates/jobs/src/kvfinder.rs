//! pyKVFinder job: command line and results parsing.
//!
//! Inputs (written by the app into the job folder): `receptor.pdb` (polymer
//! only — hetero groups would fill the cavities) and optionally `ligand.pdb`
//! for ligand-adjusted detection. Outputs land in `out/`:
//! `receptor.KVFinder.results.toml` and `receptor.KVFinder.output.pdb`
//! (cavity grid points; residue name = cavity id, B-factor = depth).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const RESULTS_FILE: &str = "out/receptor.KVFinder.results.toml";
pub const CAVITY_PDB: &str = "out/receptor.KVFinder.output.pdb";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    /// Probe-out radius, Å (bigger finds shallower pockets).
    pub probe_out: f32,
    /// Removal distance, Å.
    pub removal_distance: f32,
    /// Minimum cavity volume, Å³.
    pub volume_cutoff: f32,
    /// Restrict to the region around `ligand.pdb`.
    pub ligand_adjust: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            probe_out: 4.0,
            removal_distance: 2.4,
            volume_cutoff: 5.0,
            ligand_adjust: false,
        }
    }
}

pub fn command(settings: &Settings) -> String {
    let mut cmd = format!(
        "pyKVFinder receptor.pdb -D --hydropathy -O out -o {} -R {} -V {}",
        settings.probe_out, settings.removal_distance, settings.volume_cutoff
    );
    if settings.ligand_adjust {
        cmd.push_str(" -L ligand.pdb");
    }
    cmd
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LiningResidue {
    pub number: String,
    pub chain: String,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cavity {
    /// `KAA`, `KAB`, … (also the residue name in the cavity PDB).
    pub id: String,
    pub volume: f64,
    pub area: f64,
    pub max_depth: Option<f64>,
    pub avg_depth: Option<f64>,
    pub avg_hydropathy: Option<f64>,
    pub residues: Vec<LiningResidue>,
}

impl Cavity {
    pub fn residue_labels(&self) -> String {
        self.residues
            .iter()
            .map(|r| format!("{}:{} {}", r.chain, r.name, r.number))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Deserialize)]
struct ResultsFile {
    #[serde(rename = "RESULTS")]
    results: Results,
}

#[derive(Deserialize)]
struct Results {
    #[serde(rename = "VOLUME", default)]
    volume: BTreeMap<String, f64>,
    #[serde(rename = "AREA", default)]
    area: BTreeMap<String, f64>,
    #[serde(rename = "MAX_DEPTH", default)]
    max_depth: BTreeMap<String, f64>,
    #[serde(rename = "AVG_DEPTH", default)]
    avg_depth: BTreeMap<String, f64>,
    /// Also contains the scale range (an array), so parse values loosely.
    #[serde(rename = "AVG_HYDROPATHY", default)]
    avg_hydropathy: BTreeMap<String, toml::Value>,
    #[serde(rename = "RESIDUES", default)]
    residues: BTreeMap<String, Vec<Vec<String>>>,
}

/// Parse `results.toml`; cavities sorted by volume, largest first.
pub fn parse_results(text: &str) -> Result<Vec<Cavity>> {
    let file: ResultsFile = toml::from_str(text).context("parsing pyKVFinder results")?;
    let r = file.results;
    let mut cavities: Vec<Cavity> = r
        .volume
        .iter()
        .map(|(id, volume)| Cavity {
            id: id.clone(),
            volume: *volume,
            area: r.area.get(id).copied().unwrap_or(0.0),
            max_depth: r.max_depth.get(id).copied(),
            avg_depth: r.avg_depth.get(id).copied(),
            avg_hydropathy: r.avg_hydropathy.get(id).and_then(|v| v.as_float()),
            residues: r
                .residues
                .get(id)
                .map(|list| {
                    list.iter()
                        .filter(|t| t.len() >= 3)
                        .map(|t| LiningResidue {
                            number: t[0].clone(),
                            chain: t[1].clone(),
                            name: t[2].clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect();
    cavities.sort_by(|a, b| b.volume.total_cmp(&a.volume));
    Ok(cavities)
}

pub fn read_results(job_dir: &Path) -> Result<Vec<Cavity>> {
    let path = job_dir.join(RESULTS_FILE);
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_results(&text)
}

pub fn cavity_pdb(job_dir: &Path) -> PathBuf {
    job_dir.join(CAVITY_PDB)
}

/// The cavity PDB normalised for strict readers.
///
/// pyKVFinder writes per-point hydropathy into the occupancy column (can be
/// negative) and stops at column 66, so lines lack the element column.
/// Occupancy becomes 1.00, depth stays in the B-factor column and the element
/// (H) is added in columns 77-78.
pub fn read_cavity_pdb(job_dir: &Path) -> Result<String> {
    let path = cavity_pdb(job_dir);
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sanitize_cavity_pdb(&text))
}

pub fn sanitize_cavity_pdb(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 5 / 4);
    // pyKVFinder gives every cavity the same residue number; renumber so each
    // cavity (KAA, KAB, …) is its own residue.
    let mut ids: Vec<String> = Vec::new();
    for line in text.lines() {
        if (line.starts_with("ATOM") || line.starts_with("HETATM")) && line.len() >= 54 {
            let id = line[17..20].to_string();
            let number = match ids.iter().position(|x| *x == id) {
                Some(i) => i + 1,
                None => {
                    ids.push(id);
                    ids.len()
                }
            };
            let bfactor = line.get(60..66).unwrap_or("  0.00");
            out.push_str(&format!(
                "{}{:>4}{}  1.00{:>6}          {:>2}\n",
                &line[..22],
                number % 10_000,
                &line[26..54],
                bfactor.trim(),
                "H"
            ));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_1hsg_results() {
        let text = include_str!("../tests/fixtures/receptor.KVFinder.results.toml");
        let cavities = parse_results(text).unwrap();
        assert_eq!(cavities.len(), 2);
        let kaa = &cavities[0];
        assert_eq!(kaa.id, "KAA");
        assert!((kaa.volume - 689.69).abs() < 1e-6);
        assert_eq!(kaa.max_depth, Some(6.63));
        assert_eq!(kaa.avg_hydropathy, Some(-0.52));
        // The HIV protease active site: catalytic Asp25 of both chains.
        assert!(kaa.residues.iter().any(|r| r.chain == "A" && r.number == "25" && r.name == "ASP"));
        assert!(kaa.residues.iter().any(|r| r.chain == "B" && r.number == "25" && r.name == "ASP"));
    }

    #[test]
    fn sanitizes_cavity_lines() {
        let line = "ATOM      2  HA  KAA   259       8.830  20.686   1.811 -0.12  1.80";
        let fixed = sanitize_cavity_pdb(line);
        let fixed = fixed.trim_end_matches('\n');
        assert_eq!(&fixed[..22], &line[..22]);
        assert_eq!(fixed[22..26].trim(), "1");
        assert_eq!(&fixed[26..54], &line[26..54]);
        assert_eq!(&fixed[54..60], "  1.00");
        assert_eq!(fixed[60..66].trim(), "1.80");
        assert_eq!(fixed.len(), 78);
        assert_eq!(&fixed[76..78], " H");

        let two = "ATOM      1  HA  KAA   259       8.830  20.686   1.811 -0.12  1.80
ATOM      2  H   KAB   259       1.000   2.000   3.000  0.50  0.00";
        let fixed = sanitize_cavity_pdb(two);
        let numbers: Vec<&str> = fixed.lines().map(|l| l[22..26].trim()).collect();
        assert_eq!(numbers, ["1", "2"]);
    }

    #[test]
    fn builds_command() {
        let cmd = command(&Settings { ligand_adjust: true, ..Default::default() });
        assert!(cmd.starts_with("pyKVFinder receptor.pdb -D --hydropathy -O out"));
        assert!(cmd.ends_with("-L ligand.pdb"));
    }
}
