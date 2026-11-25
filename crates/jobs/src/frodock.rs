//! FRODOCK job: the docking pipeline and solution parsing.
//!
//! The app writes `receptor.pdb` and `ligand.pdb` (both already posed) into the
//! job folder. The container then runs FRODOCK's documented four steps:
//! potential maps with `frodockgrid`, the FFT search with `frodock`, clustering
//! with `frodockcluster`, and `frodockview -p` to write the top poses as PDBs
//! named `ligand_1.pdb`, `ligand_2.pdb`, …
//!
//! See the FRODOCK user guide (chaconlab/FRODOCK) and Garzón et al. 2009,
//! Ramírez-Aportela et al. 2016.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const RECEPTOR: &str = "receptor.pdb";
pub const LIGAND: &str = "ligand.pdb";
pub const SOLUTIONS_FILE: &str = "poses.txt";
pub const RESTRAINTS_FILE: &str = "restraints.txt";

/// Electrostatic weighting profile (`frodockgrid -t`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Interaction {
    /// Antibody / antigen.
    Antibody,
    /// Enzyme / inhibitor or enzyme / substrate.
    Enzyme,
    #[default]
    Other,
}

impl Interaction {
    fn flag(self) -> &'static str {
        match self {
            Interaction::Antibody => "A",
            Interaction::Enzyme => "E",
            Interaction::Other => "O",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Interaction::Antibody => "Antibody/antigen",
            Interaction::Enzyme => "Enzyme/inhibitor",
            Interaction::Other => "Other",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub interaction: Interaction,
    /// Number of clustered solutions to keep and write as poses.
    pub top_n: usize,
    /// RMSD between clusters, Å (`frodockcluster -d`).
    pub cluster_rmsd: f32,
    /// Local docking: restrict the search around this point (`frodock --around`).
    pub around: Option<[f32; 3]>,
    /// Residue-pair distance restraints, filtered after docking.
    #[serde(default)]
    pub restraints: Vec<Restraint>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            interaction: Interaction::Other,
            top_n: 10,
            cluster_rmsd: 5.0,
            around: None,
            restraints: Vec::new(),
        }
    }
}

/// One residue in a restraint: residue name, chain and number.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RestraintResidue {
    pub name: String,
    pub chain: String,
    pub number: i32,
}

/// "At least one atom pair of these two residues must be within `distance` Å."
/// All restraints must hold; hydrogens are excluded. Filtering runs after
/// docking, via `frodockonstraints`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Restraint {
    pub receptor: RestraintResidue,
    pub ligand: RestraintResidue,
    pub distance: f32,
}

/// FRODOCK's constraints file, in the column layout its own `consts.txt`
/// examples use: `RESNAME CHAIN RESNUM   RESNAME CHAIN RESNUM   DIST`.
pub fn restraints_file(restraints: &[Restraint]) -> String {
    let mut out = String::from(
        "# RECEPT_____     LIGAND_____ D__\n# -------------------------------\n",
    );
    for r in restraints {
        out.push_str(&format!(
            "  {:<3} {:<1} {:>4}      {:<3} {:<1} {:>5}  {:.0}\n",
            r.receptor.name,
            r.receptor.chain,
            r.receptor.number,
            r.ligand.name,
            r.ligand.chain,
            r.ligand.number,
            r.distance,
        ));
    }
    out
}

/// The whole pipeline as one `sh -c` script.
pub fn command(settings: &Settings) -> String {
    let top = settings.top_n.max(1);
    let t = settings.interaction.flag();
    // -m 3 also writes receptor_ASA.pdb / ligand_ASA.pdb, which frodock reads.
    // The SOAP potential ships with FRODOCK; the image exports its path.
    let mut frodock = format!(
        "frodock receptor_ASA.pdb ligand_ASA.pdb -w rec_W.mrc -e rec_E.mrc --th 10 \
         -d rec_DS.mrc,lig_DS.mrc -t {t} -s \"$FRODOCK_SOAP\" -o dock.dat"
    );
    if let Some([x, y, z]) = settings.around {
        frodock.push_str(&format!(" --around {x:.2},{y:.2},{z:.2}"));
    }

    // The FRODOCK tools exit non-zero even when they succeed, so every step
    // is judged by whether it produced its output file, not by its status.
    let mut steps = vec![
        "fail() { echo \"FRODOCK step failed: $1\" >&2; exit 1; }".to_string(),
        "need() { [ -s \"$1\" ] || fail \"$2 (expected $1)\"; }".to_string(),
        format!("frodockgrid {RECEPTOR} -o rec_W.mrc || true"),
        "need rec_W.mrc 'van der Waals map'".to_string(),
        format!("frodockgrid {RECEPTOR} -o rec_E.mrc -m 1 -t {t} || true"),
        "need rec_E.mrc 'electrostatic map'".to_string(),
        format!("frodockgrid {RECEPTOR} -o rec_DS.mrc -m 3 || true"),
        "need rec_DS.mrc 'receptor desolvation map'".to_string(),
        "need receptor_ASA.pdb 'receptor ASA'".to_string(),
        format!("frodockgrid {LIGAND} -o lig_DS.mrc -m 3 || true"),
        "need lig_DS.mrc 'ligand desolvation map'".to_string(),
        "need ligand_ASA.pdb 'ligand ASA'".to_string(),
        format!("{frodock} || true"),
        "need dock.dat 'docking search'".to_string(),
    ];
    // Restraints filter FRODOCK's solutions before clustering.
    let to_cluster = if settings.restraints.is_empty() {
        "dock.dat"
    } else {
        steps.push(format!(
            "frodockonstraints dock.dat {RECEPTOR} {LIGAND} {RESTRAINTS_FILE} -o dock_C.dat || true"
        ));
        steps.push("need dock_C.dat 'restraint filtering'".to_string());
        "dock_C.dat"
    };
    steps.push(format!(
        "frodockcluster {to_cluster} {LIGAND} --nc {top} -d {} -o clust.dat || true",
        settings.cluster_rmsd
    ));
    steps.push("need clust.dat 'clustering'".to_string());
    steps.push(format!(
        "frodockview clust.dat -r 1-{top} -p {LIGAND} > {SOLUTIONS_FILE} 2>&1 || true"
    ));
    steps.push(format!("need {SOLUTIONS_FILE} 'pose generation'"));
    steps.push("cat poses.txt".to_string());
    steps.join("\n")
}

/// One docking solution from `frodockview`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Solution {
    pub rank: usize,
    /// ZYZ Euler angles, degrees.
    pub euler: [f32; 3],
    /// Ligand centre of mass, Å.
    pub position: [f32; 3],
    /// FRODOCK correlation score (higher is better).
    pub score: f32,
    /// Pose file written by `frodockview -p`, if any.
    pub pose_file: Option<String>,
}

/// Parse `frodockview` output: `rank e1 e2 e3 x y z score [=> file.pdb]`.
/// Log lines that don't match this shape are ignored.
pub fn parse_solutions(text: &str) -> Vec<Solution> {
    let mut out = Vec::new();
    for line in text.lines() {
        let (numbers, pose_file) = match line.split_once("=>") {
            Some((n, f)) => (n, Some(f.trim().to_string())),
            None => (line, None),
        };
        let fields: Vec<&str> = numbers.split_whitespace().collect();
        if fields.len() < 8 {
            continue;
        }
        let Ok(rank) = fields[0].parse::<usize>() else {
            continue;
        };
        let values: Option<Vec<f32>> = fields[1..8].iter().map(|f| f.parse::<f32>().ok()).collect();
        let Some(v) = values else { continue };
        out.push(Solution {
            rank,
            euler: [v[0], v[1], v[2]],
            position: [v[3], v[4], v[5]],
            score: v[6],
            pose_file,
        });
    }
    out.sort_by_key(|s| s.rank);
    out
}

pub fn read_solutions(job_dir: &Path) -> Result<Vec<Solution>> {
    let path = job_dir.join(SOLUTIONS_FILE);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let solutions = parse_solutions(&text);
    anyhow::ensure!(
        !solutions.is_empty(),
        "FRODOCK produced no solutions (see {})",
        path.display()
    );
    Ok(solutions)
}

/// Pose PDBs in rank order, skipping any that weren't written.
pub fn pose_paths(job_dir: &Path, solutions: &[Solution]) -> Vec<PathBuf> {
    solutions
        .iter()
        .map(|s| match &s.pose_file {
            Some(name) => job_dir.join(name),
            None => job_dir.join(format!("ligand_{}.pdb", s.rank)),
        })
        .filter(|p| p.is_file())
        .collect()
}

/// Combine pose PDBs into one multi-model PDB, so the poses load as frames of
/// a single entity and can be scrubbed through with the frame slider.
///
/// FRODOCK's pose files hold only the body it moved; the receptor stays where
/// it was and is not written. With `receptor`, it is included at the front of
/// every model, so each frame is the whole complex and the predicted
/// interface can be seen. Its chains are relabelled to ids the poses do not
/// use, so the two never merge.
pub fn combine_poses(paths: &[PathBuf], receptor: Option<&Path>) -> Result<String> {
    let atom = |line: &str| line.starts_with("ATOM") || line.starts_with("HETATM");
    let keep = |line: &str| atom(line) || line.starts_with("TER");
    let chain_of = |line: &str| line.as_bytes().get(21).copied().map(char::from);

    // Chains the poses occupy; the receptor has to avoid them.
    let mut taken: Vec<char> = Vec::new();
    if let Some(first) = paths.first() {
        let text = std::fs::read_to_string(first)
            .with_context(|| format!("reading {}", first.display()))?;
        for line in text.lines().filter(|l| atom(l)) {
            if let Some(c) = chain_of(line)
                && !taken.contains(&c)
            {
                taken.push(c);
            }
        }
    }

    // The receptor, with its chains remapped into the free ids.
    let mut receptor_lines = String::new();
    if let Some(path) = receptor {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut map: Vec<(char, char)> = Vec::new();
        for line in text.lines() {
            if !keep(line) {
                continue;
            }
            let mut bytes = line.as_bytes().to_vec();
            if let Some(original) = chain_of(line) {
                let replacement = match map.iter().find(|(from, _)| *from == original) {
                    Some((_, to)) => *to,
                    None => {
                        let free = ('A'..='Z')
                            .chain('a'..='z')
                            .chain('0'..='9')
                            .find(|c| !taken.contains(c))
                            .unwrap_or(original);
                        taken.push(free);
                        map.push((original, free));
                        free
                    }
                };
                if bytes.len() > 21 {
                    bytes[21] = replacement as u8;
                }
            }
            receptor_lines.push_str(&String::from_utf8_lossy(&bytes));
            receptor_lines.push('\n');
        }
        if !receptor_lines.is_empty() && !receptor_lines.ends_with("TER\n") {
            receptor_lines.push_str("TER\n");
        }
    }

    let mut out = String::new();
    for (i, path) in paths.iter().enumerate() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        out.push_str(&format!("MODEL     {:>4}\n", i + 1));
        out.push_str(&receptor_lines);
        for line in text.lines().filter(|l| keep(l)) {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("ENDMDL\n");
    }
    out.push_str("END\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUTPUT: &str = "\
Reading clust.dat
 1  86.00  158.32  201.22    19.36    6.83   55.42    1359.015259 => ligand_1.pdb
 2  12.00   42.10   99.00   -5.00   12.30   -4.20     981.500000 => ligand_2.pdb
Done.
";

    #[test]
    fn parses_frodockview_output() {
        let s = parse_solutions(OUTPUT);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].rank, 1);
        assert_eq!(s[0].euler, [86.0, 158.32, 201.22]);
        assert_eq!(s[0].position, [19.36, 6.83, 55.42]);
        assert!((s[0].score - 1359.0153).abs() < 1e-3);
        assert_eq!(s[0].pose_file.as_deref(), Some("ligand_1.pdb"));
        assert_eq!(s[1].rank, 2);
    }

    #[test]
    fn builds_the_pipeline() {
        let cmd = command(&Settings::default());
        assert!(cmd.contains("frodockgrid receptor.pdb -o rec_E.mrc -m 1 -t O"));
        assert!(cmd.contains("frodock receptor_ASA.pdb ligand_ASA.pdb"));
        assert!(cmd.contains("-t O"));
        assert!(cmd.contains("-s \"$FRODOCK_SOAP\""));
        assert!(cmd.contains("--nc 10"));
        // Steps are checked by their outputs, not exit status.
        assert!(cmd.contains("need dock.dat"));
        assert!(!cmd.contains("set -e"));
        assert!(!cmd.contains("--around"));
        assert!(!cmd.contains("frodockonstraints"));

        let local = command(&Settings {
            interaction: Interaction::Enzyme,
            around: Some([1.0, 2.0, 3.5]),
            restraints: vec![Restraint {
                receptor: RestraintResidue { name: "GLY".into(), chain: "A".into(), number: 269 },
                ligand: RestraintResidue { name: "SER".into(), chain: "A".into(), number: 81 },
                distance: 5.0,
            }],
            ..Default::default()
        });
        assert!(local.contains("-t E"));
        assert!(local.contains("--around 1.00,2.00,3.50"));
        assert!(local.contains("frodockonstraints dock.dat"));
        assert!(local.contains("frodockcluster dock_C.dat"));
        assert!(local.contains("need dock_C.dat"));
    }

    #[test]
    fn combines_poses_into_models() {
        let dir = std::env::temp_dir().join("openmols-frodock-poses");
        std::fs::create_dir_all(&dir).unwrap();
        let mut paths = Vec::new();
        for i in 1..=2 {
            let p = dir.join(format!("ligand_{i}.pdb"));
            std::fs::write(
                &p,
                format!(
                    "REMARK generated\nATOM      1  CA  ALA A   1       {i}.000   0.000   0.000  1.00  0.00           C\nTER\nEND\n"
                ),
            )
            .unwrap();
            paths.push(p);
        }
        let text = combine_poses(&paths, None).unwrap();
        assert_eq!(text.matches("MODEL").count(), 2);
        assert_eq!(text.matches("ENDMDL").count(), 2);
        assert!(!text.contains("REMARK"));
        assert_eq!(text.matches("ATOM").count(), 2);

        // With a receptor, every model holds it as well, on a chain the
        // poses do not use, so the complex loads as one entity per frame.
        let receptor = dir.join("receptor.pdb");
        std::fs::write(
            &receptor,
            "ATOM      1  CA  GLY A   1       9.000   0.000   0.000  1.00  0.00           C
TER
END
",
        )
        .unwrap();
        let text = combine_poses(&paths, Some(&receptor)).unwrap();
        assert_eq!(text.matches("MODEL").count(), 2);
        assert_eq!(text.matches("ATOM").count(), 4, "receptor plus pose in each model");
        let chains: Vec<char> = text
            .lines()
            .filter(|l| l.starts_with("ATOM"))
            .map(|l| l.as_bytes()[21] as char)
            .collect();
        assert_eq!(chains, ['B', 'A', 'B', 'A'], "receptor moved off the pose's chain A");
        // Both models hold the same atoms, which a strict reader requires.
        assert_eq!(
            text.matches("GLY B").count(),
            2,
            "the receptor is in every model"
        );
    }

    #[test]
    fn writes_restraints_file() {
        // Matches FRODOCK's own consts.txt example.
        let text = restraints_file(&[Restraint {
            receptor: RestraintResidue { name: "GLY".into(), chain: "A".into(), number: 269 },
            ligand: RestraintResidue { name: "LEU".into(), chain: "A".into(), number: 84 },
            distance: 10.0,
        }]);
        let rows: Vec<&str> = text.lines().skip(2).collect();
        assert_eq!(
            rows[0].split_whitespace().collect::<Vec<_>>(),
            ["GLY", "A", "269", "LEU", "A", "84", "10"]
        );
    }
}
