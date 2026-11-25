//! AutoDock Vina and AutoDock4 docking, including covalent docking.
//!
//! The app writes `receptor.pdb` and `ligand.pdb` (both in world coordinates)
//! and `covalent_prep.py` into the job folder. The container (conda-forge /
//! bioconda: `vina`, `autodock`, `autogrid`, `meeko`, `rdkit`, `openbabel`)
//! then prepares both molecules with Meeko, docks inside the search box and
//! leaves `out.pdbqt` (Vina) or `lig.dlg` (AutoDock4) behind, which
//! [`parse_vina`] and [`parse_dlg`] turn into poses.
//!
//! Covalent docking uses Meeko's tethered method: the ligand is first turned
//! into its adduct with the target residue ([`COVALENT_PREP`]; cysteine,
//! lysine, serine or tyrosine), whose stand-in for the side chain is
//! superposed on the residue's CA and CB. The adduct is docked as that
//! residue's flexible side chain, with a one-atom dummy ligand because Vina and
//! AutoDock4 both insist on a ligand. Vina does not report the flexible
//! residue's energy, so covalent Vina poses come with ranks but no score;
//! AutoDock4 does report it.
//!
//! Cite: Eberhardt et al. 2021 J Chem Inf Model 61:3891 (Vina 1.2);
//! Morris et al. 2009 J Comput Chem 30:2785 (AutoDock4); Meeko (Forli lab).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const RECEPTOR: &str = "receptor.pdb";
pub const LIGAND: &str = "ligand.pdb";
pub const VINA_OUT: &str = "out.pdbqt";
pub const AD4_OUT: &str = "lig.dlg";
pub const COVALENT_PREP_FILE: &str = "covalent_prep.py";
pub const COVALENT_PREP: &str = include_str!("covalent_prep.py");

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    #[default]
    Vina,
    AutoDock4,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::Vina => "AutoDock Vina",
            Engine::AutoDock4 => "AutoDock4",
        }
    }
}

/// Reactive groups the ligand can bond to a residue through.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Warhead {
    #[default]
    Acrylamide,
    Chloroacetamide,
    VinylSulfonamide,
    /// Sulfonyl fluoride exchange: sulfonamide with Lys, sulfonate ester with Ser/Tyr.
    SulfonylFluoride,
    /// Forms a Schiff base with a lysine amine.
    Aldehyde,
}

/// Residues covalent docking supports.
pub const RESIDUES: [&str; 4] = ["CYS", "LYS", "SER", "TYR"];

impl Warhead {
    pub const ALL: [Warhead; 5] = [
        Warhead::Acrylamide,
        Warhead::Chloroacetamide,
        Warhead::VinylSulfonamide,
        Warhead::SulfonylFluoride,
        Warhead::Aldehyde,
    ];

    /// The name `covalent_prep.py` knows it by.
    pub fn key(self) -> &'static str {
        match self {
            Warhead::Acrylamide => "acrylamide",
            Warhead::Chloroacetamide => "chloroacetamide",
            Warhead::VinylSulfonamide => "vinylsulfonamide",
            Warhead::SulfonylFluoride => "sulfonylfluoride",
            Warhead::Aldehyde => "aldehyde",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Warhead::Acrylamide => "acrylamide",
            Warhead::Chloroacetamide => "chloroacetamide",
            Warhead::VinylSulfonamide => "vinyl sulfonamide",
            Warhead::SulfonylFluoride => "sulfonyl fluoride",
            Warhead::Aldehyde => "aldehyde (imine)",
        }
    }

    /// Can this group bond to `residue` (a three-letter name)?
    pub fn reacts_with(self, residue: &str) -> bool {
        match residue.to_ascii_uppercase().as_str() {
            "CYS" => matches!(self, Warhead::Acrylamide | Warhead::Chloroacetamide | Warhead::VinylSulfonamide),
            "LYS" => true,
            "SER" | "TYR" => self == Warhead::SulfonylFluoride,
            _ => false,
        }
    }

    pub fn for_residue(residue: &str) -> Vec<Warhead> {
        Warhead::ALL.into_iter().filter(|w| w.reacts_with(residue)).collect()
    }
}

/// SMARTS matching the stand-in side chain in the adduct, so Meeko can pin its
/// first two atoms (CA, CB) to the receptor residue.
fn tether_smarts(residue: &str) -> &'static str {
    match residue {
        "LYS" => "[CH3][CH2][CH2][CH2][CH2]N",
        "SER" => "[CH3][CH2]O",
        "TYR" => "[CH3][CH2]c1ccc(cc1)O",
        _ => "[CH3][CH2]S",
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Covalent {
    /// Chain id, residue name and number as `A:CYS:481` (chain may be empty).
    pub residue: String,
    pub warhead: Warhead,
}

impl Covalent {
    /// Upper-case residue name from the residue string (`A:CYS:481` → `CYS`).
    pub fn residue_name(&self) -> String {
        self.residue.split(':').nth(1).unwrap_or_default().to_ascii_uppercase()
    }

    /// `(chain, number)` if the residue string is well formed, the residue is
    /// one we support and the warhead can react with it.
    pub fn parse(&self) -> Result<(String, i32)> {
        let parts: Vec<&str> = self.residue.split(':').collect();
        let [chain, name, number] = parts[..] else {
            bail!("residue must look like A:CYS:481, got '{}'", self.residue);
        };
        if !chain.chars().all(|c| c.is_ascii_alphanumeric()) {
            bail!("bad chain id '{chain}'");
        }
        let upper = name.to_ascii_uppercase();
        if !RESIDUES.contains(&upper.as_str()) {
            bail!("covalent docking supports {}; {name} is not supported yet", RESIDUES.join(", "));
        }
        if !self.warhead.reacts_with(&upper) {
            bail!(
                "a {} group does not bond to {upper}; use {}",
                self.warhead.label(),
                Warhead::for_residue(&upper).iter().map(|w| w.label()).collect::<Vec<_>>().join(" or ")
            );
        }
        let number: i32 = number.parse().with_context(|| format!("bad residue number '{number}'"))?;
        Ok((chain.to_string(), number))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub engine: Engine,
    /// Search box centre and size, Å (world coordinates).
    pub center: [f32; 3],
    pub size: [f32; 3],
    /// Vina: search effort (8 is the default; 32+ for a careful run).
    pub exhaustiveness: u32,
    /// Vina: poses to keep. AutoDock4: independent runs.
    pub num_modes: u32,
    /// AutoDock4: energy evaluations per run.
    pub ad4_evals: u64,
    pub covalent: Option<Covalent>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            engine: Engine::Vina,
            center: [0.0; 3],
            size: [22.0; 3],
            exhaustiveness: 8,
            num_modes: 9,
            ad4_evals: 2_500_000,
            covalent: None,
        }
    }
}

fn check_number(name: &str, v: f32) -> Result<f32> {
    anyhow::ensure!(v.is_finite(), "{name} is not a number");
    Ok(v)
}

/// The shell script the container runs in the job folder.
pub fn command(s: &Settings) -> Result<String> {
    for (i, axis) in ["x", "y", "z"].iter().enumerate() {
        check_number(&format!("centre {axis}"), s.center[i])?;
        anyhow::ensure!(
            check_number(&format!("size {axis}"), s.size[i])? >= 4.0,
            "the box must be at least 4 Å wide (size {axis} is {})",
            s.size[i]
        );
    }
    let [cx, cy, cz] = s.center;
    let [sx, sy, sz] = s.size;
    let covalent = match &s.covalent {
        Some(c) => Some((c.parse()?, c)),
        None => None,
    };

    let mut steps = vec![
        "fail() { echo \"AutoDock step failed: $1\" >&2; exit 1; }".to_string(),
        "need() { [ -s \"$1\" ] || fail \"$2 (expected $1)\"; }".to_string(),
    ];

    // Receptor. `-x` drops residues Meeko cannot template (truncated side
    // chains are common in crystal structures) and `--default_altloc A`
    // settles alternate conformations; both are reported in the log.
    let mut receptor = format!(
        "mk_prepare_receptor.py -i {RECEPTOR} -o rec -p -v -x --default_altloc A \
         --box_center {cx:.3} {cy:.3} {cz:.3} --box_size {sx:.2} {sy:.2} {sz:.2}"
    );
    if s.engine == Engine::AutoDock4 {
        receptor.push_str(" -g");
    }
    if let Some(((chain, number), _)) = &covalent {
        // The residue is replaced by the docked adduct.
        receptor.push_str(&format!(" -d {chain}:{number}"));
    }
    steps.push(format!("{receptor} || fail 'receptor preparation'"));
    steps.push("need rec.pdbqt 'receptor'".into());

    // Ligand: protonate at pH 7.4 and perceive bond orders from the 3D structure.
    steps.push(format!(
        "obabel -ipdb {LIGAND} -osdf -O ligand.sdf -p 7.4 || fail 'ligand conversion'"
    ));
    steps.push("need ligand.sdf 'ligand'".into());

    // What is docked: the ligand, or for covalent runs a dummy ligand plus
    // the adduct as a flexible residue.
    let (ligand_arg, flex_arg) = if let Some((_, c)) = &covalent {
        let key = c.warhead.key();
        let name = c.residue_name();
        steps.push(format!(
            "python {COVALENT_PREP_FILE} ligand.sdf adduct.sdf {key} {name} || fail 'the ligand has no {key} group to bond with'"
        ));
        steps.push(format!(
            "mk_prepare_ligand.py -i adduct.sdf -o adduct.pdbqt --receptor {RECEPTOR} --rec_residue \"{}\" \
             --tether_smarts \"{}\" --tether_smarts_indices 1 2 || fail 'covalent ligand preparation'",
            c.residue,
            tether_smarts(&name)
        ));
        steps.push("need adduct.pdbqt 'covalent adduct'".into());
        // One hydrogen, which neither engine scores, parked inside the box.
        steps.push(format!(
            "printf 'ROOT\\nATOM      1  H   DUM     1    {cx:8.3}{cy:8.3}{cz:8.3}  0.00  0.00    +0.000 HD\\nENDROOT\\nTORSDOF 0\\n' > dummy.pdbqt"
        ));
        ("dummy.pdbqt", Some("adduct.pdbqt"))
    } else {
        steps.push("mk_prepare_ligand.py -i ligand.sdf -o ligand.pdbqt || fail 'ligand preparation'".into());
        steps.push("need ligand.pdbqt 'prepared ligand'".into());
        ("ligand.pdbqt", None)
    };

    match s.engine {
        Engine::Vina => {
            let flex = flex_arg.map(|f| format!(" --flex {f}")).unwrap_or_default();
            steps.push(format!(
                "vina --receptor rec.pdbqt --ligand {ligand_arg}{flex} --config rec.box.txt \
                 --exhaustiveness {} --num_modes {} --out {VINA_OUT} || fail 'Vina'",
                s.exhaustiveness.max(1),
                s.num_modes.max(1)
            ));
            steps.push(format!("need {VINA_OUT} 'Vina result'"));
        }
        Engine::AutoDock4 => {
            steps.push("autogrid4 -p rec.gpf -l rec.glg || fail 'AutoGrid'".into());
            steps.push("need rec.maps.fld 'grid maps'".into());
            // AutoDock4 takes at most 14 atom types, and the GPF lists them all:
            // use just the ones the molecules being docked have.
            let files = if flex_arg.is_some() { "adduct.pdbqt dummy.pdbqt" } else { "ligand.pdbqt" };
            steps.push(format!(
                "types=$(awk '/^(ATOM|HETATM)/{{print $NF}}' {files} | sort -u | tr '\n' ' ')"
            ));
            let flexres = flex_arg.map(|f| format!("flexres {f}\n")).unwrap_or_default();
            let torsdof = if flex_arg.is_some() { "torsdof 0" } else { "torsdof $(awk '/^TORSDOF/{print $2}' ligand.pdbqt)" };
            steps.push(format!(
                "cat > lig.dpf <<DPF\n\
                 autodock_parameter_version 4.2\n\
                 outlev 1\n\
                 intelec\n\
                 seed pid time\n\
                 ligand_types $types\n\
                 fld rec.maps.fld\n\
                 $(for t in $types; do echo \"map rec.$t.map\"; done)\n\
                 elecmap rec.e.map\n\
                 desolvmap rec.d.map\n\
                 move {ligand_arg}\n\
                 {flexres}about {cx:.3} {cy:.3} {cz:.3}\n\
                 tran0 random\n\
                 quaternion0 random\n\
                 dihe0 random\n\
                 {torsdof}\n\
                 rmstol 2.0\n\
                 extnrg 1000.0\n\
                 e0max 0.0 10000\n\
                 ga_pop_size 150\n\
                 ga_num_evals {}\n\
                 ga_num_generations 27000\n\
                 ga_elitism 1\n\
                 ga_mutation_rate 0.02\n\
                 ga_crossover_rate 0.8\n\
                 ga_window_size 10\n\
                 ga_cauchy_alpha 0.0\n\
                 ga_cauchy_beta 1.0\n\
                 set_ga\n\
                 sw_max_its 300\n\
                 sw_max_succ 4\n\
                 sw_max_fail 4\n\
                 sw_rho 1.0\n\
                 sw_lb_rho 0.01\n\
                 ls_search_freq 0.06\n\
                 set_psw1\n\
                 unbound_model bound\n\
                 ga_run {}\n\
                 analysis\n\
                 DPF",
                s.ad4_evals.max(10_000),
                s.num_modes.max(1)
            ));
            steps.push(format!("autodock4 -p lig.dpf -l {AD4_OUT} || fail 'AutoDock4'"));
            steps.push(format!("need {AD4_OUT} 'AutoDock4 result'"));
        }
    }
    steps.push("echo done".into());
    Ok(steps.join("\n"))
}

/// One docked pose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// 1-based, best first.
    pub rank: usize,
    /// Predicted binding energy, kcal/mol (more negative is better); `None`
    /// when the engine does not report one for this kind of run.
    pub energy: Option<f32>,
    /// Geometric centre, Å.
    pub position: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Poses {
    pub poses: Vec<Pose>,
    /// One multi-model PDB, a model per pose, atoms as HETATM in chain L.
    pub pdb: String,
}

struct Model {
    energy: Option<f32>,
    /// What the poses are ranked by (lower is better).
    rank_key: f32,
    atoms: Vec<String>,
}

/// Element symbol for an AutoDock atom type.
fn element_of(autodock_type: &str) -> String {
    match autodock_type {
        "A" => "C".into(),
        "NA" => "N".into(),
        "OA" => "O".into(),
        "SA" => "S".into(),
        "HD" | "H" => "H".into(),
        t => t.to_string(),
    }
}

/// A PDB `HETATM` line from a PDBQT/DLG atom line, renumbered.
fn pdb_line(serial: usize, line: &str) -> Option<String> {
    if line.len() < 54 {
        return None;
    }
    let name = line.get(12..16)?;
    let resname = line.get(17..20)?;
    let (x, y, z) = (line.get(30..38)?, line.get(38..46)?, line.get(46..54)?);
    let ty = line.split_whitespace().last()?;
    Some(format!(
        "HETATM{serial:>5} {name} {resname} L{:>4}    {x}{y}{z}  1.00  0.00          {:>2}",
        1,
        element_of(ty)
    ))
}

/// Assemble models into a multi-model PDB and the pose table.
fn assemble(models: Vec<Model>) -> Result<Poses> {
    anyhow::ensure!(!models.is_empty(), "the docking run produced no poses");
    let mut pdb = String::new();
    let mut poses = Vec::new();
    for (i, m) in models.iter().enumerate() {
        pdb.push_str(&format!("MODEL     {:>4}\n", i + 1));
        let mut sum = [0.0f32; 3];
        let mut n = 0.0;
        for (k, line) in m.atoms.iter().enumerate() {
            if let Some(l) = pdb_line(k + 1, line) {
                for (axis, range) in [(0, 30..38), (1, 38..46), (2, 46..54)] {
                    sum[axis] += line[range].trim().parse::<f32>().unwrap_or(0.0);
                }
                n += 1.0;
                pdb.push_str(&l);
                pdb.push('\n');
            }
        }
        pdb.push_str("ENDMDL\n");
        let n = if n > 0.0 { n } else { 1.0 };
        poses.push(Pose {
            rank: i + 1,
            energy: m.energy,
            position: [sum[0] / n, sum[1] / n, sum[2] / n],
        });
    }
    pdb.push_str("END\n");
    Ok(Poses { poses, pdb })
}

/// Atom lines of one model: those inside `BEGIN_RES`/`END_RES` blocks when
/// there are any (the covalent adduct; the dummy ligand is left out), else all.
fn model_atoms<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<String> {
    let (mut flex, mut all) = (Vec::new(), Vec::new());
    let mut in_res = false;
    for line in lines {
        if line.starts_with("BEGIN_RES") {
            in_res = true;
        } else if line.starts_with("END_RES") {
            in_res = false;
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") {
            all.push(line.to_string());
            if in_res {
                flex.push(line.to_string());
            }
        }
    }
    if flex.is_empty() { all } else { flex }
}

/// Poses from Vina's `out.pdbqt`. With `scores` false (covalent runs, where
/// Vina leaves the flexible residue out of its figure) energies are dropped.
pub fn parse_vina(text: &str, scores: bool) -> Result<Poses> {
    let mut models = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in text.lines() {
        if line.starts_with("MODEL") {
            current = Some(Vec::new());
        } else if line.starts_with("ENDMDL") {
            if let Some(lines) = current.take() {
                let energy = lines
                    .iter()
                    .find_map(|l| l.strip_prefix("REMARK VINA RESULT:"))
                    .and_then(|r| r.split_whitespace().next())
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|_| scores);
                let rank_key = energy.unwrap_or(f32::MAX);
                models.push(Model { energy, rank_key, atoms: model_atoms(lines.into_iter()) });
            }
        } else if let Some(lines) = &mut current {
            lines.push(line);
        }
    }
    assemble(models)
}

/// The first number after `label` on the first line that has it.
fn dlg_value(lines: &[&str], label: &str) -> Option<f32> {
    lines
        .iter()
        .find_map(|l| l.split_once(label))
        .and_then(|(_, rest)| rest.trim_start_matches([' ', '=']).split_whitespace().next())
        .and_then(|v| v.parse::<f32>().ok())
}

/// Poses from an AutoDock4 `.dlg`, best first.
///
/// Normally ranked by the estimated free energy of binding. For covalent runs
/// (`covalent`) that figure leaves the flexible adduct out, so poses are
/// ranked by intermolecular plus total internal energy instead and no energy
/// is reported.
pub fn parse_dlg(text: &str, covalent: bool) -> Result<Poses> {
    let mut models = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in text.lines() {
        let Some(body) = line.strip_prefix("DOCKED: ") else { continue };
        if body.starts_with("MODEL") {
            current = Some(Vec::new());
        } else if body.starts_with("ENDMDL") {
            if let Some(lines) = current.take() {
                let estimated = dlg_value(&lines, "Estimated Free Energy of Binding");
                let rank_key = if covalent {
                    match (
                        dlg_value(&lines, "(1) Final Intermolecular Energy"),
                        dlg_value(&lines, "(2) Final Total Internal Energy"),
                    ) {
                        (Some(a), Some(b)) => a + b,
                        _ => f32::MAX,
                    }
                } else {
                    estimated.unwrap_or(f32::MAX)
                };
                models.push(Model {
                    energy: estimated.filter(|_| !covalent),
                    rank_key,
                    atoms: model_atoms(lines.into_iter()),
                });
            }
        } else if let Some(lines) = &mut current {
            lines.push(body);
        }
    }
    models.sort_by(|a, b| a.rank_key.total_cmp(&b.rank_key));
    assemble(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
        std::fs::read_to_string(path).unwrap()
    }

    fn model_sizes(pdb: &str) -> Vec<usize> {
        pdb.split("MODEL").skip(1).map(|m| m.lines().filter(|l| l.starts_with("HETATM")).count()).collect()
    }

    #[test]
    fn vina_poses_and_scores() {
        let p = parse_vina(&fixture("vina_out.pdbqt"), true).unwrap();
        assert_eq!(p.poses.len(), 5);
        assert!((p.poses[0].energy.unwrap() + 11.534).abs() < 1e-3);
        assert!(p.poses.windows(2).all(|w| w[0].energy <= w[1].energy));
        let sizes = model_sizes(&p.pdb);
        assert_eq!(sizes.len(), 5);
        assert!(sizes.iter().all(|&n| n == sizes[0] && n > 40), "{sizes:?}");
        // The 1HSG inhibitor is docked into its own site.
        let [x, y, z] = p.poses[0].position;
        assert!((x - 13.0).abs() < 4.0 && (y - 22.5).abs() < 4.0 && (z - 5.5).abs() < 4.0, "{:?}", p.poses[0]);
    }

    #[test]
    fn covalent_vina_keeps_only_the_adduct_and_no_score() {
        let p = parse_vina(&fixture("vina_covalent_out.pdbqt"), false).unwrap();
        assert_eq!(p.poses.len(), 2);
        assert!(p.poses.iter().all(|p| p.energy.is_none()));
        let sizes = model_sizes(&p.pdb);
        // 38 adduct atoms; the one-atom dummy ligand is not part of the pose.
        assert!(sizes.iter().all(|&n| n == 38), "{sizes:?}");
        assert!(p.pdb.contains(" S  "), "the cysteine sulfur is in the pose");
    }

    #[test]
    fn autodock4_poses_sorted_by_energy() {
        let p = parse_dlg(&fixture("ad4_docked.dlg"), false).unwrap();
        assert_eq!(p.poses.len(), 5);
        assert!((p.poses[0].energy.unwrap() + 10.87).abs() < 1e-2);
        assert!(p.poses.windows(2).all(|w| w[0].energy <= w[1].energy));
        assert!(model_sizes(&p.pdb).iter().all(|&n| n > 40));
    }

    #[test]
    fn covalent_autodock4_ranks_by_total_energy_and_reports_none() {
        let p = parse_dlg(&fixture("ad4_covalent.dlg"), true).unwrap();
        assert_eq!(p.poses.len(), 5);
        assert!(p.poses.iter().all(|p| p.energy.is_none()));
        assert!(model_sizes(&p.pdb).iter().all(|&n| n == 38));
    }

    #[test]
    fn commands_for_each_engine() {
        let mut s = Settings { center: [1.0, 2.0, 3.0], ..Default::default() };
        let vina = command(&s).unwrap();
        assert!(vina.contains("vina --receptor rec.pdbqt --ligand ligand.pdbqt --config rec.box.txt"));
        assert!(!vina.contains("--flex") && !vina.contains("autogrid4"));

        s.engine = Engine::AutoDock4;
        let ad4 = command(&s).unwrap();
        assert!(ad4.contains(" -g ") || ad4.contains(" -g\n") || ad4.contains("-g ||"));
        assert!(ad4.contains("autogrid4") && ad4.contains("autodock4 -p lig.dpf"));
        assert!(ad4.contains("about 1.000 2.000 3.000"));
    }

    #[test]
    fn covalent_commands_use_the_adduct_as_a_flexible_residue() {
        let mut s = Settings {
            covalent: Some(Covalent { residue: "A:CYS:481".into(), warhead: Warhead::Acrylamide }),
            ..Default::default()
        };
        let vina = command(&s).unwrap();
        assert!(vina.contains("--flex adduct.pdbqt") && vina.contains("--ligand dummy.pdbqt"));
        assert!(vina.contains("-d A:481") && vina.contains("--rec_residue \"A:CYS:481\""));
        assert!(vina.contains("covalent_prep.py ligand.sdf adduct.sdf acrylamide CYS"));

        s.engine = Engine::AutoDock4;
        let ad4 = command(&s).unwrap();
        assert!(ad4.contains("flexres adduct.pdbqt") && ad4.contains("move dummy.pdbqt") && ad4.contains("torsdof 0"));
    }

    #[test]
    fn each_residue_gets_its_own_tether() {
        let cmd = |residue: &str, warhead| {
            command(&Settings {
                covalent: Some(Covalent { residue: residue.into(), warhead }),
                ..Default::default()
            })
            .unwrap()
        };
        let lys = cmd("A:LYS:258", Warhead::Aldehyde);
        assert!(lys.contains("adduct.sdf aldehyde LYS") && lys.contains("[CH3][CH2][CH2][CH2][CH2]N"));
        assert!(lys.contains("-d A:258"));
        let ser = cmd("A:SER:195", Warhead::SulfonylFluoride);
        assert!(ser.contains("sulfonylfluoride SER") && ser.contains("\"[CH3][CH2]O\""));
        let tyr = cmd("B:TYR:41", Warhead::SulfonylFluoride);
        assert!(tyr.contains("sulfonylfluoride TYR") && tyr.contains("[CH3][CH2]c1ccc(cc1)O") && tyr.contains("-d B:41"));
        assert_eq!(Warhead::for_residue("ser"), vec![Warhead::SulfonylFluoride]);
        assert_eq!(Warhead::for_residue("CYS").len(), 3);
        assert_eq!(Warhead::for_residue("LYS").len(), 5);
    }

    #[test]
    fn bad_input_is_refused() {
        let bad = |residue: &str| Settings {
            covalent: Some(Covalent { residue: residue.into(), warhead: Warhead::Acrylamide }),
            ..Default::default()
        };
        // Lysine is supported, but not with every warhead's chemistry, and not every residue.
        assert!(command(&bad("A:LYS:12")).is_ok());
        assert!(command(&bad("A:ASP:12")).unwrap_err().to_string().contains("not supported"));
        let wrong = Settings {
            covalent: Some(Covalent { residue: "A:SER:195".into(), warhead: Warhead::Acrylamide }),
            ..Default::default()
        };
        assert!(command(&wrong).unwrap_err().to_string().contains("sulfonyl fluoride"));
        assert!(command(&bad("A;rm -rf /:CYS:1")).is_err());
        assert!(command(&bad("CYS:12")).is_err());
        let tiny = Settings { size: [2.0, 20.0, 20.0], ..Default::default() };
        assert!(command(&tiny).is_err());
    }
}
