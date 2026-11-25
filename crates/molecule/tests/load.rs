use molecule::Structure;
use std::path::PathBuf;

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets").join(name)
}

#[test]
fn loads_1hsg() {
    let s = Structure::load(asset("1hsg.pdb")).unwrap();
    assert_eq!(s.name, "1HSG");
    assert_eq!(s.chain_count(), 2);
    // 1514 ATOM + 45 MK1 + 127 waters = 1686 HETATM/ATOM records.
    let polymer = s.atoms.iter().filter(|a| !a.hetero).count();
    assert_eq!(polymer, 1514);
    let ligand: Vec<_> = s
        .atoms
        .iter()
        .filter(|a| s.residues[a.residue as usize].name == "MK1")
        .collect();
    assert_eq!(ligand.len(), 45);
    let protein_residues = s
        .residues
        .iter()
        .filter(|r| !r.is_water() && r.name != "MK1")
        .count();
    assert_eq!(protein_residues, 198);
    // A protein has roughly as many bonds as atoms.
    let ratio = s.bonds.len() as f32 / polymer as f32;
    assert!((0.95..1.15).contains(&ratio), "bond/atom ratio {ratio}");
}

#[test]
fn loads_mk1_sdf() {
    let s = Structure::load(asset("MK1_ideal.sdf")).unwrap();
    assert_eq!(s.atoms.len(), 92);
    assert_eq!(s.bonds.len(), 96);
}

#[test]
fn assigns_secondary_structure_1hsg() {
    use molecule::SecondaryStructure as Ss;
    let s = Structure::load(asset("1hsg.pdb")).unwrap();
    let count = |ss| s.secondary.iter().filter(|x| **x == ss).count();
    // HIV protease: mostly β, one short α-helix (~residues 87-93) per monomer.
    let (helix, sheet) = (count(Ss::Helix), count(Ss::Sheet));
    assert!((8..=20).contains(&helix), "helix residues: {helix}");
    assert!(sheet >= 60, "sheet residues: {sheet}");
    let helix_a90 = s
        .residues
        .iter()
        .position(|r| r.chain == "A" && r.number == 90)
        .unwrap();
    assert_eq!(s.secondary[helix_a90], Ss::Helix);
}

#[test]
fn superposes_1hsg_chain_a_onto_b() {
    use glam::Mat4;
    use molecule::align::{AlignOptions, align};
    let s = Structure::load(asset("1hsg.pdb")).unwrap();
    let chain = |c: &str| -> Vec<usize> {
        s.residues.iter().enumerate().filter(|(_, r)| r.chain == c).map(|(i, _)| i).collect()
    };
    let options = AlignOptions {
        mobile_residues: Some(chain("A")),
        target_residues: Some(chain("B")),
        ..Default::default()
    };
    let fit = align(&s, Mat4::IDENTITY, &s, Mat4::IDENTITY, &options).unwrap();
    assert!(fit.n_pairs >= 95, "pairs {}", fit.n_pairs);
    assert!(fit.rmsd < 1.0, "rmsd {}", fit.rmsd);
}

#[test]
fn reads_nmr_models_as_frames() {
    // Two-model PDB built in memory.
    let dir = std::env::temp_dir().join("openmols-test-models");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("two_models.pdb");
    let atom = |serial: usize, x: f32| format!(
        "ATOM  {serial:>5}  CA  ALA A   1    {x:>8.3}{:>8.3}{:>8.3}  1.00  0.00           C\n", 0.0, 0.0);
    let text = format!("MODEL        1\n{}{}ENDMDL\nMODEL        2\n{}{}ENDMDL\nEND\n",
        atom(1, 0.0), atom(2, 3.8), atom(1, 1.0), atom(2, 4.8));
    std::fs::write(&path, text).unwrap();
    let mut s = Structure::load(&path).unwrap();
    assert_eq!(s.frame_count(), 2);
    s.set_frame(1);
    assert!((s.atoms[0].position.x - 1.0).abs() < 1e-4);
}

#[test]
fn bad_pdb_is_an_error_not_a_panic() {
    let dir = std::env::temp_dir().join("openmols-test-bad");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("negative_occupancy.pdb");
    std::fs::write(
        &path,
        "ATOM      1  HA  KAA   259      18.430  17.086   1.211 -0.55  0.60\nEND\n",
    )
    .unwrap();
    // Either parses or returns an error — it must not panic.
    let _ = Structure::load(&path);
}

#[test]
fn loads_sanitized_kvfinder_cavities() {
    // pyKVFinder output for 1HSG after jobs::kvfinder::sanitize_cavity_pdb.
    let s = Structure::load(asset("../crates/molecule/tests/fixtures/cavities_1hsg.pdb")).unwrap();
    assert_eq!(s.atoms.len(), 3223);
    let kaa = s.residues.iter().filter(|r| r.name == "KAA").map(|r| r.atoms.len()).sum::<usize>();
    assert_eq!(kaa, 3193);
}

#[test]
fn loads_short_pdb_lines_from_tools() {
    // FRODOCK writes 66-column ATOM lines with no element column.
    let dir = std::env::temp_dir().join("openmols-test-short");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("frodock_pose.pdb");
    std::fs::write(
        &path,
        "ATOM      1  N   LYS A   1      22.665  10.368  64.050 -0.30  0.00\n\
         ATOM      2  CA  LYS A   1      21.471   9.727  64.662  0.21  0.00\n\
         TER\nEND\n",
    )
    .unwrap();
    let s = Structure::load(&path).unwrap();
    assert_eq!(s.atoms.len(), 2);
    assert_eq!(s.atoms[0].element.symbol(), "N");
    assert_eq!(s.atoms[1].element.symbol(), "C");
    assert!((s.atoms[0].position.x - 22.665).abs() < 1e-3);
}

#[test]
fn recognises_pdb_file_names() {
    use molecule::is_pdb_like;
    assert!(is_pdb_like("1hsg.pdb"));
    assert!(is_pdb_like("1hsg.pdb1"), "biological assembly");
    assert!(is_pdb_like("/tmp/a.b/1hsg.pdb12"));
    assert!(is_pdb_like("pdb1hsg.ent"));
    assert!(is_pdb_like("old.brk"));
    assert!(!is_pdb_like("1hsg.cif"));
    assert!(!is_pdb_like("ligand.pdbqt"), "AutoDock format is not plain PDB");
    assert!(!is_pdb_like("notes.txt"));
}

#[test]
fn loads_end_delimited_conformations() {
    // GROMACS-style: frames separated by bare END, no MODEL records,
    // and short lines with no element column.
    let dir = std::env::temp_dir().join("openmols-test-endframes");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("conformations.pdb");
    let frame = |x: f32| {
        format!(
            "ATOM      1  N   ASP A   1      {x:>6.3}  10.090  15.270  1.00  0.00\n\
             ATOM      2  CA  ASP A   1      {:>6.3}   9.040  15.280  1.00  0.00\nEND\n",
            x + 1.0
        )
    };
    let text = format!("CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1\n{}{}{}",
        frame(69.98), frame(47.73), frame(50.00));
    std::fs::write(&path, text).unwrap();

    let mut s = Structure::load(&path).unwrap();
    assert_eq!(s.atoms.len(), 2, "topology comes from the first frame only");
    assert_eq!(s.frame_count(), 3);
    assert!((s.atoms[0].position.x - 69.98).abs() < 1e-2);
    s.set_frame(1);
    assert!((s.atoms[0].position.x - 47.73).abs() < 1e-2);
    s.set_frame(2);
    assert!((s.atoms[0].position.x - 50.0).abs() < 1e-2);
}

#[test]
fn separates_copies_that_reuse_residue_numbers() {
    // Writers that put every copy of an oligomer on one chain, restarting the
    // numbering, would otherwise collapse into a single molecule.
    let dir = std::env::temp_dir().join("openmols-test-copies");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("oligomer.pdb");
    let mut text = String::new();
    for copy in 0..3 {
        for residue in 1..=2 {
            text.push_str(&format!(
                "ATOM  {:>5}  CA  ALA A{:>4}      {:>6.3}   0.000   0.000  1.00  0.00\n",
                copy * 2 + residue,
                residue,
                copy as f32
            ));
        }
    }
    text.push_str("END\n");
    std::fs::write(&path, text).unwrap();

    let s = Structure::load(&path).unwrap();
    assert_eq!(s.atoms.len(), 6);
    assert_eq!(s.residues.len(), 6, "three copies of two residues");
    assert_eq!(s.chain_count(), 3, "each copy gets its own chain");
}

#[test]
fn keeps_chain_labels_consistent_across_models() {
    // Docking poses: the same oligomer in every model, written short (no
    // element column) and reusing residue numbers per copy. The chain
    // splitter must label every model the same way, or a reader that matches
    // models atom by atom rejects the file.
    let dir = std::env::temp_dir().join("openmols-test-pose-models");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("poses.pdb");
    let mut text = String::new();
    for model in 1..=3 {
        text.push_str(&format!("MODEL     {model:>4}\n"));
        for copy in 0..2 {
            for residue in 1..=2 {
                text.push_str(&format!(
                    "ATOM  {:>5}  CA  ALA A{:>4}    {:>8.3}{:>8.3}   0.000  1.00  0.00\n",
                    copy * 2 + residue,
                    residue,
                    copy as f32,
                    model as f32
                ));
            }
        }
        text.push_str("ENDMDL\n");
    }
    text.push_str("END\n");
    std::fs::write(&path, text).unwrap();

    let s = Structure::load(&path).unwrap();
    assert_eq!(s.atoms.len(), 4);
    assert_eq!(s.chain_count(), 2, "each copy gets its own chain");
    assert_eq!(s.frame_count(), 3, "models load as frames");
}
