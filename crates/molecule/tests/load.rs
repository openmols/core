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
