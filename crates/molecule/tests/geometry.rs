use glam::Vec3;
use molecule::Structure;
use molecule::geometry::{ContactKind, Site, contacts, sasa};
use std::path::PathBuf;

fn load() -> Structure {
    Structure::load(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/1hsg.pdb")).unwrap()
}

fn sites(s: &Structure, pick: impl Fn(usize) -> bool) -> Vec<Site> {
    (0..s.atoms.len())
        .filter(|&i| pick(i))
        .map(|i| Site { pos: s.atoms[i].position, element: s.atoms[i].element, id: i })
        .collect()
}

#[test]
fn ligand_contacts_and_burial_in_1hsg() {
    let s = load();
    let is_lig = |i: usize| s.residues[s.atoms[i].residue as usize].name == "MK1";
    let ligand = sites(&s, is_lig);
    let protein = sites(&s, |i| !s.atoms[i].hetero);
    assert_eq!(ligand.len(), 45);

    let found = contacts(&ligand, &protein, 4.0);
    assert!(found.len() > 30, "only {} contacts", found.len());
    assert!(found.iter().any(|c| c.kind == ContactKind::HBond), "no H-bonds to the inhibitor");
    // A crystal structure has no real clashes with its own ligand.
    assert!(found.iter().all(|c| c.kind != ContactKind::Clash || c.distance > 2.0));
    assert!(found[0].distance > 2.0 && found[0].distance < 3.5);

    // The inhibitor is mostly buried: its area in the complex is far below free.
    let ids: Vec<usize> = (0..ligand.len()).collect();
    let free: f32 = sasa(&ligand, 1.4, 200, None).iter().sum();
    let mut complex = ligand.clone();
    complex.extend(protein.iter().copied());
    let bound: f32 = sasa(&complex, 1.4, 200, Some(&ids)).iter().sum();
    assert!(bound < free * 0.4, "bound {bound} vs free {free}");
    let _ = Vec3::ZERO;
}
