//! Export the bundled example the way the app does (`receptor.pdb` = polymer,
//! `ligand.pdb` = the ligand, both via `write_pdb`) and print the box centre.
//!
//! cargo run -p jobs --example example_export -- assets/1hsg.pdb out_dir

use molecule::{AtomKind, Structure};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let s = Structure::load(&a[0]).unwrap();
    let ligand = s
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| r.atoms.start < r.atoms.end && s.kind(r.atoms.start as usize) == AtomKind::Ligand && r.atoms.len() >= 4)
        .max_by_key(|(_, r)| r.atoms.len())
        .map(|(i, _)| i)
        .unwrap();
    let in_ligand = |i: usize| s.atoms[i].residue as usize == ligand;
    let dir = std::path::Path::new(&a[1]);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("receptor.pdb"), molecule::write_pdb(&s, glam::Mat4::IDENTITY, |i| !in_ligand(i) && s.kind(i) == AtomKind::Polymer)).unwrap();
    std::fs::write(dir.join("ligand.pdb"), molecule::write_pdb(&s, glam::Mat4::IDENTITY, in_ligand)).unwrap();
    let pts: Vec<_> = (0..s.atoms.len()).filter(|&i| in_ligand(i)).map(|i| s.atoms[i].position).collect();
    let c = pts.iter().sum::<glam::Vec3>() / pts.len() as f32;
    println!("{:.2} {:.2} {:.2}", c.x, c.y, c.z);
}
