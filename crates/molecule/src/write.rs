//! PDB writer: exports (optionally transformed) coordinates as input for
//! external tools such as FRODOCK, Rosetta, pyKVFinder or GROMACS.

use crate::Structure;
use glam::Mat4;
use std::fmt::Write as _;

/// Write the atoms accepted by `keep`, transformed by `transform`, as PDB text.
/// Atoms are renumbered; a `TER` record separates chains.
pub fn write_pdb(structure: &Structure, transform: Mat4, keep: impl Fn(usize) -> bool) -> String {
    let mut out = String::new();
    let mut serial = 0usize;
    let mut last_chain: Option<&str> = None;
    for (i, atom) in structure.atoms.iter().enumerate() {
        if !keep(i) {
            continue;
        }
        let residue = &structure.residues[atom.residue as usize];
        if let Some(prev) = last_chain
            && prev != residue.chain
        {
            let _ = writeln!(out, "TER");
        }
        last_chain = Some(&residue.chain);
        serial += 1;
        let p = transform.transform_point3(atom.position);
        let symbol = atom.element.symbol();
        // PDB convention: 1-letter elements with short names start in column 14.
        let name = if atom.name.len() < 4 && symbol.len() == 1 {
            format!(" {:<3}", atom.name)
        } else {
            format!("{:<4}", atom.name)
        };
        let chain = residue.chain.chars().next().unwrap_or(' ');
        let insertion = residue
            .insertion_code
            .as_deref()
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let _ = writeln!(
            out,
            "{:<6}{:>5} {:4} {:>3} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
            if atom.hetero { "HETATM" } else { "ATOM" },
            serial % 100_000,
            name,
            &residue.name[..residue.name.len().min(3)],
            chain,
            residue.number % 10_000,
            insertion,
            p.x,
            p.y,
            p.z,
            1.0,
            atom.b_factor,
            symbol.to_uppercase(),
        );
    }
    let _ = writeln!(out, "TER\nEND");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Atom, Element, Residue};
    use glam::Vec3;

    #[test]
    fn writes_fixed_columns() {
        let s = Structure {
            atoms: vec![Atom {
                name: "CA".into(),
                element: Element::C,
                position: Vec3::new(1.0, 2.0, 3.0),
                residue: 0,
                hetero: false,
                serial: 7,
                b_factor: 12.5,
            }],
            residues: vec![Residue {
                name: "ALA".into(),
                number: 42,
                insertion_code: None,
                chain: "A".into(),
                atoms: 0..1,
            }],
            ..Default::default()
        };
        let text = write_pdb(&s, Mat4::from_translation(Vec3::X), |_| true);
        let line = text.lines().next().unwrap();
        assert_eq!(&line[0..6], "ATOM  ");
        assert_eq!(&line[12..16], " CA ");
        assert_eq!(&line[17..20], "ALA");
        assert_eq!(&line[21..22], "A");
        assert_eq!(line[22..26].trim(), "42");
        assert_eq!(line[30..38].trim(), "2.000");
        assert_eq!(line[60..66].trim(), "12.50");
    }
}
