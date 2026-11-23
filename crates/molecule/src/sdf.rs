//! Minimal MDL MOL / SDF (V2000) reader. Reads the first record only.

use crate::{Atom, Bond, Element, Residue, Structure};
use anyhow::{Context, Result, bail};
use glam::Vec3;

pub fn parse(text: &str) -> Result<Structure> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 4 {
        bail!("MOL/SDF file is too short");
    }
    let name = lines[0].trim().to_string();
    let counts = lines[3];
    if counts.contains("V3000") {
        bail!("V3000 MOL files are not supported yet");
    }
    let field = |line: &str, range: std::ops::Range<usize>| -> Option<String> {
        line.get(range).map(|s| s.trim().to_string())
    };
    let n_atoms: usize = field(counts, 0..3)
        .and_then(|s| s.parse().ok())
        .context("invalid atom count in counts line")?;
    let n_bonds: usize = field(counts, 3..6)
        .and_then(|s| s.parse().ok())
        .context("invalid bond count in counts line")?;

    if lines.len() < 4 + n_atoms + n_bonds {
        bail!("MOL/SDF file ends before the atom/bond blocks are complete");
    }

    let mut structure = Structure {
        name,
        residues: vec![Residue {
            name: "LIG".into(),
            number: 1,
            insertion_code: None,
            chain: String::new(),
            atoms: 0..n_atoms as u32,
        }],
        ..Default::default()
    };

    for (i, line) in lines[4..4 + n_atoms].iter().enumerate() {
        let mut parts = line.split_whitespace();
        let mut coord = || -> Result<f32> {
            parts
                .next()
                .and_then(|s| s.parse().ok())
                .with_context(|| format!("bad coordinate on atom line {}", i + 1))
        };
        let position = Vec3::new(coord()?, coord()?, coord()?);
        let symbol = parts.next().unwrap_or("X");
        let element = Element::from_symbol(symbol);
        structure.atoms.push(Atom {
            name: format!("{}{}", element.symbol(), i + 1),
            element,
            position,
            residue: 0,
            hetero: true,
            serial: i + 1,
            b_factor: 0.0,
        });
    }

    for (i, line) in lines[4 + n_atoms..4 + n_atoms + n_bonds].iter().enumerate() {
        let parse = |r| -> Result<usize> {
            field(line, r)
                .and_then(|s| s.parse().ok())
                .with_context(|| format!("bad bond line {}", i + 1))
        };
        let (a, b, order) = (parse(0..3)?, parse(3..6)?, parse(6..9)?);
        if a == 0 || b == 0 || a > n_atoms || b > n_atoms {
            bail!("bond line {} references a missing atom", i + 1);
        }
        structure.bonds.push(Bond {
            a: (a - 1) as u32,
            b: (b - 1) as u32,
            order: order as u8,
        });
    }

    Ok(structure)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ethanol, hydrogens suppressed.
    const ETHANOL: &str = "ethanol
  test

  3  2  0  0  0  0  0  0  0  0999 V2000
   -0.0187    1.5258    0.0104 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0021   -0.0041    0.0020 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3234   -0.4558   -0.0026 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
";

    #[test]
    fn parses_ethanol() {
        let s = parse(ETHANOL).unwrap();
        assert_eq!(s.name, "ethanol");
        assert_eq!(s.atoms.len(), 3);
        assert_eq!(s.bonds.len(), 2);
        assert_eq!(s.atoms[2].element.symbol(), "O");
        assert_eq!(s.bonds[1], Bond { a: 1, b: 2, order: 1 });
    }
}
