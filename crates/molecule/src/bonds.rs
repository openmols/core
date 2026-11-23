//! Distance-based bond perception using a uniform spatial grid.

use crate::{Atom, Bond, Element};
use std::collections::HashMap;

/// Extra slack added to the sum of covalent radii (Å), as in Jmol/RasMol.
const TOLERANCE: f32 = 0.45;
/// Pairs closer than this are treated as overlapping alt-locs, not bonds.
const MIN_DISTANCE: f32 = 0.4;

/// Infer single bonds from inter-atomic distances. Runs in roughly O(n).
pub fn perceive_bonds(atoms: &[Atom]) -> Vec<Bond> {
    let max_cov = atoms
        .iter()
        .map(|a| a.element.covalent_radius())
        .fold(0.0f32, f32::max);
    let cell = (2.0 * max_cov + TOLERANCE).max(1.0);

    let key = |p: glam::Vec3| -> (i32, i32, i32) {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    };

    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        grid.entry(key(atom.position)).or_default().push(i as u32);
    }

    let mut bonds = Vec::new();
    for (i, a) in atoms.iter().enumerate() {
        let (cx, cy, cz) = key(a.position);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &j in bucket {
                        if j as usize <= i {
                            continue;
                        }
                        let b = &atoms[j as usize];
                        if a.element == Element::H && b.element == Element::H {
                            continue;
                        }
                        let max = a.element.covalent_radius() + b.element.covalent_radius() + TOLERANCE;
                        let d2 = a.position.distance_squared(b.position);
                        if d2 > MIN_DISTANCE * MIN_DISTANCE && d2 < max * max {
                            bonds.push(Bond { a: i as u32, b: j, order: 1 });
                        }
                    }
                }
            }
        }
    }
    bonds
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn atom(symbol: &str, p: Vec3) -> Atom {
        Atom {
            name: symbol.into(),
            element: Element::from_symbol(symbol),
            position: p,
            residue: 0,
            hetero: false,
            serial: 0,
            b_factor: 0.0,
        }
    }

    #[test]
    fn ethane_like_chain() {
        let atoms = vec![
            atom("C", Vec3::ZERO),
            atom("C", Vec3::new(1.54, 0.0, 0.0)),
            atom("O", Vec3::new(2.97, 0.0, 0.0)),
            atom("C", Vec3::new(10.0, 0.0, 0.0)),
        ];
        let bonds = perceive_bonds(&atoms);
        assert_eq!(bonds.len(), 2);
    }
}
