//! Secondary-structure assignment: a compact DSSP (Kabsch & Sander 1983).
//!
//! Backbone hydrogen bonds are found with the DSSP electrostatic energy.
//! α-helices come from consecutive i→i+4 turns, strands from parallel or
//! antiparallel bridges. 3₁₀/π helices, bends and turns are reported as coil.

use crate::Structure;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum SecondaryStructure {
    #[default]
    Coil,
    Helix,
    Sheet,
}

#[derive(Copy, Clone)]
struct Backbone {
    residue: usize,
    n: Vec3,
    ca: Vec3,
    c: Vec3,
    o: Vec3,
    /// Amide hydrogen (estimated); `None` for chain starts and prolines.
    h: Option<Vec3>,
}

/// DSSP energy threshold, kcal/mol.
const HBOND_CUTOFF: f32 = -0.5;

fn hbond_energy(donor: &Backbone, acceptor: &Backbone) -> f32 {
    let Some(h) = donor.h else { return 0.0 };
    let r_on = donor.n.distance(acceptor.o);
    let r_ch = h.distance(acceptor.c);
    let r_oh = h.distance(acceptor.o);
    let r_cn = donor.n.distance(acceptor.c);
    if r_on < 0.5 || r_ch < 0.5 || r_oh < 0.5 || r_cn < 0.5 {
        return -9.9;
    }
    27.888 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
}

pub fn assign(structure: &Structure) -> Vec<SecondaryStructure> {
    let mut result = vec![SecondaryStructure::Coil; structure.residues.len()];

    // Collect protein backbones; `bb` index is the "sequence" position used below.
    let mut bb: Vec<Backbone> = Vec::new();
    // True if bb[i] and bb[i+1] are peptide-bonded.
    let mut linked: Vec<bool> = Vec::new();
    for (ri, residue) in structure.residues.iter().enumerate() {
        let get = |name| structure.residue_atom(ri, name).map(|i| structure.atoms[i].position);
        let (Some(n), Some(ca), Some(c), Some(o)) = (get("N"), get("CA"), get("C"), get("O")) else {
            continue;
        };
        if structure.atoms[residue.atoms.start as usize].hetero {
            continue;
        }
        let mut h = None;
        if let Some(prev) = bb.last() {
            let prev_res = &structure.residues[prev.residue];
            let connected = prev_res.chain == residue.chain && prev.c.distance(n) < 2.5;
            if let Some(l) = linked.last_mut() {
                *l = connected;
            }
            if connected && residue.name != "PRO" {
                h = Some(n + (prev.c - prev.o).normalize_or_zero());
            }
        }
        bb.push(Backbone { residue: ri, n, ca, c, o, h });
        linked.push(false);
    }
    if bb.len() < 3 {
        return result;
    }

    // Hydrogen bonds: (acceptor CO index, donor NH index).
    const CELL: f32 = 9.0;
    let key = |p: Vec3| ((p.x / CELL).floor() as i32, (p.y / CELL).floor() as i32, (p.z / CELL).floor() as i32);
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, b) in bb.iter().enumerate() {
        grid.entry(key(b.ca)).or_default().push(i);
    }
    let mut hb: HashSet<(usize, usize)> = HashSet::new();
    for (donor, d) in bb.iter().enumerate() {
        if d.h.is_none() {
            continue;
        }
        let (x, y, z) = key(d.ca);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(cell) = grid.get(&(x + dx, y + dy, z + dz)) else { continue };
                    for &acc in cell {
                        if acc.abs_diff(donor) < 2 || bb[acc].ca.distance(d.ca) > CELL {
                            continue;
                        }
                        if hbond_energy(d, &bb[acc]) < HBOND_CUTOFF {
                            hb.insert((acc, donor));
                        }
                    }
                }
            }
        }
    }
    let n = bb.len();
    // Residues i..=j are on one unbroken chain segment.
    let contiguous = |i: usize, j: usize| j < n && (i..j).all(|k| linked[k]);
    let bond = |a: usize, b: usize| hb.contains(&(a, b));

    // α-helix: two consecutive 4-turns at i-1 and i ⇒ residues i..i+3 are helix.
    let turn4 = |i: usize| contiguous(i, i + 4) && bond(i, i + 4);
    let mut helix = vec![false; n];
    for i in 1..n.saturating_sub(4) {
        if turn4(i - 1) && turn4(i) {
            for h in helix.iter_mut().skip(i).take(4) {
                *h = true;
            }
        }
    }

    // β-bridges.
    let mut bridge = vec![false; n];
    for i in 1..n - 1 {
        if !contiguous(i - 1, i + 1) {
            continue;
        }
        let (x, y, z) = key(bb[i].ca);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(cell) = grid.get(&(x + dx, y + dy, z + dz)) else { continue };
                    for &j in cell {
                        if j == 0 || j + 1 >= n || j.abs_diff(i) < 3 || !contiguous(j - 1, j + 1) {
                            continue;
                        }
                        let parallel = (bond(i - 1, j) && bond(j, i + 1))
                            || (bond(j - 1, i) && bond(i, j + 1));
                        let antiparallel = (bond(i, j) && bond(j, i))
                            || (bond(i - 1, j + 1) && bond(j - 1, i + 1));
                        if parallel || antiparallel {
                            bridge[i] = true;
                            bridge[j] = true;
                        }
                    }
                }
            }
        }
    }

    for i in 0..n {
        let ss = if helix[i] {
            SecondaryStructure::Helix
        } else if bridge[i]
            && ((i > 0 && bridge[i - 1] && linked[i - 1]) || (i + 1 < n && bridge[i + 1] && linked[i]))
        {
            SecondaryStructure::Sheet
        } else {
            SecondaryStructure::Coil
        };
        result[bb[i].residue] = ss;
    }
    result
}
