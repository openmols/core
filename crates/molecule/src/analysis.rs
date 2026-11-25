//! Ensemble and trajectory analysis computed straight from a structure's
//! frames: no trajectory reader needed, since ensembles already load as frames.
//!
//! Superpose the frames first (see [`crate::align::superpose`], or the
//! viewer's "Fit frames") — otherwise overall tumbling swamps the internal
//! motion these measures are meant to capture.

use crate::Structure;
use glam::Vec3;

/// Mean position of every atom across the frames.
pub fn mean_positions(structure: &Structure) -> Vec<Vec3> {
    if structure.frames.is_empty() {
        return structure.atoms.iter().map(|a| a.position).collect();
    }
    let n = structure.frames.len() as f32;
    let mut mean = vec![Vec3::ZERO; structure.atoms.len()];
    for frame in &structure.frames {
        for (m, p) in mean.iter_mut().zip(frame) {
            *m += *p;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    mean
}

/// Per-atom root-mean-square fluctuation, Å. All zeros for a single frame.
pub fn atom_rmsf(structure: &Structure) -> Vec<f32> {
    if structure.frames.len() < 2 {
        return vec![0.0; structure.atoms.len()];
    }
    let mean = mean_positions(structure);
    let n = structure.frames.len() as f32;
    let mut sum = vec![0.0f32; structure.atoms.len()];
    for frame in &structure.frames {
        for ((s, p), m) in sum.iter_mut().zip(frame).zip(&mean) {
            *s += p.distance_squared(*m);
        }
    }
    sum.iter().map(|s| (s / n).sqrt()).collect()
}

/// Per-residue RMSF, Å: the mean over each residue's atoms.
pub fn residue_rmsf(structure: &Structure) -> Vec<f32> {
    let per_atom = atom_rmsf(structure);
    structure
        .residues
        .iter()
        .map(|r| {
            let range = r.atoms.start as usize..r.atoms.end as usize;
            if range.is_empty() {
                return 0.0;
            }
            let count = range.len() as f32;
            per_atom[range].iter().sum::<f32>() / count
        })
        .collect()
}

/// Radius of gyration of the current frame, Å (mass-weighting ignored).
pub fn radius_of_gyration(structure: &Structure) -> f32 {
    let n = structure.atoms.len();
    if n == 0 {
        return 0.0;
    }
    let centre = structure.atoms.iter().map(|a| a.position).sum::<Vec3>() / n as f32;
    let sum: f32 = structure
        .atoms
        .iter()
        .map(|a| a.position.distance_squared(centre))
        .sum();
    (sum / n as f32).sqrt()
}

/// RMSD between two frames of the same structure, Å, without superposing.
pub fn frame_rmsd(structure: &Structure, a: usize, b: usize) -> f32 {
    let (Some(fa), Some(fb)) = (structure.frames.get(a), structure.frames.get(b)) else {
        return 0.0;
    };
    let n = fa.len().min(fb.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = fa[..n]
        .iter()
        .zip(&fb[..n])
        .map(|(p, q)| p.distance_squared(*q))
        .sum();
    (sum / n as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Atom, Element, Residue};

    fn two_atom_ensemble(frames: Vec<Vec<Vec3>>) -> Structure {
        Structure {
            atoms: vec![
                Atom { name: "CA".into(), element: Element::C, position: Vec3::ZERO, residue: 0, hetero: false, serial: 1, b_factor: 0.0 },
                Atom { name: "CB".into(), element: Element::C, position: Vec3::ZERO, residue: 0, hetero: false, serial: 2, b_factor: 0.0 },
            ],
            residues: vec![Residue { name: "ALA".into(), number: 1, insertion_code: None, chain: "A".into(), atoms: 0..2 }],
            frames,
            ..Default::default()
        }
    }

    #[test]
    fn rmsf_measures_spread_about_the_mean() {
        // Atom 0 moves +-1 Å along x; atom 1 never moves.
        let s = two_atom_ensemble(vec![
            vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::ZERO],
            vec![Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO],
        ]);
        let rmsf = atom_rmsf(&s);
        assert!((rmsf[0] - 1.0).abs() < 1e-5, "{:?}", rmsf);
        assert!(rmsf[1].abs() < 1e-6);
        // The residue average of 1.0 and 0.0.
        assert!((residue_rmsf(&s)[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn single_frame_has_no_fluctuation() {
        let s = two_atom_ensemble(Vec::new());
        assert_eq!(atom_rmsf(&s), vec![0.0, 0.0]);
    }

    #[test]
    fn rmsd_between_frames() {
        let s = two_atom_ensemble(vec![
            vec![Vec3::ZERO, Vec3::ZERO],
            vec![Vec3::new(3.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)],
        ]);
        // sqrt((9 + 16) / 2)
        assert!((frame_rmsd(&s, 0, 1) - (12.5f32).sqrt()).abs() < 1e-5);
    }
}
