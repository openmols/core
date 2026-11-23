//! Structure superposition.
//!
//! [`align`] works like PyMOL's `align`: a global sequence alignment (Gotoh /
//! Needleman–Wunsch, BLOSUM62) pairs up Cα atoms, then [`fit_iterative`]
//! superposes them with Kabsch and repeatedly drops outlier pairs.

use crate::Structure;
use anyhow::{Result, bail};
use glam::{Mat3, Mat4, Vec3};

/// How residues of the two structures are paired.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Pairing {
    /// Global sequence alignment of the polymer chains.
    #[default]
    Sequence,
    /// Same chain id and residue number (e.g. two states of one protein).
    ResidueNumber,
}

#[derive(Clone, Debug)]
pub struct AlignOptions {
    pub pairing: Pairing,
    /// Restrict to residues for which these return true (residue indices).
    pub mobile_residues: Option<Vec<usize>>,
    pub target_residues: Option<Vec<usize>>,
    /// Outlier-rejection cycles and distance cutoff (Å), as in PyMOL.
    pub cycles: usize,
    pub cutoff: f32,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            pairing: Pairing::Sequence,
            mobile_residues: None,
            target_residues: None,
            cycles: 5,
            cutoff: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Fit {
    /// Maps mobile coordinates onto target coordinates.
    pub transform: Mat4,
    pub rmsd: f32,
    /// Pairs kept after outlier rejection.
    pub n_aligned: usize,
    /// Pairs before outlier rejection.
    pub n_pairs: usize,
}

/// One-letter code for standard (and a few common modified) residues.
pub fn one_letter(residue: &str) -> Option<u8> {
    Some(match residue {
        "ALA" => b'A',
        "ARG" => b'R',
        "ASN" => b'N',
        "ASP" => b'D',
        "CYS" | "CYX" => b'C',
        "GLN" => b'Q',
        "GLU" => b'E',
        "GLY" => b'G',
        "HIS" | "HID" | "HIE" | "HIP" => b'H',
        "ILE" => b'I',
        "LEU" => b'L',
        "LYS" => b'K',
        "MET" | "MSE" => b'M',
        "PHE" => b'F',
        "PRO" => b'P',
        "SER" => b'S',
        "THR" => b'T',
        "TRP" => b'W',
        "TYR" => b'Y',
        "VAL" => b'V',
        _ => return None,
    })
}

const ORDER: &[u8; 20] = b"ARNDCQEGHILKMFPSTWYV";
#[rustfmt::skip]
const BLOSUM62: [[i8; 20]; 20] = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
];

fn score(a: u8, b: u8) -> i32 {
    let ia = ORDER.iter().position(|&c| c == a);
    let ib = ORDER.iter().position(|&c| c == b);
    match (ia, ib) {
        (Some(i), Some(j)) => BLOSUM62[i][j] as i32,
        _ => -1,
    }
}

/// Global alignment with affine gaps (Gotoh). Returns matched index pairs.
pub fn align_sequences(a: &[u8], b: &[u8]) -> Result<Vec<(usize, usize)>> {
    const GAP_OPEN: i32 = -10;
    const GAP_EXTEND: i32 = -1;
    const NEG: i32 = i32::MIN / 4;
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return Ok(Vec::new());
    }
    if (n + 1) * (m + 1) > 25_000_000 {
        bail!("sequences too long to align ({n} × {m}); restrict to chains or residues");
    }
    let w = m + 1;
    // h: best ending in a match; e: gap in a (horizontal); f: gap in b (vertical).
    let mut h = vec![NEG; (n + 1) * w];
    let mut e = vec![NEG; (n + 1) * w];
    let mut f = vec![NEG; (n + 1) * w];
    h[0] = 0;
    for (j, cell) in e.iter_mut().enumerate().take(m + 1).skip(1) {
        *cell = GAP_OPEN + GAP_EXTEND * (j as i32 - 1);
    }
    for i in 1..=n {
        f[i * w] = GAP_OPEN + GAP_EXTEND * (i as i32 - 1);
    }
    for i in 1..=n {
        for j in 1..=m {
            let k = i * w + j;
            let diag = (i - 1) * w + (j - 1);
            let best_prev = h[diag].max(e[diag]).max(f[diag]);
            h[k] = best_prev + score(a[i - 1], b[j - 1]);
            let left = k - 1;
            e[k] = (h[left].max(f[left]) + GAP_OPEN).max(e[left] + GAP_EXTEND);
            let up = k - w;
            f[k] = (h[up].max(e[up]) + GAP_OPEN).max(f[up] + GAP_EXTEND);
        }
    }

    // Traceback.
    let (mut i, mut j) = (n, m);
    let end = n * w + m;
    let mut state = if h[end] >= e[end] && h[end] >= f[end] {
        0
    } else if e[end] >= f[end] {
        1
    } else {
        2
    };
    let mut pairs = Vec::new();
    while i > 0 && j > 0 {
        let k = i * w + j;
        match state {
            0 => {
                pairs.push((i - 1, j - 1));
                let diag = (i - 1) * w + (j - 1);
                let target = h[k] - score(a[i - 1], b[j - 1]);
                state = if h[diag] == target {
                    0
                } else if e[diag] == target {
                    1
                } else {
                    2
                };
                i -= 1;
                j -= 1;
            }
            1 => {
                let left = k - 1;
                state = if e[k] == e[left] + GAP_EXTEND { 1 } else if h[left] + GAP_OPEN == e[k] { 0 } else { 2 };
                j -= 1;
            }
            _ => {
                let up = k - w;
                state = if f[k] == f[up] + GAP_EXTEND { 2 } else if h[up] + GAP_OPEN == f[k] { 0 } else { 1 };
                i -= 1;
            }
        }
    }
    pairs.reverse();
    Ok(pairs)
}

/// Kabsch: rigid transform minimising RMSD from `mobile` onto `target`.
pub fn superpose(mobile: &[Vec3], target: &[Vec3]) -> (Mat4, f32) {
    let n = mobile.len().min(target.len());
    if n == 0 {
        return (Mat4::IDENTITY, 0.0);
    }
    let cm = mobile[..n].iter().copied().sum::<Vec3>() / n as f32;
    let ct = target[..n].iter().copied().sum::<Vec3>() / n as f32;
    // Covariance H = Σ (m - cm)(t - ct)^T in f64 for stability.
    let mut h = nalgebra::Matrix3::<f64>::zeros();
    for (m, t) in mobile[..n].iter().zip(&target[..n]) {
        let a = (*m - cm).as_dvec3();
        let b = (*t - ct).as_dvec3();
        h += nalgebra::Vector3::new(a.x, a.y, a.z) * nalgebra::Vector3::new(b.x, b.y, b.z).transpose();
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = nalgebra::Matrix3::<f64>::identity();
    if (vt.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = vt.transpose() * d * u.transpose();
    let rot = Mat3::from_cols(
        Vec3::new(r[(0, 0)] as f32, r[(1, 0)] as f32, r[(2, 0)] as f32),
        Vec3::new(r[(0, 1)] as f32, r[(1, 1)] as f32, r[(2, 1)] as f32),
        Vec3::new(r[(0, 2)] as f32, r[(1, 2)] as f32, r[(2, 2)] as f32),
    );
    let transform = Mat4::from_translation(ct) * Mat4::from_mat3(rot) * Mat4::from_translation(-cm);
    let rmsd = rmsd(&mobile[..n], &target[..n], transform);
    (transform, rmsd)
}

pub fn rmsd(mobile: &[Vec3], target: &[Vec3], transform: Mat4) -> f32 {
    let n = mobile.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = mobile
        .iter()
        .zip(target)
        .map(|(m, t)| transform.transform_point3(*m).distance_squared(*t))
        .sum();
    (sum / n as f32).sqrt()
}

/// Kabsch with iterative outlier rejection (PyMOL `align` style).
pub fn fit_iterative(mobile: &[Vec3], target: &[Vec3], cycles: usize, cutoff: f32) -> Fit {
    let n_pairs = mobile.len().min(target.len());
    let mut keep: Vec<usize> = (0..n_pairs).collect();
    let (mut transform, mut rmsd_value) = superpose(mobile, target);
    for _ in 0..cycles {
        let next: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&i| transform.transform_point3(mobile[i]).distance(target[i]) <= cutoff)
            .collect();
        if next.len() == keep.len() || next.len() < 3 {
            break;
        }
        keep = next;
        let m: Vec<Vec3> = keep.iter().map(|&i| mobile[i]).collect();
        let t: Vec<Vec3> = keep.iter().map(|&i| target[i]).collect();
        (transform, rmsd_value) = superpose(&m, &t);
    }
    Fit {
        transform,
        rmsd: rmsd_value,
        n_aligned: keep.len(),
        n_pairs,
    }
}

/// Polymer residues with a Cα, as (residue index, Cα atom index, one-letter code).
fn ca_trace(s: &Structure, restrict: Option<&[usize]>) -> Vec<(usize, usize, u8)> {
    s.residues
        .iter()
        .enumerate()
        .filter(|(i, _)| restrict.is_none_or(|r| r.contains(i)))
        .filter_map(|(i, r)| {
            let ca = s.residue_atom(i, "CA")?;
            if s.atoms[ca].hetero && one_letter(&r.name).is_none() {
                return None;
            }
            Some((i, ca, one_letter(&r.name).unwrap_or(b'X')))
        })
        .collect()
}

/// Cα atom pairs `(mobile atom, target atom)` according to `options.pairing`.
pub fn pair_atoms(mobile: &Structure, target: &Structure, options: &AlignOptions) -> Result<Vec<(usize, usize)>> {
    let a = ca_trace(mobile, options.mobile_residues.as_deref());
    let b = ca_trace(target, options.target_residues.as_deref());
    if a.len() < 3 || b.len() < 3 {
        bail!("need at least 3 Cα atoms on each side (got {} and {})", a.len(), b.len());
    }
    Ok(match options.pairing {
        Pairing::Sequence => {
            let sa: Vec<u8> = a.iter().map(|x| x.2).collect();
            let sb: Vec<u8> = b.iter().map(|x| x.2).collect();
            align_sequences(&sa, &sb)?
                .into_iter()
                .map(|(i, j)| (a[i].1, b[j].1))
                .collect()
        }
        Pairing::ResidueNumber => a
            .iter()
            .filter_map(|(ri, ca, _)| {
                let r = &mobile.residues[*ri];
                b.iter()
                    .find(|(rj, _, _)| {
                        let t = &target.residues[*rj];
                        t.chain == r.chain && t.number == r.number && t.insertion_code == r.insertion_code
                    })
                    .map(|(_, cb, _)| (*ca, *cb))
            })
            .collect(),
    })
}

/// Superpose `mobile` (placed by `mobile_transform`) onto `target` (placed by
/// `target_transform`). Returns the fit whose `transform` maps the mobile's
/// *world* coordinates onto the target's; the new mobile transform is
/// `fit.transform * mobile_transform`.
pub fn align(
    mobile: &Structure,
    mobile_transform: Mat4,
    target: &Structure,
    target_transform: Mat4,
    options: &AlignOptions,
) -> Result<Fit> {
    let pairs = pair_atoms(mobile, target, options)?;
    if pairs.len() < 3 {
        bail!("only {} residue pairs matched; cannot superpose", pairs.len());
    }
    let m: Vec<Vec3> = pairs
        .iter()
        .map(|&(i, _)| mobile_transform.transform_point3(mobile.atoms[i].position))
        .collect();
    let t: Vec<Vec3> = pairs
        .iter()
        .map(|&(_, j)| target_transform.transform_point3(target.atoms[j].position))
        .collect();
    Ok(fit_iterative(&m, &t, options.cycles, options.cutoff))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn blosum_is_symmetric() {
        for (i, row) in BLOSUM62.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_eq!(*value, BLOSUM62[j][i], "{i},{j}");
            }
        }
    }

    #[test]
    fn aligns_sequences_with_gap() {
        let pairs = align_sequences(b"MKTAYIAKQR", b"MKTAYAKQR").unwrap();
        assert_eq!(pairs.len(), 9);
        assert_eq!(pairs[0], (0, 0));
        assert_eq!(*pairs.last().unwrap(), (9, 8));
    }

    #[test]
    fn kabsch_recovers_rigid_transform() {
        let pts: Vec<Vec3> = (0..20)
            .map(|i| {
                let f = i as f32;
                Vec3::new(f.sin() * 5.0, f * 0.7, (f * 0.3).cos() * 3.0)
            })
            .collect();
        let known = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 2.0), Vec3::new(3.0, -7.0, 12.0));
        let moved: Vec<Vec3> = pts.iter().map(|p| known.transform_point3(*p)).collect();
        let (t, r) = superpose(&moved, &pts);
        assert!(r < 1e-3, "rmsd {r}");
        let back = t * known;
        assert!(back.abs_diff_eq(Mat4::IDENTITY, 1e-3), "{back:?}");
    }

    #[test]
    fn outliers_are_rejected() {
        let mut target: Vec<Vec3> = (0..30).map(|i| Vec3::new(i as f32, (i as f32).sin() * 4.0, 0.0)).collect();
        let mobile = target.clone();
        target[5] += Vec3::new(0.0, 0.0, 15.0);
        let fit = fit_iterative(&mobile, &target, 5, 2.0);
        // The outlier tilts the first fit, so a lever-arm endpoint may go too.
        assert!((27..=29).contains(&fit.n_aligned), "kept {}", fit.n_aligned);
        assert!(fit.rmsd < 1e-3, "rmsd {}", fit.rmsd);
    }
}
