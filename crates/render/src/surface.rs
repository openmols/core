//! Gaussian molecular surface.
//!
//! Each atom contributes `exp(-k·(d²/r² − 1))` to a density grid, so an isolated
//! atom's isosurface at density 1 sits exactly on its van der Waals radius and
//! neighbouring atoms blend smoothly. The isosurface is extracted with surface
//! nets; each vertex takes the colour/pick id of the atom contributing most.

use crate::scene::{Mesh, MeshVertex};
use fast_surface_nets::ndshape::{RuntimeShape, Shape};
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};
use glam::Vec3;
use molecule::Structure;
use rayon::prelude::*;

/// Higher = tighter surface hugging the atoms; lower = blobbier.
const SMOOTHNESS: f32 = 1.5;
/// Contributions below this are ignored.
const EPSILON: f32 = 1e-3;
const BASE_SPACING: f32 = 0.7;
const MAX_VOXELS: f32 = 12.0e6;

pub(crate) fn gaussian(
    structure: &Structure,
    selected: &[bool],
    color: &(dyn Fn(usize) -> [u8; 4] + Sync),
) -> Mesh {
    // (position, radius, atom index), sorted by z for slab queries.
    let mut atoms: Vec<(Vec3, f32, u32)> = structure
        .atoms
        .iter()
        .enumerate()
        .filter(|(i, _)| selected[*i])
        .map(|(i, a)| (a.position, a.element.vdw_radius(), i as u32))
        .collect();
    if atoms.is_empty() {
        return Mesh::default();
    }
    atoms.sort_by(|a, b| a.0.z.total_cmp(&b.0.z));

    // d/r beyond which a contribution falls under EPSILON.
    let reach_factor = (1.0 + (1.0 / EPSILON).ln() / SMOOTHNESS).sqrt();
    let max_radius = atoms.iter().map(|a| a.1).fold(0.0f32, f32::max);
    let max_reach = max_radius * reach_factor;

    let (mut lo, mut hi) = (atoms[0].0, atoms[0].0);
    for a in &atoms {
        lo = lo.min(a.0);
        hi = hi.max(a.0);
    }
    let mut spacing = BASE_SPACING;
    let (origin, dims) = loop {
        let pad = Vec3::splat(max_reach + 2.0 * spacing);
        let origin = lo - pad;
        let extent = hi + pad - origin;
        let dims = (extent / spacing).ceil().as_uvec3() + glam::UVec3::ONE;
        if (dims.x as f32 * dims.y as f32 * dims.z as f32) <= MAX_VOXELS {
            break (origin, dims);
        }
        spacing *= 1.25;
    };
    let (nx, ny, nz) = (dims.x as usize, dims.y as usize, dims.z as usize);
    let slice = nx * ny;

    let mut density = vec![0.0f32; slice * nz];
    let mut nearest = vec![u32::MAX; slice * nz];
    let zs: Vec<f32> = atoms.iter().map(|a| a.0.z).collect();

    density
        .par_chunks_mut(slice)
        .zip(nearest.par_chunks_mut(slice))
        .enumerate()
        .for_each(|(k, (dens, near))| {
            let z = origin.z + k as f32 * spacing;
            let mut best = vec![0.0f32; slice];
            let start = zs.partition_point(|&az| az < z - max_reach);
            let end = zs.partition_point(|&az| az <= z + max_reach);
            for &(p, r, idx) in &atoms[start..end] {
                let dz = p.z - z;
                let reach = r * reach_factor;
                let reach2 = reach * reach;
                if dz * dz >= reach2 {
                    continue;
                }
                let inv_r2 = 1.0 / (r * r);
                let rxy = (reach2 - dz * dz).sqrt();
                let x0 = (((p.x - rxy - origin.x) / spacing).floor().max(0.0)) as usize;
                let x1 = (((p.x + rxy - origin.x) / spacing).ceil() as usize).min(nx - 1);
                let y0 = (((p.y - rxy - origin.y) / spacing).floor().max(0.0)) as usize;
                let y1 = (((p.y + rxy - origin.y) / spacing).ceil() as usize).min(ny - 1);
                for j in y0..=y1 {
                    let dy = origin.y + j as f32 * spacing - p.y;
                    let row = j * nx;
                    for i in x0..=x1 {
                        let dx = origin.x + i as f32 * spacing - p.x;
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 >= reach2 {
                            continue;
                        }
                        let v = (-SMOOTHNESS * (d2 * inv_r2 - 1.0)).exp();
                        let cell = row + i;
                        dens[cell] += v;
                        if v > best[cell] {
                            best[cell] = v;
                            near[cell] = idx;
                        }
                    }
                }
            }
        });

    // Signed field: negative inside (density above the iso level of 1).
    let sdf: Vec<f32> = density.par_iter().map(|d| 1.0 - d).collect();
    drop(density);

    let shape = RuntimeShape::<u32, 3>::new([dims.x, dims.y, dims.z]);
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&sdf, &shape, [0; 3], [dims.x - 1, dims.y - 1, dims.z - 1], &mut buffer);

    let vertices = buffer
        .positions
        .par_iter()
        .zip(buffer.normals.par_iter())
        .map(|(p, n)| {
            let local = Vec3::from_array(*p);
            let cell = local.round().as_uvec3().min(dims - glam::UVec3::ONE);
            let mut atom = nearest[shape.linearize(cell.to_array()) as usize];
            if atom == u32::MAX {
                let cell = local.floor().as_uvec3().min(dims - glam::UVec3::ONE);
                atom = nearest[shape.linearize(cell.to_array()) as usize];
            }
            let atom = if atom == u32::MAX { atoms[0].2 } else { atom };
            MeshVertex {
                position: (origin + local * spacing).to_array(),
                normal: Vec3::from_array(*n).normalize_or_zero().to_array(),
                color: color(atom as usize),
                id: atom,
            }
        })
        .collect();

    Mesh {
        vertices,
        indices: buffer.indices,
    }
}
