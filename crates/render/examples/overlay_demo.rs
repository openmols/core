//! Headless picture of the overlay: a docking box with handles, a dashed
//! measurement and a translucent surface coloured by atom.
//!
//! cargo run -p render --example overlay_demo -- assets/1hsg.pdb out.png

use anyhow::{Context, Result};
use glam::Vec3;
use molecule::Structure;
use render::{Camera, Overlay, OverlayCylinder, OverlayFace, OverlaySphere, Renderer, Volume};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().context("usage: overlay_demo <structure> <out.png>")?;
    let output = args.next().unwrap_or_else(|| "overlay.png".into());
    let structure = Structure::load(&input)?;
    let mut renderer = Renderer::new()?;
    renderer.set_scene(&structure, &render::default_preset(&structure), &[]);

    // A translucent surface over chain A, blue near the ligand and orange far from it.
    let ligand: Vec<Vec3> = structure
        .atoms
        .iter()
        .enumerate()
        .filter(|(i, _)| structure.residue_of(*i).name == "MK1")
        .map(|(_, a)| a.position)
        .collect();
    let centre = ligand.iter().sum::<Vec3>() / ligand.len() as f32;
    let n = structure.atoms.len();
    let colors = structure
        .atoms
        .iter()
        .map(|a| {
            let t = (a.position.distance(centre) / 25.0).clamp(0.0, 1.0);
            [(70.0 + 185.0 * t) as u8, (110.0 + 55.0 * t) as u8, (220.0 - 160.0 * t) as u8, 130]
        })
        .collect();
    let volume = Volume {
        selected: (0..n).map(|i| !structure.residue_of(i).is_water() && structure.residue_of(i).chain == "A").collect(),
        colors,
        wire: false,
    };
    renderer.set_entity_volume(0, &structure, Some(&volume));

    let mut camera = Camera::default();
    let (c, r) = structure.bounding_sphere();
    camera.fit(c, r + 2.0);
    camera.orbit(0.5, 0.2);

    // The docking box around the ligand, its faces and handles, and one dashed distance.
    let unit = 2.0 * camera.distance * (camera.fov_y * 0.5).tan() / 900.0;
    let (half, mut overlay) = (Vec3::splat(10.0), Overlay::default());
    let corner = |i: usize| {
        centre + Vec3::new(
            if i & 1 == 0 { -half.x } else { half.x },
            if i & 2 == 0 { -half.y } else { half.y },
            if i & 4 == 0 { -half.z } else { half.z },
        )
    };
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            if i & bit == 0 {
                overlay.cylinders.push(OverlayCylinder {
                    p0: corner(i).to_array(),
                    p1: corner(i | bit).to_array(),
                    radius: 1.8 * unit,
                    color: [0x5f, 0xb4, 0xff, 255],
                    id: 0,
                });
            }
        }
    }
    for (axis, sign) in [(0usize, -1.0f32), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)] {
        let (a1, a2) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut quad = [[0.0f32; 3]; 4];
        for (k, (s1, s2)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
            let mut u = [0.0f32; 3];
            u[axis] = sign * half[axis];
            u[a1] = s1 * half[a1];
            u[a2] = s2 * half[a2];
            quad[k] = (centre + Vec3::from(u)).to_array();
        }
        overlay.faces.push(OverlayFace { corners: quad, color: [0x5f, 0xb4, 0xff, 28] });
        let mut at = centre;
        at[axis] += sign * half[axis];
        overlay.handles.push(OverlaySphere { center: at.to_array(), radius: 5.5 * unit, color: [255, 255, 255, 255], id: 1 });
    }
    overlay.handles.push(OverlaySphere { center: centre.to_array(), radius: 7.0 * unit, color: [255, 159, 67, 255], id: 7 });
    // Dashed distance from the ligand centre to a point 14 Å away.
    let (a, b) = (centre, centre + Vec3::new(14.0, 6.0, -4.0));
    let len = a.distance(b);
    let dir = (b - a) / len;
    let mut at = 0.0;
    while at < len {
        overlay.cylinders.push(OverlayCylinder {
            p0: (a + dir * at).to_array(),
            p1: (a + dir * (at + 0.5).min(len)).to_array(),
            radius: 1.6 * unit,
            color: [0xf2, 0xc1, 0x4e, 255],
            id: 0,
        });
        at += 0.9;
    }
    renderer.set_overlay(&overlay);

    let frame = renderer.render(&camera, 1200, 900)?;
    let mut rgba = Vec::with_capacity(frame.bgra.len());
    for p in frame.bgra.chunks_exact(4) {
        rgba.extend_from_slice(&[p[2], p[1], p[0], 255]);
    }
    image::save_buffer(&output, &rgba, frame.width, frame.height, image::ColorType::Rgba8)?;
    println!("wrote {output}");
    Ok(())
}
