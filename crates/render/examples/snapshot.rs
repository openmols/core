//! Headless render to PNG, useful for checking the renderer without a UI.
//!
//! cargo run -p render --example snapshot -- assets/1hsg.pdb out.png [preset]
//! presets: default (cartoon + ligand), surface, spacefill, ball, backbone, ss

use anyhow::{Context, Result};
use molecule::Structure;
use render::{Camera, Coloring, Component, Renderer, Representation as R, Target as T};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().context("usage: snapshot <structure> <out.png> [preset]")?;
    let output = args.next().unwrap_or_else(|| "snapshot.png".into());
    let preset = args.next().unwrap_or_else(|| "default".into());

    let structure = Structure::load(&input)?;
    let ligand = Component::new(T::Ligand, R::BallAndStick, Coloring::Element);
    let components = match preset.as_str() {
        "surface" => vec![Component::new(T::Polymer, R::GaussianSurface, Coloring::Chain), ligand],
        "spacefill" => vec![Component::new(T::All, R::Spacefill, Coloring::Element)],
        "ball" => vec![Component::new(T::All, R::BallAndStick, Coloring::Element)],
        "backbone" => vec![Component::new(T::Polymer, R::Backbone, Coloring::Chain), ligand],
        "ss" => vec![Component::new(T::Polymer, R::Cartoon, Coloring::Secondary), ligand],
        _ => render::default_preset(&structure),
    };
    let mut renderer = Renderer::new()?;
    println!("adapter: {}", renderer.adapter_name());
    println!("{} atoms, {} bonds", structure.atoms.len(), structure.bonds.len());

    let t = std::time::Instant::now();
    renderer.set_scene(&structure, &components, &[]);
    println!("scene build ({preset}): {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);

    // "two": a second, translated and rotated copy as entity 1.
    let (mut center, mut radius) = structure.bounding_sphere();
    if preset == "two" {
        let offset = glam::Vec3::new(radius * 2.2, 0.0, 0.0);
        let pose = glam::Mat4::from_translation(center + offset)
            * glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)
            * glam::Mat4::from_translation(-center);
        let ball = [Component::new(T::All, R::BallAndStick, Coloring::Element)];
        renderer.upload_entity(1, &structure, &ball, &[]);
        renderer.set_entity_pose(1, pose, true);
        center += offset * 0.5;
        radius *= 2.2;
    }
    let mut camera = Camera::default();
    camera.fit(center, radius + 2.0);

    let (w, h) = (1600, 1000);
    renderer.render(&camera, w, h)?; // warm-up (pipeline/driver caches)
    let mut total = std::time::Duration::ZERO;
    let n = 20;
    let mut frame = None;
    for _ in 0..n {
        if preset != "two" {
            camera.orbit(0.02, 0.0);
        }
        let f = renderer.render(&camera, w, h)?;
        total += f.elapsed;
        frame = Some(f);
    }
    let frame = frame.unwrap();
    if preset == "two" {
        for (label, x) in [("left", w / 4), ("right", 3 * w / 4)] {
            let hit = (0..h).step_by(8).find_map(|y| renderer.pick(x, y).ok().flatten());
            println!("{label} column first hit: {hit:?}");
        }
    }
    println!("render + readback at {w}x{h}: {:.2} ms/frame (avg of {n})", total.as_secs_f64() * 1000.0 / n as f64);

    match renderer.pick(w / 2, h / 2)? {
        Some((e, i)) => println!("centre pixel: entity {e}: {}", structure.describe_atom(i)),
        None => println!("centre pixel: background"),
    }

    let mut rgba = frame.bgra;
    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    image::save_buffer(&output, &rgba, w, h, image::ColorType::Rgba8)?;
    println!("wrote {output}");
    Ok(())
}
