//! Headless GPU checks for the overlay and translucent volumes. Skipped when
//! the machine has no usable adapter.

use glam::Vec3;
use molecule::Structure;
use render::{Camera, OVERLAY_ENTITY, Overlay, OverlayCylinder, OverlayFace, OverlaySphere, Renderer, Volume};
use std::path::PathBuf;

const W: u32 = 400;
const H: u32 = 400;

fn setup() -> Option<(Renderer, Structure, Camera)> {
    // CI machines have no (or only a software) GPU.
    if std::env::var_os("OPENMOLS_SKIP_GPU_TESTS").is_some() {
        return None;
    }
    let renderer = Renderer::new().ok()?;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/1hsg.pdb");
    let structure = Structure::load(path).unwrap();
    let mut renderer = renderer;
    renderer.set_scene(&structure, &render::default_preset(&structure), &[]);
    let (center, radius) = structure.bounding_sphere();
    let mut camera = Camera::default();
    camera.fit(center, radius + 2.0);
    Some((renderer, structure, camera))
}


fn differing(a: &render::Frame, b: &render::Frame) -> usize {
    a.bgra.chunks_exact(4).zip(b.bgra.chunks_exact(4)).filter(|(p, q)| p != q).count()
}

#[test]
fn translucent_volume_blends_and_cannot_be_picked() {
    let Some((mut renderer, structure, camera)) = setup() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let before = renderer.render(&camera, W, H).unwrap();
    let pick_before = renderer.pick(W / 2, H / 2).unwrap();

    let n = structure.atoms.len();
    let volume = Volume {
        selected: (0..n).map(|i| !structure.residue_of(i).is_water()).collect(),
        colors: vec![[255, 80, 40, 110]; n],
        wire: true,
    };
    renderer.set_entity_volume(0, &structure, Some(&volume));
    let after = renderer.render(&camera, W, H).unwrap();
    assert!(differing(&before, &after) > 2000, "the surface should visibly change the image");
    // Blended: the surface is see-through, so the scene underneath still shows
    // (the frame is not simply replaced by one flat colour).
    let distinct = after.bgra.chunks_exact(4).map(|p| (p[0], p[1], p[2])).collect::<std::collections::HashSet<_>>();
    assert!(distinct.len() > 200);
    // The pick target is masked for translucent shapes: clicks go through to the atoms.
    assert_eq!(renderer.pick(W / 2, H / 2).unwrap(), pick_before);

    // Removing it restores the original image.
    renderer.set_entity_volume(0, &structure, None);
    let restored = renderer.render(&camera, W, H).unwrap();
    assert_eq!(differing(&before, &restored), 0);
}

#[test]
fn overlay_shapes_are_depth_correct_and_pickable() {
    let Some((mut renderer, _structure, camera)) = setup() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let baseline = renderer.render(&camera, W, H).unwrap();
    let eye = camera.eye();
    // Just past the near clip plane: in front of everything, but not cut off.
    let forward = (camera.target - eye).normalize();
    let ahead = eye + forward * (camera.clip_planes().0 + 3.0);

    // A handle sphere in front of everything covers the centre pixel and is pickable by id.
    let mut overlay = Overlay::default();
    overlay.spheres.push(OverlaySphere { center: ahead.to_array(), radius: 1.5, color: [255, 255, 255, 255], id: 7 });
    renderer.set_overlay(&overlay);
    let with_sphere = renderer.render(&camera, W, H).unwrap();
    assert!(differing(&baseline, &with_sphere) > 100);
    assert_eq!(renderer.pick(W / 2, H / 2).unwrap(), Some((OVERLAY_ENTITY, 7)));

    // The same sphere placed far behind the molecule is hidden by it: the centre still picks an atom.
    let behind = camera.target + (camera.target - eye) * 2.0;
    let mut overlay = Overlay::default();
    overlay.spheres.push(OverlaySphere { center: behind.to_array(), radius: 2.5, color: [255, 255, 255, 255], id: 7 });
    renderer.set_overlay(&overlay);
    renderer.render(&camera, W, H).unwrap();
    let hit = renderer.pick(W / 2, H / 2).unwrap();
    assert!(hit.is_some_and(|(e, _)| e != OVERLAY_ENTITY), "a sphere behind the protein must not cover it: {hit:?}");

    // A translucent face in front tints the image but does not take clicks.
    let right = camera.rotation * Vec3::X;
    let up = camera.rotation * Vec3::Y;
    let corner = |sx: f32, sy: f32| (ahead + right * sx * 30.0 + up * sy * 30.0).to_array();
    let mut overlay = Overlay::default();
    overlay.faces.push(OverlayFace {
        corners: [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)],
        color: [90, 180, 255, 70],
    });
    overlay.cylinders.push(OverlayCylinder { p0: ahead.to_array(), p1: (ahead + right * 8.0).to_array(), radius: 0.2, color: [255, 200, 0, 255], id: 3 });
    renderer.set_overlay(&overlay);
    let faced = renderer.render(&camera, W, H).unwrap();
    assert!(differing(&baseline, &faced) > 20_000, "the face should tint a large part of the image");
    // The solid edge cylinder through the centre is pickable by its id...
    assert_eq!(renderer.pick(W / 2, H / 2).unwrap(), Some((OVERLAY_ENTITY, 3)));
    // ...but where only the translucent face lies, clicks reach what is behind it.
    let picked = renderer.pick(W / 2, H / 2 + 70).unwrap();
    assert!(
        picked.is_none_or(|(e, _)| e != OVERLAY_ENTITY),
        "a translucent face must not be pickable: {picked:?}"
    );
}
