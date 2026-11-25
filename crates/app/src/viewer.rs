//! The 3D viewport: a scene of entities (structures placed with rigid
//! transforms), camera navigation, picking, Blender-style posing (G/R) and
//! frame playback for ensembles/trajectories.
//!
//! The canvas asks the wgpu renderer for a frame during paint (only when
//! something changed), wraps the BGRA pixels in a `RenderImage` and paints it.
//! Mouse handlers are registered on the window during paint so drags keep
//! working when the cursor leaves the viewport.
//!
//! Everything the panels and the LLM tools need is a public method here.

use crate::annotations::{self, Annotation, AtomRef, MeasureMode};
use std::hash::{Hash, Hasher};
use glam::{Mat4, Quat, Vec3};
use gpui::{
    Bounds, Context, Corners, DispatchPhase, EventEmitter, FocusHandle, IntoElement, KeyDownEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, Pixels, Point, Render,
    RenderImage, ScrollWheelEvent, Styled, Task, Window, canvas, div, prelude::*, px, rgb,
};
use molecule::Structure;
use molecule::align::{AlignOptions, Fit};
use project::Id;
use render::{Camera, Coloring, Component, Renderer, Representation, Target};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

const CLICK_SLOP: f32 = 4.0;
const ORBIT_SPEED: f32 = 0.008;
const POSE_ROTATE_SPEED: f32 = 0.01;

/// A structure placed in the scene.
pub struct ViewerEntity {
    pub name: String,
    pub structure: Arc<Structure>,
    pub structure_id: Option<Id>,
    pub components: Vec<Component>,
    pub selection: Vec<usize>,
    pub transform: Mat4,
    pub visible: bool,
    pub frame: usize,
    /// A translucent or wireframe surface around some of this entity's atoms.
    pub volume: Option<render::Volume>,
    volume_dirty: bool,
    undo: Vec<Mat4>,
    redo: Vec<Mat4>,
    geometry_dirty: bool,
}

impl ViewerEntity {
    /// Centre of the entity in world space (bounding-sphere centre).
    pub fn world_center(&self) -> Vec3 {
        self.transform.transform_point3(self.structure.bounding_sphere().0)
    }

    /// Radius of the enclosing sphere, Å. Rigid-body docking compares these
    /// to decide which body is searched around the other.
    pub fn radius(&self) -> f32 {
        self.structure.bounding_sphere().1
    }
}

/// Persisted form of an entity inside a saved view.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityState {
    pub name: String,
    pub structure_id: Option<Id>,
    pub transform: Mat4,
    pub components: Vec<Component>,
    #[serde(default)]
    pub selection: Vec<usize>,
    #[serde(default)]
    pub frame: usize,
    #[serde(default = "yes")]
    pub visible: bool,
}

fn yes() -> bool {
    true
}

/// Everything needed to restore a view: stored in the project as JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViewState {
    pub camera: Camera,
    #[serde(default)]
    pub entities: Vec<EntityState>,
    #[serde(default)]
    pub active: Option<usize>,
    // Legacy single-structure views (before multi-entity scenes).
    #[serde(default)]
    pub components: Vec<Component>,
    #[serde(default)]
    pub selection: Vec<usize>,
    /// Measurements, contact sets and the docking box.
    #[serde(default)]
    pub annotations: Vec<Annotation>,
    #[serde(default)]
    pub annotation_hidden: Vec<bool>,
}

pub enum ViewerEvent {
    /// Something was picked (or the selection cleared) by clicking.
    Picked(Option<(usize, usize)>),
    /// Entities were added, removed, renamed or re-posed.
    SceneChanged,
}

impl EventEmitter<ViewerEvent> for Viewer {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PoseMode {
    Grab,
    Rotate,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
    Z,
}

struct Pose {
    mode: PoseMode,
    entity: usize,
    start_transform: Mat4,
    start_mouse: Option<Point<Pixels>>,
    mouse: Option<Point<Pixels>>,
    /// Movement along the view direction (wheel), Å.
    depth: f32,
    axis: Option<(Axis, bool)>,
}

/// Dragging one of the docking box's handles.
struct BoxDrag {
    handle: u32,
    center0: Vec3,
    size0: Vec3,
    /// Where on the handle's axis (or the camera plane) the drag began.
    anchor: f32,
    anchor_point: Vec3,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MenuAction {
    Measure(MeasureMode),
    AddToMeasurement,
    SelectResidue,
    CenterHere,
    BoxHere,
    PlaceBox,
    ToggleVolume,
    ClearMeasurements,
}

/// The right-click menu, in viewport coordinates.
struct Menu {
    at: Point<Pixels>,
    hit: Option<(usize, usize)>,
    items: Vec<(String, MenuAction)>,
    hover: Option<usize>,
}

const MENU_WIDTH: f32 = 250.0;
const MENU_ROW: f32 = 24.0;
const MENU_PAD: f32 = 4.0;
/// Edge of a docking box made by clicking an atom, Å.
const DEFAULT_BOX: f32 = 16.0;
const MIN_BOX: f32 = 4.0;

struct Drag {
    button: MouseButton,
    last: Point<Pixels>,
    travelled: f32,
}

pub struct Viewer {
    renderer: Result<Renderer, String>,
    entities: Vec<ViewerEntity>,
    active: Option<usize>,
    camera: Camera,
    pub error: Option<String>,
    focus: FocusHandle,

    dirty: bool,
    /// Renderer slots must be rebuilt from scratch (after removals).
    resync: bool,
    frame: Option<Arc<RenderImage>>,
    frame_size: (u32, u32),
    retired: Vec<Arc<RenderImage>>,
    last_render: Duration,

    viewport: Bounds<Pixels>,
    drag: Option<Drag>,
    hover: Option<(usize, usize, Point<Pixels>)>,
    pose: Option<Pose>,
    playing: Option<Task<()>>,
    box_drag: Option<BoxDrag>,
    /// The next click on an atom centres a new docking box there.
    placing_box: bool,
    menu: Option<Menu>,
    /// Skip the mouse-up that ends a click already used by the menu.
    swallow_up: bool,
    overlay_next: Option<render::Overlay>,
    overlay_fingerprint: u64,
    annotations: Vec<Annotation>,
    /// Per annotation: switched off with its eye. Missing entries mean shown.
    annotation_hidden: Vec<bool>,
    /// Click-to-measure: what a click builds, and the atoms picked so far.
    measure: Option<MeasureMode>,
    pending: Vec<AtomRef>,
}

impl Viewer {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            renderer: Renderer::new().map_err(|e| format!("{e:#}")),
            entities: Vec::new(),
            active: None,
            camera: Camera::default(),
            error: None,
            focus: cx.focus_handle(),
            dirty: true,
            resync: false,
            frame: None,
            frame_size: (0, 0),
            retired: Vec::new(),
            last_render: Duration::ZERO,
            viewport: Bounds::default(),
            drag: None,
            hover: None,
            pose: None,
            playing: None,
            box_drag: None,
            placing_box: false,
            menu: None,
            swallow_up: false,
            overlay_next: None,
            overlay_fingerprint: 0,
            annotations: Vec::new(),
            annotation_hidden: Vec::new(),
            measure: None,
            pending: Vec::new(),
        }
    }

    // ------------------------------------------------------------- accessors

    pub fn entities(&self) -> &[ViewerEntity] {
        &self.entities
    }

    pub fn active_index(&self) -> Option<usize> {
        self.active.filter(|&i| i < self.entities.len())
    }

    pub fn active(&self) -> Option<&ViewerEntity> {
        self.active_index().map(|i| &self.entities[i])
    }

    /// The active entity's structure (most tools operate on it).
    pub fn structure(&self) -> Option<&Arc<Structure>> {
        self.active().map(|e| &e.structure)
    }

    pub fn structure_id(&self) -> Option<Id> {
        self.active().and_then(|e| e.structure_id)
    }

    pub fn components(&self) -> &[Component] {
        self.active().map(|e| e.components.as_slice()).unwrap_or(&[])
    }

    pub fn selection(&self) -> &[usize] {
        self.active().map(|e| e.selection.as_slice()).unwrap_or(&[])
    }

    pub fn is_playing(&self) -> bool {
        self.playing.is_some()
    }

    pub fn adapter_name(&self) -> String {
        match &self.renderer {
            Ok(r) => r.adapter_name(),
            Err(e) => format!("unavailable: {e}"),
        }
    }

    pub fn last_render(&self) -> Duration {
        self.last_render
    }

    pub fn describe(&self, entity: usize, atom: usize) -> String {
        let Some(e) = self.entities.get(entity) else { return String::new() };
        let label = e.structure.describe_atom(atom);
        if self.entities.len() > 1 { format!("{} · {label}", e.name) } else { label }
    }

    /// Find an entity by index or (case-insensitive) name.
    pub fn find_entity(&self, key: &str) -> Option<usize> {
        key.parse::<usize>()
            .ok()
            .filter(|&i| i < self.entities.len())
            .or_else(|| self.entities.iter().position(|e| e.name.eq_ignore_ascii_case(key)))
    }

    // ------------------------------------------------------------ scene edits

    fn mark(&mut self, index: usize) {
        if let Some(e) = self.entities.get_mut(index) {
            e.geometry_dirty = true;
        }
        self.dirty = true;
    }

    fn unique_name(&self, base: &str) -> String {
        if !self.entities.iter().any(|e| e.name == base) {
            return base.to_string();
        }
        (2..)
            .map(|i| format!("{base} ({i})"))
            .find(|n| !self.entities.iter().any(|e| &e.name == n))
            .unwrap()
    }

    /// Add a structure to the scene and make it active. Returns its index.
    pub fn add_structure(&mut self, structure: Structure, id: Option<Id>, cx: &mut Context<Self>) -> usize {
        let first = self.entities.is_empty();
        let name = self.unique_name(if structure.name.is_empty() { "structure" } else { &structure.name });
        let components = render::default_preset(&structure);
        self.entities.push(ViewerEntity {
            name,
            structure: Arc::new(structure),
            structure_id: id,
            components,
            selection: Vec::new(),
            transform: Mat4::IDENTITY,
            visible: true,
            frame: 0,
            volume: None,
            volume_dirty: false,
            undo: Vec::new(),
            redo: Vec::new(),
            geometry_dirty: true,
        });
        let index = self.entities.len() - 1;
        self.active = Some(index);
        self.error = None;
        if first {
            self.reset_camera();
        }
        self.dirty = true;
        cx.emit(ViewerEvent::SceneChanged);
        cx.notify();
        index
    }

    /// Replace the whole scene with one structure.
    pub fn set_structure(&mut self, structure: Structure, id: Option<Id>, cx: &mut Context<Self>) {
        self.clear(cx);
        self.add_structure(structure, id, cx);
    }

    pub fn clear(&mut self, cx: &mut Context<Self>) {
        self.stop_playback();
        self.pose = None;
        self.entities.clear();
        self.active = None;
        self.hover = None;
        self.resync = true;
        self.dirty = true;
        cx.emit(ViewerEvent::SceneChanged);
        cx.notify();
    }

    pub fn remove_entity(&mut self, index: usize, cx: &mut Context<Self>) {
        if index >= self.entities.len() {
            return;
        }
        self.pose = None;
        self.entities.remove(index);
        self.active = match self.active {
            Some(a) if a == index => self.entities.len().checked_sub(1),
            Some(a) if a > index => Some(a - 1),
            other => other,
        };
        self.hover = None;
        self.resync = true;
        self.dirty = true;
        cx.emit(ViewerEvent::SceneChanged);
        cx.notify();
    }

    pub fn set_active(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.entities.len() {
            self.active = Some(index);
            cx.emit(ViewerEvent::SceneChanged);
            cx.notify();
        }
    }

    pub fn set_visible(&mut self, index: usize, visible: bool, cx: &mut Context<Self>) {
        if let Some(e) = self.entities.get_mut(index) {
            e.visible = visible;
            self.dirty = true;
            cx.notify();
        }
    }

    pub fn set_components(&mut self, components: Vec<Component>, cx: &mut Context<Self>) {
        if let Some(i) = self.active_index() {
            self.entities[i].components = components;
            self.mark(i);
            cx.notify();
        }
    }

    pub fn add_component(&mut self, component: Component, cx: &mut Context<Self>) {
        if let Some(i) = self.active_index()
            && !self.entities[i].components.contains(&component)
        {
            self.entities[i].components.push(component);
            self.mark(i);
            cx.notify();
        }
    }

    pub fn remove_component(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(i) = self.active_index()
            && index < self.entities[i].components.len()
        {
            self.entities[i].components.remove(index);
            self.mark(i);
            cx.notify();
        }
    }

    pub fn set_water_visible(&mut self, visible: bool, cx: &mut Context<Self>) {
        let Some(i) = self.active_index() else { return };
        let e = &mut self.entities[i];
        e.components.retain(|c| c.target != Target::Water);
        if visible {
            e.components
                .push(Component::new(Target::Water, Representation::BallAndStick, Coloring::Element));
        }
        self.mark(i);
        cx.notify();
    }

    pub fn water_visible(&self) -> bool {
        self.components().iter().any(|c| c.target == Target::Water)
    }

    /// Replace the active entity's selection. Selected atoms are shown as ball
    /// & stick on top of the other representations (Mol*'s focus style).
    pub fn set_selection(&mut self, atoms: Vec<usize>, cx: &mut Context<Self>) {
        if let Some(i) = self.active_index() {
            self.set_entity_selection(i, atoms, cx);
        }
    }

    fn set_entity_selection(&mut self, index: usize, atoms: Vec<usize>, cx: &mut Context<Self>) {
        let focus = Component::new(Target::Selection, Representation::BallAndStick, Coloring::Element);
        let e = &mut self.entities[index];
        e.selection = atoms;
        if e.selection.is_empty() {
            e.components.retain(|c| *c != focus);
        } else if !e.components.contains(&focus) {
            e.components.push(focus);
        }
        self.mark(index);
        cx.notify();
    }

    // ---------------------------------------------------------------- frames

    pub fn set_frame(&mut self, index: usize, frame: usize, cx: &mut Context<Self>) {
        let Some(e) = self.entities.get_mut(index) else { return };
        let frame = frame.min(e.structure.frame_count() - 1);
        if frame != e.frame {
            e.frame = frame;
            Arc::make_mut(&mut e.structure).set_frame(frame);
            self.mark(index);
            cx.notify();
        }
    }

    pub fn toggle_playback(&mut self, cx: &mut Context<Self>) {
        if self.playing.is_some() {
            self.stop_playback();
            // Rebuild surfaces skipped during playback.
            if let Some(i) = self.active_index() {
                self.mark(i);
            }
            cx.notify();
            return;
        }
        let Some(index) = self.active_index() else { return };
        if self.entities[index].structure.frame_count() < 2 {
            return;
        }
        self.playing = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(66)).await;
                let ok = this.update(cx, |this, cx| {
                    let Some(e) = this.entities.get(index) else { return false };
                    let next = (e.frame + 1) % e.structure.frame_count();
                    this.set_frame(index, next, cx);
                    true
                });
                if !matches!(ok, Ok(true)) {
                    break;
                }
            }
        }));
        cx.notify();
    }

    fn stop_playback(&mut self) {
        self.playing = None;
    }

    // ------------------------------------------------------------ transforms

    fn push_undo(&mut self, index: usize, previous: Mat4) {
        let e = &mut self.entities[index];
        e.undo.push(previous);
        e.redo.clear();
    }

    /// Set an entity's transform (recorded for undo).
    pub fn set_transform(&mut self, index: usize, transform: Mat4, cx: &mut Context<Self>) {
        let Some(previous) = self.entities.get(index).map(|e| e.transform) else { return };
        self.push_undo(index, previous);
        self.entities[index].transform = transform;
        self.dirty = true;
        cx.emit(ViewerEvent::SceneChanged);
        cx.notify();
    }

    /// Translate (Å, world axes) and rotate (degrees about world X/Y/Z, around
    /// the entity centre).
    pub fn move_entity(&mut self, index: usize, translate: Vec3, rotate_deg: Vec3, cx: &mut Context<Self>) {
        let Some(e) = self.entities.get(index) else { return };
        let pivot = e.world_center();
        let r = Quat::from_euler(
            glam::EulerRot::XYZ,
            rotate_deg.x.to_radians(),
            rotate_deg.y.to_radians(),
            rotate_deg.z.to_radians(),
        );
        let delta = Mat4::from_translation(pivot + translate) * Mat4::from_quat(r) * Mat4::from_translation(-pivot);
        let t = delta * e.transform;
        self.set_transform(index, t, cx);
    }

    pub fn undo(&mut self, cx: &mut Context<Self>) {
        let Some(i) = self.active_index() else { return };
        let e = &mut self.entities[i];
        if let Some(prev) = e.undo.pop() {
            e.redo.push(e.transform);
            e.transform = prev;
            self.dirty = true;
            cx.notify();
        }
    }

    pub fn redo(&mut self, cx: &mut Context<Self>) {
        let Some(i) = self.active_index() else { return };
        let e = &mut self.entities[i];
        if let Some(next) = e.redo.pop() {
            e.undo.push(e.transform);
            e.transform = next;
            self.dirty = true;
            cx.notify();
        }
    }

    /// Bake the transform into the coordinates (all frames) and reset it.
    pub fn apply_transform(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(e) = self.entities.get_mut(index) else { return };
        let t = e.transform;
        let s = Arc::make_mut(&mut e.structure);
        for atom in &mut s.atoms {
            atom.position = t.transform_point3(atom.position);
        }
        for frame in &mut s.frames {
            for p in frame.iter_mut() {
                *p = t.transform_point3(*p);
            }
        }
        e.transform = Mat4::IDENTITY;
        e.undo.clear();
        e.redo.clear();
        self.mark(index);
        cx.notify();
    }

    /// Superpose entity `mobile` onto `target`; the move can be undone.
    pub fn align(&mut self, mobile: usize, target: usize, options: &AlignOptions, cx: &mut Context<Self>) -> anyhow::Result<Fit> {
        anyhow::ensure!(mobile != target, "pick two different entities");
        let (m, t) = (
            self.entities.get(mobile).ok_or_else(|| anyhow::anyhow!("no entity {mobile}"))?,
            self.entities.get(target).ok_or_else(|| anyhow::anyhow!("no entity {target}"))?,
        );
        let fit = molecule::align::align(&m.structure, m.transform, &t.structure, t.transform, options)?;
        let new = fit.transform * m.transform;
        self.set_transform(mobile, new, cx);
        Ok(fit)
    }

    /// Superpose every frame of an entity onto `reference` frame (Cα fit).
    pub fn fit_frames(&mut self, index: usize, reference: usize, cx: &mut Context<Self>) -> anyhow::Result<f32> {
        let e = self.entities.get_mut(index).ok_or_else(|| anyhow::anyhow!("no entity {index}"))?;
        let s = Arc::make_mut(&mut e.structure);
        anyhow::ensure!(s.frames.len() > 1, "entity has a single frame");
        let ca: Vec<usize> = (0..s.residues.len()).filter_map(|r| s.residue_atom(r, "CA")).collect();
        anyhow::ensure!(ca.len() >= 3, "fewer than 3 Cα atoms");
        let reference = reference.min(s.frames.len() - 1);
        let target: Vec<Vec3> = ca.iter().map(|&a| s.frames[reference][a]).collect();
        let mut total = 0.0;
        for frame in &mut s.frames {
            let mobile: Vec<Vec3> = ca.iter().map(|&a| frame[a]).collect();
            let (t, rmsd) = molecule::align::superpose(&mobile, &target);
            total += rmsd;
            for p in frame.iter_mut() {
                *p = t.transform_point3(*p);
            }
        }
        let n = s.frames.len();
        let current = e.frame;
        s.set_frame(current);
        self.mark(index);
        cx.notify();
        Ok(total / n as f32)
    }

    // ---------------------------------------------------------------- camera

    fn scene_sphere(&self) -> Option<(Vec3, f32)> {
        let spheres: Vec<(Vec3, f32)> = self
            .entities
            .iter()
            .filter(|e| e.visible)
            .map(|e| {
                let (c, r) = e.structure.bounding_sphere();
                (e.transform.transform_point3(c), r)
            })
            .collect();
        let first = spheres.first()?;
        let (mut lo, mut hi) = (first.0 - Vec3::splat(first.1), first.0 + Vec3::splat(first.1));
        for (c, r) in &spheres {
            lo = lo.min(*c - Vec3::splat(*r));
            hi = hi.max(*c + Vec3::splat(*r));
        }
        let center = (lo + hi) * 0.5;
        let radius = spheres
            .iter()
            .map(|(c, r)| c.distance(center) + r)
            .fold(0.0f32, f32::max);
        Some((center, radius))
    }

    fn reset_camera(&mut self) {
        if let Some((center, radius)) = self.scene_sphere() {
            self.camera = Camera::default();
            self.camera.fit(center, radius + 2.0);
        }
        self.dirty = true;
    }

    pub fn reset_view(&mut self, cx: &mut Context<Self>) {
        self.reset_camera();
        cx.notify();
    }

    /// Centre the camera on atoms of the active entity, keeping orientation.
    pub fn focus_atoms(&mut self, atoms: &[usize], cx: &mut Context<Self>) {
        let Some(e) = self.active() else { return };
        let points: Vec<Vec3> = atoms
            .iter()
            .filter_map(|&i| e.structure.atoms.get(i).map(|a| e.transform.transform_point3(a.position)))
            .collect();
        let Some(&first) = points.first() else { return };
        let (lo, hi) = points
            .iter()
            .fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let center = (lo + hi) * 0.5;
        let radius = points.iter().map(|p| p.distance(center)).fold(0.0f32, f32::max) + 4.0;
        let rotation = self.camera.rotation;
        let scene_radius = self.scene_sphere().map(|s| s.1 + 2.0).unwrap_or(radius);
        self.camera.fit(center, radius);
        self.camera.rotation = rotation;
        self.camera.scene_radius = scene_radius.max(radius);
        self.dirty = true;
        cx.notify();
    }

    // ----------------------------------------------------------- annotations

    pub fn annotations(&self) -> &[Annotation] {
        &self.annotations
    }

    pub fn annotation_hidden(&self, index: usize) -> bool {
        self.annotation_hidden.get(index).copied().unwrap_or(false)
    }

    /// Show or hide one measurement or object with its eye.
    pub fn toggle_annotation_hidden(&mut self, index: usize, cx: &mut Context<Self>) {
        if index >= self.annotations.len() {
            return;
        }
        if self.annotation_hidden.len() <= index {
            self.annotation_hidden.resize(index + 1, false);
        }
        self.annotation_hidden[index] = !self.annotation_hidden[index];
        self.dirty = true;
        cx.notify();
    }

    pub fn set_annotation_hidden(&mut self, hidden: Vec<bool>, cx: &mut Context<Self>) {
        self.annotation_hidden = hidden;
        self.dirty = true;
        cx.notify();
    }


    pub fn set_annotations(&mut self, annotations: Vec<Annotation>, cx: &mut Context<Self>) {
        self.annotations = annotations;
        self.annotation_hidden.clear();
        self.dirty = true;
        cx.notify();
    }

    pub fn add_annotation(&mut self, annotation: Annotation, cx: &mut Context<Self>) {
        self.annotations.push(annotation);
        self.dirty = true;
        cx.notify();
    }

    pub fn remove_annotation(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.annotations.len() {
            self.annotations.remove(index);
            if index < self.annotation_hidden.len() {
                self.annotation_hidden.remove(index);
            }
            self.dirty = true;
            cx.notify();
        }
    }

    pub fn clear_annotations(&mut self, cx: &mut Context<Self>) {
        self.annotations.clear();
        self.annotation_hidden.clear();
        self.pending.clear();
        self.dirty = true;
        cx.notify();
    }

    /// The docking box: the most recently added one.
    pub fn dock_box(&self) -> Option<([f32; 3], [f32; 3])> {
        self.annotations.iter().rev().find_map(|a| match a {
            Annotation::DockBox { center, size } => Some((*center, *size)),
            _ => None,
        })
    }

    /// Replace any docking box with this one.
    pub fn set_dock_box(&mut self, center: [f32; 3], size: [f32; 3], cx: &mut Context<Self>) {
        let keep: Vec<bool> = self.annotations.iter().map(|a| !matches!(a, Annotation::DockBox { .. })).collect();
        let mut kept = keep.iter();
        self.annotation_hidden.retain(|_| kept.next().copied().unwrap_or(true));
        self.annotations.retain(|a| !matches!(a, Annotation::DockBox { .. }));
        self.annotations.push(Annotation::DockBox { center, size });
        self.dirty = true;
        cx.notify();
    }

    /// Wrap atoms of `entity` in a translucent surface (`None` removes it).
    pub fn set_volume(&mut self, entity: usize, volume: Option<render::Volume>, cx: &mut Context<Self>) {
        if let Some(e) = self.entities.get_mut(entity) {
            e.volume = volume;
            e.volume_dirty = true;
            self.dirty = true;
            cx.notify();
        }
    }

    /// A translucent surface over the entity's selection, or all of it without water.
    pub fn default_volume(&self, entity: usize) -> Option<render::Volume> {
        let e = self.entities.get(entity)?;
        let n = e.structure.atoms.len();
        let selected: Vec<bool> = (0..n)
            .map(|i| if e.selection.is_empty() { !e.structure.residue_of(i).is_water() } else { e.selection.contains(&i) })
            .collect();
        Some(render::Volume { selected, colors: vec![[110, 170, 255, 105]; n], wire: false })
    }

    pub fn placing_box(&self) -> bool {
        self.placing_box
    }

    /// The next click on an atom centres the docking box on it.
    pub fn set_placing_box(&mut self, on: bool, cx: &mut Context<Self>) {
        self.placing_box = on;
        if on {
            self.measure = None;
            self.pending.clear();
        }
        self.dirty = true;
        cx.notify();
    }

    /// A box of the default size centred on an atom.
    fn box_at_atom(&mut self, entity: usize, atom: usize, cx: &mut Context<Self>) {
        let Some(e) = self.entities.get(entity) else { return };
        let Some(a) = e.structure.atoms.get(atom) else { return };
        let center = e.transform.transform_point3(a.position);
        self.set_dock_box(center.to_array(), [DEFAULT_BOX; 3], cx);
        cx.emit(ViewerEvent::SceneChanged);
    }

    pub fn measure_mode(&self) -> Option<MeasureMode> {
        self.measure
    }

    pub fn set_measure_mode(&mut self, mode: Option<MeasureMode>, cx: &mut Context<Self>) {
        self.measure = mode;
        self.pending.clear();
        self.placing_box = false;
        self.dirty = true;
        cx.notify();
    }

    pub fn set_camera(&mut self, camera: Camera, cx: &mut Context<Self>) {
        self.camera = camera;
        self.dirty = true;
        cx.notify();
    }

    pub fn view_state(&self) -> ViewState {
        ViewState {
            camera: self.camera.clone(),
            entities: self
                .entities
                .iter()
                .map(|e| EntityState {
                    name: e.name.clone(),
                    structure_id: e.structure_id,
                    transform: e.transform,
                    components: e.components.clone(),
                    selection: e.selection.clone(),
                    frame: e.frame,
                    visible: e.visible,
                })
                .collect(),
            active: self.active,
            components: Vec::new(),
            selection: Vec::new(),
            annotations: self.annotations.clone(),
            annotation_hidden: self.annotation_hidden.clone(),
        }
    }

    /// Restore per-entity state for an entity already added to the scene.
    pub fn apply_entity_state(&mut self, index: usize, state: &EntityState, cx: &mut Context<Self>) {
        if let Some(e) = self.entities.get_mut(index) {
            e.name = state.name.clone();
            e.transform = state.transform;
            e.components = state.components.clone();
            e.selection = state.selection.clone();
            e.visible = state.visible;
        }
        self.set_frame(index, state.frame, cx);
        self.mark(index);
    }

    pub fn set_active_index(&mut self, active: Option<usize>) {
        self.active = active.filter(|&a| a < self.entities.len()).or(self.active);
    }

    // ---------------------------------------------------------------- posing

    pub fn start_pose(&mut self, mode: PoseMode, cx: &mut Context<Self>) {
        let Some(i) = self.active_index() else { return };
        if let Some(pose) = &self.pose {
            // Switching G <-> R keeps the original start transform.
            let start = pose.start_transform;
            self.entities[i].transform = start;
        }
        self.drag = None;
        self.pose = Some(Pose {
            mode,
            entity: i,
            start_transform: self.entities[i].transform,
            start_mouse: None,
            mouse: None,
            depth: 0.0,
            axis: None,
        });
        cx.notify();
    }

    fn set_pose_axis(&mut self, axis: Axis, cx: &mut Context<Self>) {
        if let Some(pose) = &mut self.pose {
            pose.axis = match pose.axis {
                Some((a, false)) if a == axis => Some((axis, true)),
                Some((a, true)) if a == axis => None,
                _ => Some((axis, false)),
            };
            self.update_pose(cx);
        }
    }

    fn pose_transform(&self, pose: &Pose) -> Mat4 {
        let start = pose.start_transform;
        let (Some(a), Some(b)) = (pose.start_mouse, pose.mouse) else {
            return Mat4::from_translation(self.camera_dir() * pose.depth) * start;
        };
        let (dx, dy) = (f32::from(b.x - a.x), f32::from(b.y - a.y));
        let rot = self.camera.rotation;
        let (right, up) = (rot * Vec3::X, rot * Vec3::Y);
        let axis_vec = pose.axis.map(|(axis, local)| {
            let v = match axis {
                Axis::X => Vec3::X,
                Axis::Y => Vec3::Y,
                Axis::Z => Vec3::Z,
            };
            if local {
                start.transform_vector3(v).normalize_or(v)
            } else {
                v
            }
        });
        match pose.mode {
            PoseMode::Grab => {
                let h = f32::from(self.viewport.size.height).max(1.0);
                let k = 2.0 * self.camera.distance * (self.camera.fov_y * 0.5).tan() / h;
                let mut delta = (right * dx - up * dy) * k + self.camera_dir() * pose.depth;
                if let Some(axis) = axis_vec {
                    delta = axis * delta.dot(axis);
                }
                Mat4::from_translation(delta) * start
            }
            PoseMode::Rotate => {
                let pivot = start.transform_point3(self.entities[pose.entity].structure.bounding_sphere().0);
                let q = match axis_vec {
                    Some(axis) => Quat::from_axis_angle(axis, dx * POSE_ROTATE_SPEED),
                    None => Quat::from_axis_angle(up, dx * POSE_ROTATE_SPEED)
                        * Quat::from_axis_angle(right, dy * POSE_ROTATE_SPEED),
                };
                Mat4::from_translation(pivot) * Mat4::from_quat(q) * Mat4::from_translation(-pivot) * start
            }
        }
    }

    /// Unit vector pointing from the camera into the scene.
    fn camera_dir(&self) -> Vec3 {
        -(self.camera.rotation * Vec3::Z)
    }

    fn update_pose(&mut self, cx: &mut Context<Self>) {
        let Some(pose) = &self.pose else { return };
        let t = self.pose_transform(pose);
        let i = pose.entity;
        if let Some(e) = self.entities.get_mut(i) {
            e.transform = t;
        }
        self.dirty = true;
        cx.notify();
    }

    fn confirm_pose(&mut self, cx: &mut Context<Self>) {
        if let Some(pose) = self.pose.take() {
            if self.entities.get(pose.entity).is_some_and(|e| e.transform != pose.start_transform) {
                self.push_undo(pose.entity, pose.start_transform);
            }
            cx.emit(ViewerEvent::SceneChanged);
            cx.notify();
        }
    }

    fn cancel_pose(&mut self, cx: &mut Context<Self>) {
        if let Some(pose) = self.pose.take() {
            if let Some(e) = self.entities.get_mut(pose.entity) {
                e.transform = pose.start_transform;
            }
            self.dirty = true;
            cx.notify();
        }
    }

    fn pose_badge(&self) -> Option<String> {
        let pose = self.pose.as_ref()?;
        let e = self.entities.get(pose.entity)?;
        let mode = match pose.mode {
            PoseMode::Grab => "GRAB",
            PoseMode::Rotate => "ROTATE",
        };
        let axis = match pose.axis {
            Some((a, local)) => format!(" · {:?} {}", a, if local { "(local)" } else { "(global)" }),
            None => String::new(),
        };
        let delta = match pose.mode {
            PoseMode::Grab => format!(
                "{:.2} Å",
                e.transform.w_axis.truncate().distance(pose.start_transform.w_axis.truncate())
            ),
            PoseMode::Rotate => {
                let (_, r0, _) = pose.start_transform.to_scale_rotation_translation();
                let (_, r1, _) = e.transform.to_scale_rotation_translation();
                format!("{:.1}°", (r1 * r0.inverse()).angle_between(Quat::IDENTITY).to_degrees())
            }
        };
        Some(format!(
            "{mode} {}{axis} · {delta}   —   X/Y/Z axis · wheel depth · click/Enter confirm · Esc/right-click cancel",
            e.name
        ))
    }


    // ------------------------------------------------- box handles and menu

    /// The ray through a window position: `(origin, direction)` in world space.
    fn ray_at(&self, position: Point<Pixels>) -> Option<(Vec3, Vec3)> {
        let local = position - self.viewport.origin;
        let (w, h) = (f32::from(self.viewport.size.width), f32::from(self.viewport.size.height));
        if w < 1.0 || h < 1.0 {
            return None;
        }
        let ndc_x = 2.0 * f32::from(local.x) / w - 1.0;
        let ndc_y = 1.0 - 2.0 * f32::from(local.y) / h;
        let inverse = (self.camera.projection(w / h) * self.camera.view()).inverse();
        let near = inverse.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inverse.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize_or_zero();
        (dir != Vec3::ZERO).then_some((near, dir))
    }

    /// Where along the line `point + t·axis` the ray passes closest.
    fn closest_on_axis(point: Vec3, axis: Vec3, origin: Vec3, dir: Vec3) -> Option<f32> {
        let w0 = point - origin;
        let (a, b, c) = (axis.dot(axis), axis.dot(dir), dir.dot(dir));
        let (d, e) = (axis.dot(w0), dir.dot(w0));
        let denom = a * c - b * b;
        (denom.abs() > 1e-4).then(|| (b * e - c * d) / denom)
    }

    /// Where the ray meets the plane through `point` facing the camera.
    fn on_view_plane(&self, point: Vec3, origin: Vec3, dir: Vec3) -> Option<Vec3> {
        let normal = (self.camera.target - self.camera.eye()).normalize_or_zero();
        let denom = normal.dot(dir);
        (denom.abs() > 1e-4).then(|| origin + dir * (normal.dot(point - origin) / denom))
    }

    /// The pick under a window position, including overlay shapes.
    fn pick_raw(&self, position: Point<Pixels>, scale: f32) -> Option<(usize, usize)> {
        let renderer = self.renderer.as_ref().ok()?;
        let local = position - self.viewport.origin;
        let (x, y) = (f32::from(local.x) * scale, f32::from(local.y) * scale);
        if x < 0.0 || y < 0.0 {
            return None;
        }
        renderer.pick(x as u32, y as u32).ok().flatten()
    }

    /// Grab a box handle if one is under the cursor. True when the drag began.
    fn grab_box_handle(&mut self, position: Point<Pixels>, scale: f32) -> bool {
        let Some((center, size)) = self.dock_box() else { return false };
        let Some((entity, handle)) = self.pick_raw(position, scale) else { return false };
        if entity != render::OVERLAY_ENTITY || !(1..=annotations::HANDLE_CENTER_ID as usize).contains(&handle) {
            return false;
        }
        let handle = handle as u32;
        let Some((origin, dir)) = self.ray_at(position) else { return false };
        let center = Vec3::from(center);
        let (anchor, anchor_point) = match annotations::handle_axis(handle) {
            Some((axis, _)) => {
                let mut a = Vec3::ZERO;
                a[axis] = 1.0;
                (Self::closest_on_axis(center, a, origin, dir).unwrap_or(0.0), center)
            }
            None => (0.0, self.on_view_plane(center, origin, dir).unwrap_or(center)),
        };
        self.box_drag = Some(BoxDrag { handle, center0: center, size0: Vec3::from(size), anchor, anchor_point });
        self.drag = None;
        self.hover = None;
        true
    }

    /// Move or resize the box while a handle is dragged.
    fn drag_box(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        let Some(drag) = &self.box_drag else { return };
        let Some((origin, dir)) = self.ray_at(position) else { return };
        let (mut center, mut size) = (drag.center0, drag.size0);
        match annotations::handle_axis(drag.handle) {
            Some((axis, sign)) => {
                let mut a = Vec3::ZERO;
                a[axis] = 1.0;
                let Some(t) = Self::closest_on_axis(drag.center0, a, origin, dir) else { return };
                // Dragging a face moves it; the opposite face stays put.
                let wanted = drag.size0[axis] + sign * (t - drag.anchor);
                size[axis] = wanted.max(MIN_BOX);
                center[axis] = drag.center0[axis] + sign * (size[axis] - drag.size0[axis]) * 0.5;
            }
            None => {
                let Some(p) = self.on_view_plane(drag.center0, origin, dir) else { return };
                center = drag.center0 + (p - drag.anchor_point);
            }
        }
        if let Some(Annotation::DockBox { center: c, size: s }) =
            self.annotations.iter_mut().rev().find(|a| matches!(a, Annotation::DockBox { .. }))
        {
            *c = center.to_array();
            *s = size.to_array();
        }
        self.dirty = true;
        cx.notify();
    }

    fn open_menu(&mut self, position: Point<Pixels>, hit: Option<(usize, usize)>, cx: &mut Context<Self>) {
        let mut items: Vec<(String, MenuAction)> = Vec::new();
        if hit.is_some() {
            if let Some(mode) = self.measure.filter(|_| !self.pending.is_empty()) {
                items.push((format!("Add this atom to the {}", mode.label().to_lowercase()), MenuAction::AddToMeasurement));
            }
            items.push(("Measure distance from this atom".into(), MenuAction::Measure(MeasureMode::Distance)));
            items.push(("Measure angle from this atom".into(), MenuAction::Measure(MeasureMode::Angle)));
            items.push(("Measure dihedral from this atom".into(), MenuAction::Measure(MeasureMode::Dihedral)));
            items.push(("Select this residue".into(), MenuAction::SelectResidue));
            items.push(("Centre the view here".into(), MenuAction::CenterHere));
            items.push((format!("Docking box ({DEFAULT_BOX:.0} Å) around this atom"), MenuAction::BoxHere));
            items.push(("Toggle volume surface of this entity".into(), MenuAction::ToggleVolume));
        } else {
            items.push(("Place docking box on an atom…".into(), MenuAction::PlaceBox));
        }
        if !self.annotations.is_empty() {
            items.push(("Clear measurements and objects".into(), MenuAction::ClearMeasurements));
        }
        // Keep the menu inside the viewport.
        let local = position - self.viewport.origin;
        let (w, h) = (f32::from(self.viewport.size.width), f32::from(self.viewport.size.height));
        let height = items.len() as f32 * MENU_ROW + 2.0 * MENU_PAD;
        let at = gpui::point(
            px(f32::from(local.x).min((w - MENU_WIDTH).max(0.0))),
            px(f32::from(local.y).min((h - height).max(0.0))),
        );
        self.menu = Some(Menu { at, hit, items, hover: None });
        self.hover = None;
        cx.notify();
    }

    /// Row of the open menu under a window position.
    fn menu_row(&self, position: Point<Pixels>) -> Option<usize> {
        let menu = self.menu.as_ref()?;
        let local = position - self.viewport.origin - menu.at;
        let (x, y) = (f32::from(local.x), f32::from(local.y) - MENU_PAD);
        (x >= 0.0 && x < MENU_WIDTH && y >= 0.0)
            .then(|| (y / MENU_ROW) as usize)
            .filter(|&row| row < menu.items.len())
    }

    /// Is a window position anywhere over the open menu?
    fn over_menu(&self, position: Point<Pixels>) -> bool {
        let Some(menu) = &self.menu else { return false };
        let local = position - self.viewport.origin - menu.at;
        let height = menu.items.len() as f32 * MENU_ROW + 2.0 * MENU_PAD;
        (0.0..MENU_WIDTH).contains(&f32::from(local.x)) && (0.0..height).contains(&f32::from(local.y))
    }

    fn run_menu_action(&mut self, action: MenuAction, hit: Option<(usize, usize)>, cx: &mut Context<Self>) {
        let atom_ref = |this: &Self| {
            hit.and_then(|(e, a)| this.entities.get(e).map(|ent| AtomRef { entity: ent.name.clone(), atom: a }))
        };
        match action {
            MenuAction::Measure(mode) => {
                if let Some(first) = atom_ref(self) {
                    self.measure = Some(mode);
                    self.pending = vec![first];
                    self.placing_box = false;
                }
            }
            MenuAction::AddToMeasurement => {
                if let (Some(mode), Some(atom)) = (self.measure, atom_ref(self)) {
                    self.pending.push(atom);
                    if self.pending.len() >= mode.atoms() {
                        if let Some(a) = mode.build(&self.pending) {
                            self.annotations.push(a);
                        }
                        self.pending.clear();
                        cx.emit(ViewerEvent::SceneChanged);
                    }
                }
            }
            MenuAction::SelectResidue => {
                if let Some((entity, atom)) = hit {
                    let s = &self.entities[entity].structure;
                    let r = &s.residues[s.atoms[atom].residue as usize];
                    let atoms = (r.atoms.start as usize..r.atoms.end as usize).collect();
                    self.active = Some(entity);
                    self.set_entity_selection(entity, atoms, cx);
                }
            }
            MenuAction::CenterHere => {
                if let Some((entity, atom)) = hit {
                    self.active = Some(entity);
                    self.focus_atoms(&[atom], cx);
                }
            }
            MenuAction::BoxHere => {
                if let Some((entity, atom)) = hit {
                    self.box_at_atom(entity, atom, cx);
                }
            }
            MenuAction::PlaceBox => self.placing_box = true,
            MenuAction::ToggleVolume => {
                if let Some((entity, _)) = hit {
                    let next = if self.entities[entity].volume.is_some() { None } else { self.default_volume(entity) };
                    self.set_volume(entity, next, cx);
                }
            }
            MenuAction::ClearMeasurements => self.clear_annotations(cx),
        }
        self.dirty = true;
        cx.notify();
    }

    fn toggle_measure(&mut self, mode: MeasureMode, cx: &mut Context<Self>) {
        let next = if self.measure == Some(mode) { None } else { Some(mode) };
        self.set_measure_mode(next, cx);
    }

    fn key_down(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        let key = event.keystroke.key.as_str();
        let m = event.keystroke.modifiers;
        if m.control || m.platform {
            match key {
                "z" if m.shift => self.redo(cx),
                "z" => self.undo(cx),
                "y" => self.redo(cx),
                _ => {}
            }
            return;
        }
        match key {
            "g" => self.start_pose(PoseMode::Grab, cx),
            "r" => self.start_pose(PoseMode::Rotate, cx),
            "x" if self.pose.is_some() => self.set_pose_axis(Axis::X, cx),
            "y" if self.pose.is_some() => self.set_pose_axis(Axis::Y, cx),
            "z" if self.pose.is_some() => self.set_pose_axis(Axis::Z, cx),
            "enter" => self.confirm_pose(cx),
            "escape" if self.menu.is_some() => {
                self.menu = None;
                cx.notify();
            }
            "escape" if self.placing_box => self.set_placing_box(false, cx),
            "escape" if !self.pending.is_empty() => {
                self.pending.clear();
                cx.notify();
            }
            "escape" if self.pose.is_none() && self.measure.is_some() => self.set_measure_mode(None, cx),
            "escape" => self.cancel_pose(cx),
            "m" => self.toggle_measure(MeasureMode::Distance, cx),
            "a" => self.toggle_measure(MeasureMode::Angle, cx),
            "d" => self.toggle_measure(MeasureMode::Dihedral, cx),
            "space" => self.toggle_playback(cx),
            "home" => self.reset_view(cx),
            _ => {}
        }
    }

    // -------------------------------------------------------------- painting

    fn sync_renderer(&mut self) {
        let Ok(renderer) = &mut self.renderer else { return };
        if self.resync {
            renderer.truncate_entities(0);
            for e in &mut self.entities {
                e.geometry_dirty = true;
                e.volume_dirty = e.volume.is_some();
            }
            self.resync = false;
        }
        let playing = self.playing.is_some();
        for (slot, e) in self.entities.iter_mut().enumerate() {
            if e.geometry_dirty {
                // Surfaces are too slow to rebuild every frame during playback.
                let components: Vec<Component> = if playing {
                    e.components
                        .iter()
                        .copied()
                        .filter(|c| c.representation != Representation::GaussianSurface)
                        .collect()
                } else {
                    e.components.clone()
                };
                renderer.upload_entity(slot, &e.structure, &components, &e.selection);
                e.geometry_dirty = false;
            }
            // Surfaces are too slow to follow every frame of a playback.
            if e.volume_dirty && !playing {
                renderer.set_entity_volume(slot, &e.structure, e.volume.as_ref());
                e.volume_dirty = false;
            }
            renderer.set_entity_pose(slot, e.transform, e.visible);
        }
        renderer.truncate_entities(self.entities.len());
    }

    fn paint_viewport(&mut self, bounds: Bounds<Pixels>, window: &mut Window) {
        self.viewport = bounds;
        for image in self.retired.drain(..) {
            window.drop_image(image).ok();
        }
        if self.entities.iter().any(|e| e.geometry_dirty || e.volume_dirty) || self.resync {
            self.dirty = true;
        }

        let scale = window.scale_factor();
        let width = (f32::from(bounds.size.width) * scale).round().max(1.0) as u32;
        let height = (f32::from(bounds.size.height) * scale).round().max(1.0) as u32;

        if self.dirty || self.frame.is_none() || self.frame_size != (width, height) {
            self.sync_renderer();
            if let (Some(overlay), Ok(renderer)) = (self.overlay_next.take(), &mut self.renderer) {
                renderer.set_overlay(&overlay);
            }
            if let Ok(renderer) = &mut self.renderer {
                match renderer.render(&self.camera, width, height) {
                    Ok(frame) => {
                        self.last_render = frame.elapsed;
                        let buffer = image::RgbaImage::from_raw(frame.width, frame.height, frame.bgra)
                            .expect("frame buffer has width * height * 4 bytes");
                        let image = Arc::new(RenderImage::new(smallvec::smallvec![image::Frame::new(buffer)]));
                        if let Some(old) = self.frame.replace(image) {
                            self.retired.push(old);
                        }
                        self.frame_size = (width, height);
                        self.dirty = false;
                    }
                    Err(e) => self.error = Some(format!("render failed: {e:#}")),
                }
            }
        }

        if let Some(frame) = &self.frame {
            window
                .paint_image(bounds, Corners::default(), frame.clone(), 0, false)
                .ok();
        }
    }

    fn pick_at(&self, position: Point<Pixels>, scale: f32) -> Option<(usize, usize)> {
        let renderer = self.renderer.as_ref().ok()?;
        let local = position - self.viewport.origin;
        let (x, y) = (f32::from(local.x) * scale, f32::from(local.y) * scale);
        if x < 0.0 || y < 0.0 {
            return None;
        }
        renderer
            .pick(x as u32, y as u32)
            .ok()
            .flatten()
            .filter(|(e, a)| self.entities.get(*e).is_some_and(|ent| *a < ent.structure.atoms.len()))
    }

    fn mouse_down(&mut self, event: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus);
        if self.menu.is_some() {
            if event.button == MouseButton::Left && self.over_menu(event.position) {
                // The menu is drawn by this view and clicked here, so its rows need no handlers of their own.
                if let (Some(row), Some(menu)) = (self.menu_row(event.position), self.menu.take()) {
                    let action = menu.items[row].1;
                    self.run_menu_action(action, menu.hit, cx);
                } else {
                    self.menu = None;
                }
                self.swallow_up = true;
                cx.notify();
                return;
            }
            self.menu = None;
            cx.notify();
        }
        if event.button == MouseButton::Left
            && self.pose.is_none()
            && self.measure.is_none()
            && !self.placing_box
            && self.grab_box_handle(event.position, window.scale_factor())
        {
            return;
        }
        if self.pose.is_some() {
            match event.button {
                MouseButton::Left => self.confirm_pose(cx),
                _ => self.cancel_pose(cx),
            }
            return;
        }
        self.drag = Some(Drag {
            button: event.button,
            last: event.position,
            travelled: 0.0,
        });
    }

    fn mouse_move(&mut self, event: &MouseMoveEvent, scale: f32, cx: &mut Context<Self>) {
        if self.box_drag.is_some() {
            self.drag_box(event.position, cx);
            return;
        }
        if self.menu.is_some() {
            let row = self.menu_row(event.position);
            if let Some(menu) = &mut self.menu
                && menu.hover != row
            {
                menu.hover = row;
                cx.notify();
            }
            return;
        }
        if let Some(pose) = &mut self.pose {
            if pose.start_mouse.is_none() {
                pose.start_mouse = Some(event.position);
            }
            pose.mouse = Some(event.position);
            self.update_pose(cx);
            return;
        }
        if let Some(drag) = &mut self.drag {
            let delta = event.position - drag.last;
            let (dx, dy) = (f32::from(delta.x), f32::from(delta.y));
            drag.last = event.position;
            drag.travelled += dx.abs() + dy.abs();
            if drag.button != MouseButton::Left || event.modifiers.shift {
                self.camera.pan(dx, dy, f32::from(self.viewport.size.height));
            } else {
                self.camera.orbit(dx * ORBIT_SPEED, dy * ORBIT_SPEED);
            }
            self.hover = None;
            self.dirty = true;
            cx.notify();
        } else if self.viewport.contains(&event.position) {
            let hover = self.pick_at(event.position, scale).map(|(e, a)| (e, a, event.position));
            if hover.is_some() || self.hover.is_some() {
                self.hover = hover;
                cx.notify();
            }
        } else if self.hover.take().is_some() {
            cx.notify();
        }
    }

    fn mouse_up(&mut self, event: &MouseUpEvent, scale: f32, cx: &mut Context<Self>) {
        if std::mem::take(&mut self.swallow_up) {
            self.drag = None;
            return;
        }
        if self.box_drag.take().is_some() {
            cx.emit(ViewerEvent::SceneChanged);
            cx.notify();
            return;
        }
        let Some(drag) = self.drag.take() else { return };
        if drag.button == MouseButton::Right && drag.travelled < CLICK_SLOP && self.pose.is_none() {
            // A right click that did not pan opens the menu.
            let hit = self.pick_at(event.position, scale);
            self.open_menu(event.position, hit, cx);
            return;
        }
        if drag.button == MouseButton::Left && drag.travelled < CLICK_SLOP {
            let picked = self.pick_at(event.position, scale);
            if self.placing_box {
                if let Some((entity, atom)) = picked {
                    self.box_at_atom(entity, atom, cx);
                    self.placing_box = false;
                    self.dirty = true;
                    cx.notify();
                }
                return;
            }
            if let Some(mode) = self.measure {
                if let Some((entity, atom)) = picked {
                    let name = self.entities[entity].name.clone();
                    self.pending.push(AtomRef { entity: name, atom });
                    if self.pending.len() >= mode.atoms() {
                        if let Some(a) = mode.build(&self.pending) {
                            self.annotations.push(a);
                        }
                        self.pending.clear();
                        cx.emit(ViewerEvent::SceneChanged);
                    }
                    cx.notify();
                }
                return;
            }
            match picked {
                Some((entity, atom)) => {
                    // Clicking activates the entity and selects the residue (Mol* style).
                    self.active = Some(entity);
                    let s = &self.entities[entity].structure;
                    let r = &s.residues[s.atoms[atom].residue as usize];
                    let atoms = (r.atoms.start as usize..r.atoms.end as usize).collect();
                    self.set_entity_selection(entity, atoms, cx);
                }
                None => self.set_selection(Vec::new(), cx),
            }
            cx.emit(ViewerEvent::Picked(picked));
            cx.emit(ViewerEvent::SceneChanged);
        }
    }

    fn scroll(&mut self, event: &ScrollWheelEvent, cx: &mut Context<Self>) {
        let dy = f32::from(event.delta.pixel_delta(px(20.0)).y);
        if let Some(pose) = &mut self.pose {
            pose.depth -= dy * 0.02;
            self.update_pose(cx);
            return;
        }
        self.camera.zoom((-dy * 0.002).exp());
        self.dirty = true;
        cx.notify();
    }
}

impl gpui::Focusable for Viewer {
    fn focus_handle(&self, _: &gpui::App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for Viewer {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity();
        let viewport = canvas(
            |_, _, _| {},
            move |bounds, (), window, cx| {
                entity.update(cx, |this, _| this.paint_viewport(bounds, window));
                let scale = window.scale_factor();
                let e = entity.clone();
                window.on_mouse_event(move |event: &MouseDownEvent, phase, window, cx| {
                    if phase == DispatchPhase::Bubble && bounds.contains(&event.position) {
                        e.update(cx, |this, cx| this.mouse_down(event, window, cx));
                    }
                });
                let e = entity.clone();
                window.on_mouse_event(move |event: &MouseMoveEvent, phase, _, cx| {
                    if phase == DispatchPhase::Bubble {
                        e.update(cx, |this, cx| this.mouse_move(event, scale, cx));
                    }
                });
                let e = entity.clone();
                window.on_mouse_event(move |event: &MouseUpEvent, phase, _, cx| {
                    if phase == DispatchPhase::Bubble {
                        e.update(cx, |this, cx| this.mouse_up(event, scale, cx));
                    }
                });
                let e = entity.clone();
                window.on_mouse_event(move |event: &ScrollWheelEvent, phase, _, cx| {
                    if phase == DispatchPhase::Bubble && bounds.contains(&event.position) {
                        e.update(cx, |this, cx| this.scroll(event, cx));
                    }
                });
            },
        )
        .size_full();

        let tooltip = self.hover.filter(|_| self.pose.is_none()).map(|(entity, atom, pos)| {
            let local = pos - self.viewport.origin;
            div()
                .absolute()
                .left(local.x + px(14.0))
                .top(local.y + px(14.0))
                .px_2()
                .py_1()
                .rounded_md()
                .bg(rgb(0x262a33))
                .border_1()
                .border_color(rgb(0x3a3f4b))
                .text_xs()
                .text_color(rgb(0xe6e8ec))
                .child(self.describe(entity, atom))
        });

        // The 3D overlay (lines, box, handles) is uploaded with the next frame;
        // only labels are laid over the image here.
        let (vw, vh) = (f32::from(self.viewport.size.width), f32::from(self.viewport.size.height));
        let unit = if vh > 1.0 { 2.0 * self.camera.distance * (self.camera.fov_y * 0.5).tan() / vh } else { 0.05 };
        let edit_box = self.pose.is_none() && self.measure.is_none() && !self.placing_box;
        let (overlay, anchors) =
            annotations::shapes(&self.entities, &self.annotations, &self.annotation_hidden, &self.pending, unit, edit_box);
        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            unit.to_bits().hash(&mut h);
            for c in &overlay.cylinders {
                (c.p0.map(f32::to_bits), c.p1.map(f32::to_bits), c.id).hash(&mut h);
            }
            for p in overlay.spheres.iter().chain(&overlay.handles) {
                (p.center.map(f32::to_bits), p.radius.to_bits(), p.id).hash(&mut h);
            }
            for f in &overlay.faces {
                f.corners.map(|c| c.map(f32::to_bits)).hash(&mut h);
            }
            h.finish()
        };
        if fingerprint != self.overlay_fingerprint {
            self.overlay_fingerprint = fingerprint;
            self.overlay_next = Some(overlay);
            self.dirty = true;
        }
        let labels: Vec<_> = if vw > 1.0 && vh > 1.0 {
            let projector = annotations::Projector {
                matrix: self.camera.projection(vw / vh) * self.camera.view(),
                width: vw,
                height: vh,
            };
            anchors
                .into_iter()
                .filter_map(|l| {
                    let (x, y) = projector.point(l.at)?;
                    Some(
                        div()
                            .absolute()
                            .left(px(x + 6.0))
                            .top(px(y - 8.0))
                            .px_1()
                            .rounded_sm()
                            .bg(gpui::rgba(0x131418cc))
                            .text_xs()
                            .text_color(rgb(l.color))
                            .child(l.text),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let menu = self.menu.as_ref().map(|m| {
            let rows: Vec<_> = m
                .items
                .iter()
                .enumerate()
                .map(|(i, (label, _))| {
                    div()
                        .h(px(MENU_ROW))
                        .px_2()
                        .flex()
                        .items_center()
                        .text_xs()
                        .text_color(rgb(0xe6e8ec))
                        .when(m.hover == Some(i), |d| d.bg(rgb(0x2f4a7a)))
                        .child(label.clone())
                })
                .collect();
            div()
                .absolute()
                .left(m.at.x)
                .top(m.at.y)
                .w(px(MENU_WIDTH))
                .py(px(MENU_PAD))
                .rounded_md()
                .bg(rgb(0x20232b))
                .border_1()
                .border_color(rgb(0x3a3f4b))
                .children(rows)
        });
        let measure_hint = self.measure.map(|mode| {
            div()
                .absolute()
                .top_10()
                .left_0()
                .right_0()
                .flex()
                .justify_center()
                .child(
                    div()
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0x3a3320))
                        .border_1()
                        .border_color(rgb(0xf2c14e))
                        .text_sm()
                        .text_color(rgb(0xfbe8b0))
                        .child(format!(
                            "{}: click atom {} of {} · Esc cancels",
                            mode.label(),
                            self.pending.len() + 1,
                            mode.atoms()
                        )),
                )
        });

        let place_hint = self.placing_box.then(|| {
            div()
                .absolute()
                .top_10()
                .left_0()
                .right_0()
                .flex()
                .justify_center()
                .child(
                    div()
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0x1d3350))
                        .border_1()
                        .border_color(rgb(0x5fb4ff))
                        .text_sm()
                        .text_color(rgb(0xd6e8ff))
                        .child("Click an atom to centre the docking box on it · then drag its handles · Esc cancels"),
                )
        });

        let badge = self.pose_badge().map(|text| {
            div()
                .absolute()
                .bottom_3()
                .left_3()
                .right_3()
                .flex()
                .justify_center()
                .child(
                    div()
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0x2b3a57))
                        .border_1()
                        .border_color(rgb(0x4c8bf5))
                        .text_sm()
                        .text_color(rgb(0xe8eefc))
                        .child(text),
                )
        });

        div()
            .id("viewer")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| this.key_down(event, cx)))
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(rgb(0x131418))
            .child(viewport)
            .children(labels)
            .children(measure_hint)
            .children(place_hint)
            .children(menu)
            .children(tooltip)
            .children(badge)
            .when(self.entities.is_empty(), |el| {
                el.child(
                    div()
                        .absolute()
                        .inset_0()
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0x7d818c))
                        .child("Open a project, then add a structure — or ask the assistant to fetch one from the PDB"),
                )
            })
            .when_some(self.error.clone(), |el, error| {
                el.child(
                    div()
                        .absolute()
                        .top_10()
                        .left_3()
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .bg(rgb(0x3a1f22))
                        .text_sm()
                        .text_color(rgb(0xff9b94))
                        .child(error),
                )
            })
    }
}
