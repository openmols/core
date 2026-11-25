//! The Measure tab: click-to-measure modes, the list of annotations,
//! contacts between entities, the docking box and volume/surface metrics.

use crate::annotations::{self, Annotation, MeasureMode, kind_name, value_text};
use crate::workspace::{Workspace, heading, muted, section_body};
use futures::channel::oneshot;
use gpui::{Context, ParentElement, SharedString, Styled, div, prelude::*};
use gpui_component::{
    Disableable, IconName, Sizable,
    button::{Button, ButtonVariants},
};
use molecule::geometry::{self, Site};

/// Pocket probe points this close to a ligand atom count as overlapping it.
const OVERLAP_RADIUS: f32 = 2.0;

impl Workspace {
    pub fn render_measure_tab(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let viewer = self.viewer.read(cx);
        let mode = viewer.measure_mode();
        let mut panel = section_body();

        // Modes.
        let mut modes = div().flex().flex_wrap().gap_1();
        for (m, key) in [(MeasureMode::Distance, "M"), (MeasureMode::Angle, "A"), (MeasureMode::Dihedral, "D")] {
            let b = Button::new(SharedString::from(format!("measure-{}", m.label())))
                .label(format!("{} ({key})", m.label()))
                .xsmall()
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.viewer.update(cx, |v, cx| {
                        let next = if v.measure_mode() == Some(m) { None } else { Some(m) };
                        v.set_measure_mode(next, cx);
                    });
                }));
            modes = modes.child(if mode == Some(m) { b.primary() } else { b });
        }
        panel = panel.child(modes).child(
            div()
                .text_xs()
                .text_color(muted())
                .child("Pick a mode, then click atoms in the viewport (any entity). Esc cancels."),
        );

        // The list.
        let entities = viewer.entities();
        let rows: Vec<(usize, &'static str, String, bool)> = viewer
            .annotations()
            .iter()
            .enumerate()
            .map(|(i, a)| {
                (i, kind_name(a), value_text(entities, a).unwrap_or_else(|| "(hidden or missing entity)".into()), viewer.annotation_hidden(i))
            })
            .collect();
        if !rows.is_empty() {
            panel = panel.child(heading("Annotations"));
            for (i, kind, text, hidden) in rows {
                panel = panel.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_1()
                        .text_xs()
                        .child(
                            Button::new(("annotation-eye", i))
                                .icon(if hidden { IconName::EyeOff } else { IconName::Eye })
                                .ghost()
                                .xsmall()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.viewer.update(cx, |v, cx| v.toggle_annotation_hidden(i, cx));
                                })),
                        )
                        .child(div().flex_1().when(hidden, |d| d.opacity(0.45)).child(format!("{kind}  {text}")))
                        .child(
                            Button::new(("annotation-remove", i))
                                .label("×")
                                .ghost()
                                .xsmall()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.viewer.update(cx, |v, cx| v.remove_annotation(i, cx));
                                })),
                        ),
                );
            }
            panel = panel.child(
                Button::new("annotations-clear")
                    .label("Clear all")
                    .xsmall()
                    .on_click(cx.listener(|this, _, _, cx| this.viewer.update(cx, |v, cx| v.clear_annotations(cx)))),
            );
        }

        // Contacts of the active entity (its selection, if any) with another.
        let viewer = self.viewer.read(cx);
        if let Some(active) = viewer.active() {
            let others: Vec<String> = viewer
                .entities()
                .iter()
                .filter(|e| e.name != active.name)
                .map(|e| e.name.clone())
                .collect();
            if !others.is_empty() {
                let what = if active.selection.is_empty() { "all of" } else { "the selection of" };
                panel = panel.child(heading("Contacts")).child(
                    div()
                        .text_xs()
                        .text_color(muted())
                        .child(format!("{what} {} against:", active.name)),
                );
                for (i, other) in others.into_iter().enumerate() {
                    panel = panel.child(
                        Button::new(("contacts-with", i))
                            .label(format!("{other}  (≤ 4 Å)"))
                            .xsmall()
                            .on_click(cx.listener(move |this, _, _, cx| this.add_contacts(other.clone(), cx))),
                    );
                }
                panel = panel.child(div().text_xs().text_color(muted()).child(
                    "cyan: H-bond candidate (N/O ≤ 3.5 Å) · grey: contact · red: clash (< 75% of vdW sum)",
                ));
            }
        }

        // Docking box.
        let viewer = self.viewer.read(cx);
        panel = panel.child(heading("Docking box"));
        match viewer.dock_box() {
            Some((c, s)) => {
                panel = panel.child(div().text_xs().child(format!(
                    "centre {:.1}, {:.1}, {:.1} · size {:.1} × {:.1} × {:.1} Å",
                    c[0], c[1], c[2], s[0], s[1], s[2]
                )));
            }
            None => {
                panel = panel.child(div().text_xs().text_color(muted()).child("none yet"));
            }
        }
        let has_active = viewer.active().is_some();
        let placing = viewer.placing_box();
        panel = panel.child(
            div()
                .flex()
                .flex_wrap()
                .gap_1()
                .child({
                    let b = Button::new("box-place")
                        .label("Place on an atom…")
                        .xsmall()
                        .disabled(!has_active)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.viewer.update(cx, |v, cx| {
                                let on = !v.placing_box();
                                v.set_placing_box(on, cx);
                            });
                        }));
                    if placing { b.primary() } else { b }
                })
                .child(
                    Button::new("box-from-selection")
                        .label("Box the selection / ligand")
                        .xsmall()
                        .disabled(!has_active)
                        .on_click(cx.listener(|this, _, _, cx| this.box_selection(cx))),
                )
                .child(
                    Button::new("box-clear")
                        .label("Remove")
                        .xsmall()
                        .ghost()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.viewer.update(cx, |v, cx| {
                                let rest: Vec<Annotation> = v
                                    .annotations()
                                    .iter()
                                    .filter(|a| !matches!(a, Annotation::DockBox { .. }))
                                    .cloned()
                                    .collect();
                                v.set_annotations(rest, cx);
                            });
                        })),
                ),
        );
        panel = panel.child(div().text_xs().text_color(muted()).child(
            "Place on an atom drops a 16 Å box there (or right-click an atom). Drag a white sphere to move that face, the orange one to move the box. Vina and AutoDock4 use it as their grid.",
        ));

        // Volume and surface metrics.
        let volume = self.viewer.read(cx).active().and_then(|e| e.volume.as_ref().map(|v| v.wire));
        panel = panel.child(heading("Volume & surface")).child(
            div()
                .flex()
                .flex_wrap()
                .gap_1()
                .child({
                    let b = Button::new("volume-toggle")
                        .label("Translucent surface")
                        .xsmall()
                        .disabled(!has_active)
                        .on_click(cx.listener(|this, _, _, cx| this.toggle_volume(cx)));
                    if volume.is_some() { b.primary() } else { b }
                })
                .child({
                    let b = Button::new("volume-wire")
                        .label("Wireframe")
                        .xsmall()
                        .disabled(!has_active)
                        .on_click(cx.listener(|this, _, _, cx| this.toggle_wire(cx)));
                    if volume == Some(true) { b.primary() } else { b }
                })
                .child(
                    Button::new("volume-exposure")
                        .label("Colour by exposure")
                        .xsmall()
                        .disabled(!has_active || self.metrics_busy)
                        .on_click(cx.listener(|this, _, _, cx| this.paint_exposure(cx))),
                ),
        ).child(div().text_xs().text_color(muted()).child(
            "Wraps the selection (or the whole entity) in a see-through surface; exposure colours buried atoms blue and free ones orange, counting the other entities in the scene.",
        )).child(
            Button::new("metrics-run")
                .label(if self.metrics_busy { "Computing…" } else { "Measure active selection" })
                .xsmall()
                .disabled(!has_active || self.metrics_busy)
                .on_click(cx.listener(|this, _, _, cx| this.compute_metrics(cx))),
        );
        if let Some(rows) = &self.metrics {
            for (name, value) in rows {
                panel = panel.child(crate::workspace::stat_row(name.clone(), value.clone()));
            }
        }
        panel
    }

    /// Contacts between the active entity (its selection if it has one) and `other`.
    fn add_contacts(&mut self, other: String, cx: &mut Context<Self>) {
        let result = {
            let viewer = self.viewer.read(cx);
            let Some(active) = viewer.active() else { return };
            let selection = (!active.selection.is_empty()).then(|| active.selection.clone());
            annotations::contacts_between(viewer.entities(), &active.name, &other, selection.as_deref(), 4.0)
        };
        match result {
            Ok(annotation) => {
                let summary = {
                    let v = self.viewer.read(cx);
                    value_text(v.entities(), &annotation).unwrap_or_default()
                };
                self.status = Some(format!("Contacts: {summary}"));
                self.viewer.update(cx, |v, cx| v.add_annotation(annotation, cx));
            }
            Err(e) => self.status = Some(e),
        }
        cx.notify();
    }

    /// Box the selection of the active entity; failing that its ligand; failing that all of it.
    fn box_selection(&mut self, cx: &mut Context<Self>) {
        let points = {
            let viewer = self.viewer.read(cx);
            let Some(e) = viewer.active() else { return };
            let atoms: Vec<usize> = if !e.selection.is_empty() {
                e.selection.clone()
            } else {
                let ligand: Vec<usize> = (0..e.structure.atoms.len())
                    .filter(|&i| e.structure.kind(i) == molecule::AtomKind::Ligand)
                    .collect();
                if ligand.is_empty() { (0..e.structure.atoms.len()).collect() } else { ligand }
            };
            atoms
                .into_iter()
                .map(|i| e.transform.transform_point3(e.structure.atoms[i].position))
                .collect::<Vec<_>>()
        };
        if let Some(Annotation::DockBox { center, size }) = annotations::box_around(&points, 5.0) {
            self.viewer.update(cx, |v, cx| v.set_dock_box(center, size, cx));
        }
        cx.notify();
    }

    /// SASA and van der Waals volume of the active selection, on a worker thread.
    pub(crate) fn compute_metrics(&mut self, cx: &mut Context<Self>) {
        struct Job {
            name: String,
            target: Vec<Site>,
            scene: Vec<Site>,
            /// Positions in `scene` of the target atoms.
            target_in_scene: Vec<usize>,
            cavities: Vec<(String, Vec<glam::Vec3>)>,
        }
        let job = {
            let viewer = self.viewer.read(cx);
            let Some(active) = viewer.active_index() else { return };
            let mut job = Job {
                name: viewer.entities()[active].name.clone(),
                target: Vec::new(),
                scene: Vec::new(),
                target_in_scene: Vec::new(),
                cavities: Vec::new(),
            };
            for (i, e) in viewer.entities().iter().enumerate() {
                if !e.visible {
                    continue;
                }
                let is_cavity = e.name.ends_with(" cavities");
                if is_cavity {
                    let pts = (0..e.structure.atoms.len())
                        .map(|a| e.transform.transform_point3(e.structure.atoms[a].position))
                        .collect();
                    job.cavities.push((e.name.clone(), pts));
                    continue;
                }
                let wanted = |a: usize| i == active && (e.selection.is_empty() || e.selection.contains(&a));
                for a in 0..e.structure.atoms.len() {
                    if e.structure.residue_of(a).is_water() {
                        continue;
                    }
                    let site = Site {
                        pos: e.transform.transform_point3(e.structure.atoms[a].position),
                        element: e.structure.atoms[a].element,
                        id: job.scene.len(),
                    };
                    if wanted(a) {
                        job.target_in_scene.push(job.scene.len());
                        job.target.push(Site { id: a, ..site });
                    }
                    job.scene.push(site);
                }
            }
            job
        };
        if job.target.is_empty() {
            self.status = Some("nothing to measure (empty selection or only water)".into());
            cx.notify();
            return;
        }
        self.metrics_busy = true;
        cx.notify();
        let (tx, rx) = oneshot::channel();
        std::thread::spawn(move || {
            let free: f32 = geometry::sasa(&job.target, 1.4, 200, None).iter().sum();
            let in_scene: f32 = geometry::sasa(&job.scene, 1.4, 200, Some(&job.target_in_scene))
                .iter()
                .sum();
            let volume = geometry::vdw_volume(&job.target, 0.5);
            let positions: Vec<glam::Vec3> = job.target.iter().map(|s| s.pos).collect();
            let mut rows = vec![
                ("atoms".to_string(), job.target.len().to_string()),
                ("vdW volume".to_string(), format!("{volume:.0} Å³")),
                ("SASA alone".to_string(), format!("{free:.0} Å²")),
            ];
            if job.scene.len() > job.target.len() {
                let buried = if free > 0.0 { (1.0 - in_scene / free) * 100.0 } else { 0.0 };
                rows.push(("SASA in scene".to_string(), format!("{in_scene:.0} Å²")));
                rows.push(("buried by others".to_string(), format!("{buried:.0}%")));
            }
            for (name, pts) in &job.cavities {
                let frac = geometry::fraction_within(&positions, pts, OVERLAP_RADIUS);
                rows.push((format!("in {name}"), format!("{:.0}% of atoms", frac * 100.0)));
            }
            let _ = tx.send((job.name, rows));
        });
        cx.spawn(async move |this, cx| {
            if let Ok((name, mut rows)) = rx.await {
                this.update(cx, |this, cx| {
                    rows.insert(0, ("of".to_string(), name));
                    this.metrics = Some(rows);
                    this.metrics_busy = false;
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }
}

// ------------------------------------------------------------ assistant tools

impl Workspace {
    /// Resolve `{entity?, chain?, residue, atom}` to a reference.
    fn atom_spec(&self, spec: &serde_json::Value, cx: &Context<Self>) -> anyhow::Result<annotations::AtomRef> {
        use anyhow::Context as _;
        let viewer = self.viewer.read(cx);
        let entity = match spec["entity"].as_str() {
            Some(key) => viewer.find_entity(key).with_context(|| format!("no entity '{key}'"))?,
            None => viewer.active_index().context("the scene is empty")?,
        };
        let e = &viewer.entities()[entity];
        let number = spec["residue"].as_i64().context("each atom needs a residue number")? as isize;
        let chain = spec["chain"].as_str();
        let atom_name = spec["atom"].as_str().context("each atom needs an atom name, e.g. CA")?;
        let residue = e
            .structure
            .residues
            .iter()
            .position(|r| r.number == number && chain.is_none_or(|c| r.chain.eq_ignore_ascii_case(c)) && !r.is_water())
            .with_context(|| format!("no residue {number} in {}", e.name))?;
        let atom = e
            .structure
            .residue_atom(residue, atom_name)
            .with_context(|| format!("residue {number} has no atom {atom_name}"))?;
        Ok(annotations::AtomRef { entity: e.name.clone(), atom })
    }

    /// `measure` tool: a distance, angle or dihedral between named atoms.
    pub(crate) fn tool_measure(&mut self, args: &serde_json::Value, cx: &mut Context<Self>) -> anyhow::Result<String> {
        use anyhow::Context as _;
        let mode = match args["kind"].as_str() {
            Some("distance") => MeasureMode::Distance,
            Some("angle") => MeasureMode::Angle,
            Some("dihedral") => MeasureMode::Dihedral,
            _ => anyhow::bail!("kind must be distance, angle or dihedral"),
        };
        let specs = args["atoms"].as_array().context("atoms must be a list")?;
        anyhow::ensure!(specs.len() == mode.atoms(), "{} needs {} atoms", mode.label(), mode.atoms());
        let atoms = specs.iter().map(|s| self.atom_spec(s, cx)).collect::<anyhow::Result<Vec<_>>>()?;
        let annotation = mode.build(&atoms).context("could not build the measurement")?;
        let text = value_text(self.viewer.read(cx).entities(), &annotation).context("atoms not found")?;
        self.viewer.update(cx, |v, cx| v.add_annotation(annotation, cx));
        Ok(format!("{}: {text}", mode.label()))
    }

    /// `find_contacts` tool.
    pub(crate) fn tool_contacts(&mut self, args: &serde_json::Value, cx: &mut Context<Self>) -> anyhow::Result<String> {
        use anyhow::Context as _;
        let (a, b) = {
            let viewer = self.viewer.read(cx);
            let name = |key: &serde_json::Value, what: &str| -> anyhow::Result<String> {
                let key = key.as_str().with_context(|| format!("{what} is required"))?;
                let i = viewer.find_entity(key).with_context(|| format!("no entity '{key}'"))?;
                Ok(viewer.entities()[i].name.clone())
            };
            (name(&args["entity"], "entity")?, name(&args["against"], "against")?)
        };
        let cutoff = args["cutoff"].as_f64().unwrap_or(4.0).clamp(2.0, 8.0) as f32;
        let use_selection = args["use_selection"].as_bool().unwrap_or(false);
        let annotation = {
            let viewer = self.viewer.read(cx);
            let selection = use_selection
                .then(|| viewer.entities().iter().find(|e| e.name == a).map(|e| e.selection.clone()))
                .flatten()
                .filter(|s| !s.is_empty());
            annotations::contacts_between(viewer.entities(), &a, &b, selection.as_deref(), cutoff)
                .map_err(anyhow::Error::msg)?
        };
        let summary = value_text(self.viewer.read(cx).entities(), &annotation).unwrap_or_default();
        // The polar and clashing pairs by residue, so the model can talk about them.
        let mut lines = Vec::new();
        if let Annotation::Contacts { pairs, .. } = &annotation {
            let viewer = self.viewer.read(cx);
            let find = |n: &str| viewer.entities().iter().find(|e| e.name == n);
            if let (Some(ea), Some(eb)) = (find(&a), find(&b)) {
                for &(i, j, kind) in pairs.iter().filter(|p| p.2 != annotations::PairKind::Contact).take(12) {
                    let pa = ea.transform.transform_point3(ea.structure.atoms[i].position);
                    let pb = eb.transform.transform_point3(eb.structure.atoms[j].position);
                    lines.push(format!(
                        "{kind:?}: {} — {} ({:.2} Å)",
                        ea.structure.describe_atom(i),
                        eb.structure.describe_atom(j),
                        pa.distance(pb)
                    ));
                }
            }
        }
        self.viewer.update(cx, |v, cx| v.add_annotation(annotation, cx));
        Ok(format!("{a} vs {b} within {cutoff} Å: {summary}\n{}", lines.join("\n")))
    }

    /// `set_dock_box` tool.
    pub(crate) fn tool_dock_box(&mut self, args: &serde_json::Value, cx: &mut Context<Self>) -> anyhow::Result<String> {
        use anyhow::Context as _;
        let vec3 = |v: &serde_json::Value| -> Option<[f32; 3]> {
            let a = v.as_array()?;
            Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32])
        };
        match (vec3(&args["center"]), vec3(&args["size"])) {
            (Some(center), Some(size)) => {
                self.viewer.update(cx, |v, cx| v.set_dock_box(center, size, cx));
            }
            _ => {
                let padding = args["padding"].as_f64().unwrap_or(5.0) as f32;
                let points: Vec<glam::Vec3> = {
                    let viewer = self.viewer.read(cx);
                    let e = match args["entity"].as_str() {
                        Some(key) => viewer.find_entity(key).map(|i| &viewer.entities()[i]),
                        None => viewer.active(),
                    }
                    .context("no such entity")?;
                    let atoms: Vec<usize> = if !e.selection.is_empty() && args["around"].as_str() != Some("ligand") {
                        e.selection.clone()
                    } else {
                        let ligand: Vec<usize> = (0..e.structure.atoms.len())
                            .filter(|&i| e.structure.kind(i) == molecule::AtomKind::Ligand)
                            .collect();
                        if ligand.is_empty() { (0..e.structure.atoms.len()).collect() } else { ligand }
                    };
                    atoms
                        .into_iter()
                        .map(|i| e.transform.transform_point3(e.structure.atoms[i].position))
                        .collect()
                };
                if let Some(Annotation::DockBox { center, size }) = annotations::box_around(&points, padding) {
                    self.viewer.update(cx, |v, cx| v.set_dock_box(center, size, cx));
                }
            }
        }
        let (c, s) = self.viewer.read(cx).dock_box().context("no box was made")?;
        Ok(format!(
            "docking box centre {:.1}, {:.1}, {:.1}; size {:.1} x {:.1} x {:.1} A",
            c[0], c[1], c[2], s[0], s[1], s[2]
        ))
    }

    /// `list_annotations` tool.
    pub(crate) fn tool_list_annotations(&self, cx: &Context<Self>) -> String {
        let v = self.viewer.read(cx);
        if v.annotations().is_empty() {
            return "no measurements or objects in the scene".into();
        }
        v.annotations()
            .iter()
            .enumerate()
            .map(|(i, a)| format!("{}. {} {}", i + 1, kind_name(a), value_text(v.entities(), a).unwrap_or_default()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ------------------------------------------------------------------ volumes

/// Blue (buried) to orange (exposed), `t` in 0..=1.
fn exposure_color(t: f32) -> [u8; 4] {
    let (lo, hi) = ([70.0, 110.0, 220.0], [255.0, 165.0, 60.0]);
    let t = t.clamp(0.0, 1.0);
    let c = |i: usize| (lo[i] + (hi[i] - lo[i]) * t) as u8;
    [c(0), c(1), c(2), 150]
}

impl Workspace {
    /// Toggle a translucent surface on the active entity (its selection, if any).
    pub(crate) fn toggle_volume(&mut self, cx: &mut Context<Self>) {
        let Some(index) = self.viewer.read(cx).active_index() else { return };
        self.viewer.update(cx, |v, cx| {
            let next = if v.entities()[index].volume.is_some() { None } else { v.default_volume(index) };
            v.set_volume(index, next, cx);
        });
    }

    /// Show or hide the surface's wireframe.
    pub(crate) fn toggle_wire(&mut self, cx: &mut Context<Self>) {
        let Some(index) = self.viewer.read(cx).active_index() else { return };
        self.viewer.update(cx, |v, cx| {
            let next = v.entities()[index]
                .volume
                .clone()
                .or_else(|| v.default_volume(index))
                .map(|mut vol| {
                    vol.wire = !vol.wire;
                    vol
                });
            v.set_volume(index, next, cx);
        });
    }

    /// Paint the active entity's surface by how exposed each atom is in the scene:
    /// blue where it is buried (by itself or its partners), orange where it is free.
    pub(crate) fn paint_exposure(&mut self, cx: &mut Context<Self>) {
        struct Job {
            entity: usize,
            scene: Vec<Site>,
            /// Per atom of the entity: its place in `scene`, if it is in it.
            slot: Vec<Option<usize>>,
            radii: Vec<f32>,
            selected: Vec<bool>,
        }
        let job = {
            let viewer = self.viewer.read(cx);
            let Some(active) = viewer.active_index() else { return };
            let mut job = Job { entity: active, scene: Vec::new(), slot: Vec::new(), radii: Vec::new(), selected: Vec::new() };
            for (i, e) in viewer.entities().iter().enumerate() {
                if !e.visible || e.name.ends_with(" cavities") {
                    continue;
                }
                for a in 0..e.structure.atoms.len() {
                    let dry = !e.structure.residue_of(a).is_water();
                    if i == active {
                        job.radii.push(e.structure.atoms[a].element.vdw_radius());
                        job.selected.push(dry && (e.selection.is_empty() || e.selection.contains(&a)));
                        job.slot.push(dry.then_some(job.scene.len()));
                    }
                    if dry {
                        job.scene.push(Site {
                            pos: e.transform.transform_point3(e.structure.atoms[a].position),
                            element: e.structure.atoms[a].element,
                            id: job.scene.len(),
                        });
                    }
                }
            }
            job
        };
        if !job.selected.iter().any(|s| *s) {
            self.status = Some("nothing to paint (empty selection or only water)".into());
            cx.notify();
            return;
        }
        self.metrics_busy = true;
        cx.notify();
        let (tx, rx) = oneshot::channel();
        std::thread::spawn(move || {
            let wanted: Vec<usize> = job
                .slot
                .iter()
                .zip(&job.selected)
                .filter_map(|(slot, sel)| slot.filter(|_| *sel))
                .collect();
            let areas = geometry::sasa(&job.scene, 1.4, 60, Some(&wanted));
            let colors: Vec<[u8; 4]> = job
                .slot
                .iter()
                .zip(&job.radii)
                .map(|(slot, r)| match slot {
                    Some(s) => {
                        let full = 4.0 * std::f32::consts::PI * (r + 1.4).powi(2);
                        // Even fully exposed atoms rarely show more than about half their sphere.
                        exposure_color(areas[*s] / (full * 0.5))
                    }
                    None => [110, 170, 255, 105],
                })
                .collect();
            let _ = tx.send((job.entity, job.selected, colors));
        });
        cx.spawn(async move |this, cx| {
            if let Ok((entity, selected, colors)) = rx.await {
                this.update(cx, |this, cx| {
                    this.viewer.update(cx, |v, cx| {
                        let wire = v.entities().get(entity).and_then(|e| e.volume.as_ref()).is_some_and(|x| x.wire);
                        v.set_volume(entity, Some(render::Volume { selected, colors, wire }), cx);
                    });
                    this.metrics_busy = false;
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }
}
