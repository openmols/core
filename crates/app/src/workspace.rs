//! Window layout: projects/scene drawer (left), viewport (centre), assistant
//! drawer (right). Owns the open project and the user config.

use crate::chat::ChatPanel;
use crate::config::{Config, LeftTab};
use crate::jobs_ui::{CavityResults, DockSetup, DockingResults, EnsemblePockets};
use crate::tools;
use crate::viewer::{EntityState, ViewState, Viewer, ViewerEvent};
use anyhow::{Context as _, Result};
use gpui::{
    AppContext, ClickEvent, Context, Entity, ExternalPaths, Focusable, FontWeight, Hsla, IntoElement, ParentElement,
    PathPromptOptions, Render, SharedString, Styled, Subscription, Window, div, prelude::*, px, rgb,
};
use gpui_component::{
    IconName, Sizable,
    button::{Button, ButtonVariants},
    input::{Input, InputEvent, InputState},
    resizable::{ResizableState, h_resizable, resizable_panel},
};
use molecule::Structure;
use molecule::align::AlignOptions;
use project::{Id, Project, ViewEntry};
use render::{Coloring, Component, Representation, Target};
use std::collections::HashMap;
use std::path::PathBuf;

const LEFT_WIDTH: f32 = 290.0;
const RIGHT_WIDTH: f32 = 400.0;
/// Drag limits for the side drawers.
const SIDE_MIN: f32 = 220.0;
const SIDE_MAX: f32 = 720.0;

pub struct Workspace {
    pub viewer: Entity<Viewer>,
    pub chat: Entity<ChatPanel>,
    pub project: Option<Project>,
    pub config: Config,
    new_project_name: Entity<InputState>,
    /// Editable pocket filter rules, one per line.
    pub(crate) pocket_filter: Entity<InputState>,
    /// Filter text waiting to be shown in that editor. Setting an input
    /// needs a Window, which only the render pass has.
    pub(crate) pending_filter_text: Option<String>,
    left_open: bool,
    right_open: bool,
    pub(crate) status: Option<String>,
    /// `docker version` result (checked once at startup).
    pub(crate) docker: Option<Result<String, String>>,
    /// Running jobs → container name (for cancellation).
    pub(crate) running: HashMap<Id, String>,
    pub(crate) cavities: Option<CavityResults>,
    pub(crate) docking: Option<DockingResults>,
    /// Volume/surface figures of the last Measure run.
    pub(crate) metrics: Option<Vec<(String, String)>>,
    pub(crate) metrics_busy: bool,
    pub(crate) ensemble: Option<EnsemblePockets>,
    /// Which entities the Dock button will send to FRODOCK, and how.
    pub(crate) dock: DockSetup,
    /// Structures being read on a worker thread: id → name, for the status
    /// line and to stop a second click starting the same read again.
    pub(crate) loading: HashMap<Id, String>,
    /// Structures waiting for the current read to finish.
    pub(crate) load_queue: Vec<(Id, bool)>,
    /// Widths of the three resizable columns (drawers + viewport).
    layout: Entity<ResizableState>,
    _subscriptions: Vec<Subscription>,
}

/// ` (1.2 GB)` for a file worth warning about, nothing for a small one.
fn describe_size(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    match bytes as f64 / MB {
        mb if mb >= 1024.0 => format!(" ({:.1} GB)", mb / 1024.0),
        mb if mb >= 20.0 => format!(" ({mb:.0} MB)"),
        _ => String::new(),
    }
}

/// Does this look like a structure openmols can read?
fn is_structure_file(path: &std::path::Path) -> bool {
    let lower = path.to_string_lossy().to_lowercase();
    let lower = lower.strip_suffix(".gz").unwrap_or(&lower);
    molecule::is_pdb_like(lower)
        || [".cif", ".mmcif", ".sdf", ".mol"].iter().any(|e| lower.ends_with(e))
}

pub(crate) fn muted() -> Hsla {
    rgb(0x7d818c).into()
}

pub(crate) fn section(title: &'static str) -> gpui::Div {
    div().flex().flex_col().gap_1().child(
        div()
            .text_xs()
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(muted())
            .child(title.to_uppercase()),
    )
}

/// A section without its heading: [`Workspace::fold`] supplies a clickable one.
pub(crate) fn section_body() -> gpui::Div {
    div().flex().flex_col().gap_1()
}

/// A small heading for a part of a panel.
pub(crate) fn heading(title: &'static str) -> gpui::Div {
    div()
        .pt_2()
        .text_xs()
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(muted())
        .child(title.to_uppercase())
}

/// A sub-heading with rows under it, for read-only figures.
pub(crate) fn section_rows(title: &'static str) -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .mt_1()
        .child(
            div()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(muted())
                .child(title.to_uppercase()),
        )
}

/// `name   value`, the name muted and the value right-aligned.
pub(crate) fn stat_row(name: impl Into<SharedString>, value: impl Into<SharedString>) -> gpui::Div {
    div()
        .flex()
        .justify_between()
        .gap_2()
        .px_2()
        .text_xs()
        .child(div().text_color(muted()).child(name.into()))
        .child(div().child(value.into()))
}

fn list_row(id: impl Into<gpui::ElementId>, label: impl Into<SharedString>, active: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .px_2()
        .py_1()
        .rounded_md()
        .text_sm()
        .cursor_pointer()
        .when(active, |d| d.bg(rgb(0x24324d)).text_color(rgb(0xe8eefc)))
        .hover(|s| s.bg(rgb(0x2c3039)))
        .child(label.into())
}

impl Workspace {
    pub fn new(initial: Option<PathBuf>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let viewer = cx.new(Viewer::new);
        let weak = cx.entity().downgrade();
        let chat = cx.new(|cx| ChatPanel::new(weak, window, cx));
        let new_project_name = cx.new(|cx| InputState::new(window, cx).placeholder("New project name"));
        let pocket_filter = cx.new(|cx| {
            InputState::new(window, cx)
                .multi_line(true)
                .auto_grow(3, 12)
                .placeholder("volume >= 120
hydropathy <= 0
overlap 0.6")
        });
        let subscriptions = vec![
            cx.subscribe(&viewer, |this, _, event: &ViewerEvent, cx| {
                if let ViewerEvent::Picked(pick) = event {
                    this.status = pick.map(|(e, a)| this.viewer.read(cx).describe(e, a));
                }
                cx.notify();
            }),
            cx.subscribe_in(&new_project_name, window, |this, _, event: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { .. } = event {
                    this.create_project(window, cx);
                }
            }),
        ];
        let mut this = Self {
            viewer,
            chat,
            project: None,
            config: Config::load(),
            new_project_name,
            pocket_filter,
            pending_filter_text: None,
            left_open: true,
            right_open: true,
            status: None,
            docker: None,
            running: HashMap::new(),
            cavities: None,
            docking: None,
            metrics: None,
            metrics_busy: false,
            ensemble: None,
            dock: DockSetup::default(),
            loading: HashMap::new(),
            load_queue: Vec::new(),
            layout: cx.new(|_| ResizableState::default()),
            _subscriptions: subscriptions,
        };
        this.check_docker(cx);
        {
            let d = &this.config.job_defaults;
            this.dock.top_n = d.dock_top_n;
            this.dock.cluster_rmsd = d.dock_cluster_rmsd;
            this.dock.around_ligand = d.dock_around_ligand;
            this.dock.restrain_to_selection = d.dock_restrain_to_selection;
        }

        match initial {
            Some(path) if Project::is_project(&path) => this.open_project(path, cx),
            Some(path) => this.load_file_without_project(path, cx),
            None => {
                if let Some(last) = this.config.recent_projects.first().cloned()
                    && Project::is_project(&last)
                {
                    this.open_project(last, cx);
                }
            }
        }
        // Developer hook: send a prompt on startup (scripted demos / smoke tests).
        // Deferred so the chat can read the fully constructed workspace.
        if let Ok(prompt) = std::env::var("OPENMOLS_PROMPT")
            && !prompt.trim().is_empty()
        {
            let chat = this.chat.clone();
            cx.spawn(async move |_, cx| {
                chat.update(cx, |c, cx| c.send_text(prompt, cx)).ok();
            })
            .detach();
        }
        this
    }

    // -------------------------------------------------------------- projects

    pub fn open_project(&mut self, root: PathBuf, cx: &mut Context<Self>) {
        match Project::open(&root) {
            Ok(project) => self.set_project(project, cx),
            Err(e) => self.status = Some(format!("{e:#}")),
        }
        cx.notify();
    }

    pub(crate) fn set_project(&mut self, project: Project, cx: &mut Context<Self>) {
        self.config.touch_recent(project.root().to_path_buf());
        let first = project.structures().ok().and_then(|s| s.first().map(|e| e.id));
        self.project = Some(project);
        self.running.clear();
        self.restore_cavity_results();
        self.restore_docking_results();
        self.restore_ensemble_results(cx);
        self.viewer.update(cx, |v, cx| v.clear(cx));
        if let Some(id) = first {
            self.load_structure(id, cx).ok();
        }
        let (conversation, messages) = self
            .project
            .as_ref()
            .and_then(|p| {
                let last = p.conversations().ok()?.pop()?;
                let messages = p
                    .messages(last.id)
                    .ok()?
                    .into_iter()
                    .filter_map(|v| serde_json::from_value(v).ok())
                    .collect();
                Some((Some(last.id), messages))
            })
            .unwrap_or_default();
        self.chat
            .update(cx, |c, cx| c.show_conversation(conversation, messages, cx));
        cx.notify();
    }

    fn create_project(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let name = self.new_project_name.read(cx).value().trim().to_string();
        if name.is_empty() {
            self.status = Some("Enter a project name first".into());
            cx.notify();
            return;
        }
        let slug: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
            .collect::<String>()
            .trim_matches('-')
            .to_string();
        let mut root = self.config.projects_dir.join(if slug.is_empty() { "project" } else { &slug });
        let mut n = 2;
        while root.exists() {
            root = self.config.projects_dir.join(format!("{slug}-{n}"));
            n += 1;
        }
        match Project::create(&root, &name) {
            Ok(project) => {
                self.new_project_name.update(cx, |i, cx| i.set_value("", window, cx));
                self.set_project(project, cx);
                self.status = Some(format!("Created {}", root.display()));
            }
            Err(e) => self.status = Some(format!("{e:#}")),
        }
        cx.notify();
    }

    fn prompt_open_project(&mut self, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Open project folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = paths.await
                && let Some(path) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| this.open_project(path, cx)).ok();
            }
        })
        .detach();
    }

    fn close_project(&mut self, cx: &mut Context<Self>) {
        self.project = None;
        self.cavities = None;
        self.docking = None;
        self.ensemble = None;
        self.viewer.update(cx, |v, cx| v.clear(cx));
        self.chat.update(cx, |c, cx| c.new_conversation(cx));
        cx.notify();
    }

    // ------------------------------------------------------------ structures

    pub(crate) fn read_structure(&self, id: Id) -> Result<Structure> {
        let project = self.project.as_ref().context("no project is open")?;
        let entry = project
            .structure(id)?
            .with_context(|| format!("no structure with id {id}"))?;
        let mut structure = Structure::load(project.structure_path(&entry))?;
        structure.name = entry.name.clone();
        Ok(structure)
    }

    /// Read a project structure on a worker thread, then put it in the scene.
    ///
    /// Parsing is the slow part of opening a structure — seconds for a
    /// trajectory of a thousand frames, which is over a gigabyte of text —
    /// and on the UI thread that freezes the window. Only the finished
    /// structure comes back to be added, which takes milliseconds.
    /// Queued behind any read already in flight, so adding a folder of
    /// trajectories does not try to hold them all in memory at once.
    pub fn load_structure_async(&mut self, id: Id, replace: bool, cx: &mut Context<Self>) {
        if self.loading.contains_key(&id) || self.load_queue.iter().any(|(q, _)| *q == id) {
            return;
        }
        if self.loading.is_empty() {
            self.begin_structure_load(id, replace, cx);
        } else {
            self.load_queue.push((id, replace));
            cx.notify();
        }
    }

    fn begin_structure_load(&mut self, id: Id, replace: bool, cx: &mut Context<Self>) {
        let entry = match self
            .project
            .as_ref()
            .context("no project is open")
            .and_then(|p| p.structure(id))
        {
            Ok(Some(entry)) => entry,
            Ok(None) => {
                self.status = Some(format!("no structure with id {id}"));
                return;
            }
            Err(e) => {
                self.status = Some(format!("{e:#}"));
                return;
            }
        };
        let Some(project) = self.project.as_ref() else { return };
        let path = project.structure_path(&entry);
        let name = entry.name.clone();
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        self.loading.insert(id, name.clone());
        self.status = Some(format!("Reading {name}{}…", describe_size(size)));
        cx.notify();

        let (tx, rx) = futures::channel::oneshot::channel();
        let thread_name = name.clone();
        std::thread::spawn(move || {
            let result = Structure::load(&path).map(|mut s| {
                s.name = thread_name;
                s
            });
            let _ = tx.send(result.map_err(|e| format!("{e:#}")));
        });

        cx.spawn(async move |this, cx| {
            let result = rx.await;
            this.update(cx, |ws, cx| {
                ws.loading.remove(&id);
                match result {
                    Ok(Ok(structure)) => {
                        let (atoms, frames) = (structure.atoms.len(), structure.frame_count());
                        ws.viewer.update(cx, |v, cx| {
                            if replace {
                                v.set_structure(structure, Some(id), cx);
                            } else {
                                v.add_structure(structure, Some(id), cx);
                            }
                        });
                        ws.status = Some(match frames {
                            1 => format!("{name}: {atoms} atoms"),
                            n => format!("{name}: {atoms} atoms, {n} frames"),
                        });
                    }
                    Ok(Err(e)) => ws.status = Some(e),
                    Err(_) => ws.status = Some(format!("reading {name} was interrupted")),
                }
                if !ws.load_queue.is_empty() {
                    let (next, replace) = ws.load_queue.remove(0);
                    ws.begin_structure_load(next, replace, cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Replace the scene with one project structure.
    pub fn load_structure(&mut self, id: Id, cx: &mut Context<Self>) -> Result<()> {
        let structure = self.read_structure(id)?;
        self.viewer
            .update(cx, |v, cx| v.set_structure(structure, Some(id), cx));
        cx.notify();
        Ok(())
    }

    /// Add a project structure to the scene; returns its entity index.
    pub fn add_structure_to_scene(&mut self, id: Id, cx: &mut Context<Self>) -> Result<usize> {
        let structure = self.read_structure(id)?;
        let index = self
            .viewer
            .update(cx, |v, cx| v.add_structure(structure, Some(id), cx));
        cx.notify();
        Ok(index)
    }

    fn add_files(&mut self, paths: Vec<PathBuf>, cx: &mut Context<Self>) {
        let Some(project) = &self.project else {
            if let Some(path) = paths.into_iter().next() {
                self.load_file_without_project(path, cx);
            }
            return;
        };
        let mut added = Vec::new();
        for path in paths {
            match project.add_structure_file(&path, None) {
                Ok(entry) => added.push(entry.id),
                Err(e) => self.status = Some(format!("{e:#}")),
            }
        }
        for id in added {
            self.load_structure_async(id, false, cx);
        }
        cx.notify();
    }

    fn load_file_without_project(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        match Structure::load(&path) {
            Ok(s) => {
                self.viewer.update(cx, |v, cx| {
                    v.add_structure(s, None, cx);
                });
                self.status = Some("Opened without a project — create a project to keep it".into());
            }
            Err(e) => self.status = Some(format!("{e:#}")),
        }
        cx.notify();
    }

    fn prompt_add_structure(&mut self, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Add structures".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = paths.await {
                this.update(cx, |this, cx| this.add_files(paths, cx)).ok();
            }
        })
        .detach();
    }

    /// Add every structure file in a folder.
    ///
    /// Windows' dialog shows either files or folders, never both, so picking
    /// a folder of structures needs its own action.
    fn prompt_add_folder(&mut self, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Add every structure in this folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = paths.await
                && let Some(folder) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| this.add_folder(folder, cx)).ok();
            }
        })
        .detach();
    }

    fn add_folder(&mut self, folder: PathBuf, cx: &mut Context<Self>) {
        let Ok(entries) = std::fs::read_dir(&folder) else {
            self.status = Some(format!("Could not read {}", folder.display()));
            cx.notify();
            return;
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_file() && is_structure_file(p))
            .collect();
        files.sort();
        if files.is_empty() {
            self.status = Some(format!(
                "No .pdb, .cif, .sdf or .mol files in {}",
                folder.display()
            ));
            cx.notify();
            return;
        }
        let count = files.len();
        self.add_files(files, cx);
        self.status = Some(format!("Added {count} structure file(s) from {}", folder.display()));
        cx.notify();
    }

    /// Write an entity's posed coordinates into the project as a new structure.
    pub fn export_pose(&mut self, index: usize, cx: &mut Context<Self>) -> Result<String> {
        let project = self.project.as_ref().context("no project is open")?;
        let (name, text) = {
            let viewer = self.viewer.read(cx);
            let entity = viewer.entities().get(index).context("no such entity")?;
            (
                entity.name.clone(),
                molecule::write_pdb(&entity.structure, entity.transform, |_| true),
            )
        };
        let slug: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let entry = project.add_structure_bytes(
            &format!("{slug}_pose.pdb"),
            text.as_bytes(),
            &format!("{name} (posed)"),
            Some(&format!("pose:{name}")),
        )?;
        cx.notify();
        Ok(format!("exported {name} as structure id={} ({})", entry.id, entry.file))
    }

    // ----------------------------------------------------------------- views

    pub fn save_view(&mut self, name: &str, cx: &mut Context<Self>) -> Result<()> {
        let project = self.project.as_ref().context("no project is open")?;
        let viewer = self.viewer.read(cx);
        anyhow::ensure!(!viewer.entities().is_empty(), "the scene is empty");
        project.save_view(name, viewer.structure_id(), &tools::view_state_json(&viewer.view_state()))?;
        cx.notify();
        Ok(())
    }

    pub fn apply_view(&mut self, view: ViewEntry, cx: &mut Context<Self>) -> Result<()> {
        let mut state: ViewState = serde_json::from_value(view.state).context("view state is invalid")?;
        if state.entities.is_empty() {
            // Legacy single-structure view.
            state.entities.push(EntityState {
                name: String::new(),
                structure_id: view.structure_id,
                transform: glam::Mat4::IDENTITY,
                components: std::mem::take(&mut state.components),
                selection: std::mem::take(&mut state.selection),
                frame: 0,
                visible: true,
            });
        }
        let mut loaded = Vec::new();
        for entity in &state.entities {
            let id = entity.structure_id.context("view references a structure outside the project")?;
            loaded.push((self.read_structure(id)?, id, entity.clone()));
        }
        self.viewer.update(cx, |v, cx| {
            v.clear(cx);
            for (structure, id, entity) in loaded {
                let index = v.add_structure(structure, Some(id), cx);
                let mut entity = entity;
                if entity.name.is_empty() {
                    entity.name = v.entities()[index].name.clone();
                }
                v.apply_entity_state(index, &entity, cx);
            }
            v.set_active_index(state.active);
            v.set_annotations(state.annotations.clone(), cx);
            v.set_annotation_hidden(state.annotation_hidden.clone(), cx);
            v.set_camera(state.camera, cx);
        });
        cx.notify();
        Ok(())
    }

    fn quick_save_view(&mut self, cx: &mut Context<Self>) {
        let count = self
            .project
            .as_ref()
            .and_then(|p| p.views().ok())
            .map(|v| v.len())
            .unwrap_or(0);
        if let Err(e) = self.save_view(&format!("View {}", count + 1), cx) {
            self.status = Some(format!("{e:#}"));
        }
        cx.notify();
    }

    fn apply_preset(&mut self, preset: &str, cx: &mut Context<Self>) {
        let Some(structure) = self.viewer.read(cx).structure().cloned() else { return };
        let ligand = Component::new(Target::Ligand, Representation::BallAndStick, Coloring::Element);
        let polymer = |r, c| Component::new(Target::Polymer, r, c);
        let mut components = match preset {
            "surface" => vec![polymer(Representation::GaussianSurface, Coloring::Chain), ligand],
            "cartoon+surface" => vec![
                polymer(Representation::Cartoon, Coloring::Chain),
                polymer(Representation::GaussianSurface, Coloring::Chain),
                ligand,
            ],
            "secondary" => vec![polymer(Representation::Cartoon, Coloring::Secondary), ligand],
            "ball" => vec![Component::new(Target::All, Representation::BallAndStick, Coloring::Element)],
            "spacefill" => vec![Component::new(Target::All, Representation::Spacefill, Coloring::Element)],
            "backbone" => vec![polymer(Representation::Backbone, Coloring::Chain), ligand],
            _ => render::default_preset(&structure),
        };
        let viewer = self.viewer.read(cx);
        if viewer.water_visible() {
            components.push(Component::new(Target::Water, Representation::BallAndStick, Coloring::Element));
        }
        if !viewer.selection().is_empty() {
            components.push(Component::new(Target::Selection, Representation::BallAndStick, Coloring::Element));
        }
        self.viewer.update(cx, |v, cx| v.set_components(components, cx));
    }

    fn align_onto_active(&mut self, mobile: usize, cx: &mut Context<Self>) {
        let Some(target) = self.viewer.read(cx).active_index() else { return };
        let result = self
            .viewer
            .update(cx, |v, cx| v.align(mobile, target, &AlignOptions::default(), cx));
        self.status = Some(match result {
            Ok(fit) => format!(
                "Aligned: RMSD {:.2} Å over {} of {} Cα pairs (Ctrl+Z to undo)",
                fit.rmsd, fit.n_aligned, fit.n_pairs
            ),
            Err(e) => format!("Align failed: {e:#}"),
        });
        cx.notify();
    }

    // ---------------------------------------------------------------- render

    fn render_scene_section(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let viewer = self.viewer.read(cx);
        let active = viewer.active_index();
        let mut scene = section_body();
        if viewer.entities().len() > 1 {
            scene = scene.child(
                div()
                    .flex()
                    .gap_1()
                    .child(
                        Button::new("show-all")
                            .icon(IconName::Eye)
                            .label("Show all")
                            .ghost()
                            .xsmall()
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.viewer.update(cx, |v, cx| {
                                    for j in 0..v.entities().len() {
                                        v.set_visible(j, true, cx);
                                    }
                                });
                            })),
                    )
                    .child(
                        Button::new("hide-all")
                            .icon(IconName::EyeOff)
                            .label("Hide all")
                            .ghost()
                            .xsmall()
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.viewer.update(cx, |v, cx| {
                                    for j in 0..v.entities().len() {
                                        v.set_visible(j, false, cx);
                                    }
                                });
                            })),
                    ),
            );
        }
        for (i, e) in viewer.entities().iter().enumerate() {
            let is_active = Some(i) == active;
            let visible = e.visible;
            let mut row = div()
                .flex()
                .items_center()
                .gap_1()
                .child(
                    // The eye: click to show or hide, shift-click to show only this one.
                    Button::new(("vis", i))
                        .icon(if visible { IconName::Eye } else { IconName::EyeOff })
                        .ghost()
                        .xsmall()
                        .tooltip(if visible { "Hide (shift-click: show only this)" } else { "Show (shift-click: show only this)" })
                        .on_click(cx.listener(move |this, event: &ClickEvent, _, cx| {
                            let solo = event.modifiers().shift;
                            this.viewer.update(cx, |v, cx| {
                                if solo {
                                    for j in 0..v.entities().len() {
                                        v.set_visible(j, j == i, cx);
                                    }
                                } else {
                                    v.set_visible(i, !visible, cx);
                                }
                            });
                        })),
                )
                .child(div().flex_1().when(!visible, |d| d.opacity(0.45)).child(
                    list_row(("entity", i), e.name.clone(), is_active).on_click(cx.listener(move |this, _, _, cx| {
                        this.viewer.update(cx, |v, cx| v.set_active(i, cx));
                    })),
                ));
            if !is_active && active.is_some() {
                row = row.child(
                    Button::new(("align", i))
                        .label("Align")
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| this.align_onto_active(i, cx))),
                );
            }
            row = row.child(
                Button::new(("remove-entity", i))
                    .label("×")
                    .ghost()
                    .xsmall()
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.viewer.update(cx, |v, cx| v.remove_entity(i, cx));
                    })),
            );
            scene = scene.child(row);
        }

        if let Some(i) = active {
            let e = &viewer.entities()[i];
            // A complex (protein and ligand together) cannot be docked until the ligand is its own entity.
            let has_ligand = e.structure.residues.iter().any(|r| {
                r.atoms.len() >= 4 && r.atoms.start < r.atoms.end && e.structure.kind(r.atoms.start as usize) == molecule::AtomKind::Ligand
            });
            let has_polymer = (0..e.structure.atoms.len()).any(|a| e.structure.kind(a) == molecule::AtomKind::Polymer);
            if has_ligand && has_polymer {
                scene = scene
                    .child(
                        Button::new("split-ligand")
                            .label("Split ligand off")
                            .xsmall()
                            .on_click(cx.listener(move |this, _, _, cx| this.split_ligand(i, cx))),
                    )
                    .child(div().text_xs().text_color(muted()).child(
                        "This entity holds a protein and a ligand. Split them to dock the ligand or measure against the protein.",
                    ));
            }
            let frames = e.structure.frame_count();
            if frames > 1 {
                let frame = e.frame;
                let playing = viewer.is_playing();
                scene = scene.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1()
                        .text_xs()
                        .child(Button::new("frame-prev").label("◀").xsmall().on_click(cx.listener(
                            move |this, _, _, cx| {
                                this.viewer
                                    .update(cx, |v, cx| v.set_frame(i, (frame + frames - 1) % frames, cx));
                            },
                        )))
                        .child(format!("frame {}/{}", frame + 1, frames))
                        .child(Button::new("frame-next").label("▶").xsmall().on_click(cx.listener(
                            move |this, _, _, cx| {
                                this.viewer.update(cx, |v, cx| v.set_frame(i, (frame + 1) % frames, cx));
                            },
                        )))
                        .child({
                            let b = Button::new("play")
                                .label(if playing { "Pause" } else { "Play" })
                                .xsmall()
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.viewer.update(cx, |v, cx| v.toggle_playback(cx));
                                }));
                            if playing { b.primary() } else { b }
                        })
                        .child(Button::new("fit-frames").label("Fit frames").xsmall().on_click(cx.listener(
                            move |this, _, _, cx| {
                                let r = this.viewer.update(cx, |v, cx| v.fit_frames(i, 0, cx));
                                this.status = Some(match r {
                                    Ok(rmsd) => format!("Frames fitted to frame 1 (mean Cα RMSD {rmsd:.2} Å)"),
                                    Err(e) => format!("{e:#}"),
                                });
                                cx.notify();
                            },
                        ))),
                );
            }
            scene = scene.child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_1()
                    .child(Button::new("grab").label("Grab (G)").xsmall().on_click(cx.listener(|this, _, window, cx| {
                        this.viewer.update(cx, |v, cx| v.start_pose(crate::viewer::PoseMode::Grab, cx));
                        window.focus(&this.viewer.focus_handle(cx));
                    })))
                    .child(Button::new("rotate").label("Rotate (R)").xsmall().on_click(cx.listener(
                        |this, _, window, cx| {
                            this.viewer.update(cx, |v, cx| v.start_pose(crate::viewer::PoseMode::Rotate, cx));
                            window.focus(&this.viewer.focus_handle(cx));
                        },
                    )))
                    .child(Button::new("undo").label("Undo").xsmall().on_click(cx.listener(|this, _, _, cx| {
                        this.viewer.update(cx, |v, cx| v.undo(cx));
                    })))
                    .child(Button::new("apply-transform").label("Apply transform").xsmall().on_click(cx.listener(
                        move |this, _, _, cx| {
                            this.viewer.update(cx, |v, cx| v.apply_transform(i, cx));
                        },
                    )))
                    .when(self.project.is_some(), |el| {
                        el.child(Button::new("export-pose").label("Export pose").xsmall().on_click(cx.listener(
                            move |this, _, _, cx| {
                                this.status = Some(match this.export_pose(i, cx) {
                                    Ok(s) => s,
                                    Err(e) => format!("{e:#}"),
                                });
                                cx.notify();
                            },
                        )))
                    }),
            );
        }
        if viewer.entities().len() > 1 {
            scene = scene.child(div().text_xs().text_color(muted()).child(
                "Click an entity to make it active · Align superposes a row onto the active entity · Ctrl+Z undoes",
            ));
        }
        scene
    }

    fn render_left(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let tab = self.config.left_tab;
        let mut panel = div()
            .id("left-drawer")
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .h_full()
            .min_w_0()
            .p_2()
            .overflow_y_scroll();

        // Project.
        let mut project_section = section_body();
        match &self.project {
            _ if tab != LeftTab::Project => {}
            Some(project) => {
                project_section = project_section
                    .child(div().text_base().font_weight(FontWeight::SEMIBOLD).child(project.name()))
                    .child(
                        div()
                            .text_xs()
                            .text_color(muted())
                            .child(project.root().display().to_string()),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child(
                                Button::new("switch-project")
                                    .label("Open project…")
                                    .small()
                                    .on_click(cx.listener(|this, _, _, cx| this.prompt_open_project(cx))),
                            )
                            .child(
                                Button::new("close-project")
                                    .label("Close")
                                    .small()
                                    .on_click(cx.listener(|this, _, _, cx| this.close_project(cx))),
                            ),
                    );
            }
            None => {
                project_section = project_section
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child(div().flex_1().child(Input::new(&self.new_project_name).small()))
                            .child(
                                Button::new("create-project")
                                    .label("Create")
                                    .primary()
                                    .small()
                                    .on_click(cx.listener(|this, _, window, cx| this.create_project(window, cx))),
                            ),
                    )
                    .child(
                        Button::new("open-project")
                            .label("Open project folder…")
                            .small()
                            .on_click(cx.listener(|this, _, _, cx| this.prompt_open_project(cx))),
                    );
                let recent: Vec<PathBuf> = self
                    .config
                    .recent_projects
                    .iter()
                    .filter(|p| Project::is_project(p))
                    .cloned()
                    .collect();
                if !recent.is_empty() {
                    project_section = project_section.child(div().pt_2().text_xs().text_color(muted()).child("Recent"));
                    for (i, path) in recent.into_iter().enumerate() {
                        let label = path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        project_section = project_section.child(
                            list_row(("recent", i), label, false)
                                .on_click(cx.listener(move |this, _, _, cx| this.open_project(path.clone(), cx))),
                        );
                    }
                }
            }
        }
        if tab == LeftTab::Project {
            panel = panel.child(project_section);
            panel = panel
                .child(heading("Example"))
                .child(div().text_xs().text_color(muted()).child(
                    "HIV-1 protease with its inhibitor indinavir (1HSG), already split into receptor and ligand with a search box, \
                     ready to dock. Creates a project if none is open.",
                ))
                .child(
                    Button::new("load-example")
                        .label("Load the example")
                        .primary()
                        .small()
                        .on_click(cx.listener(|this, _, _, cx| this.load_example(cx))),
                );
        }

        if let (LeftTab::Project, Some(project)) = (tab, &self.project) {
            // Structures: click adds to the scene, shift-click replaces it.
            let in_scene: Vec<Id> = self
                .viewer
                .read(cx)
                .entities()
                .iter()
                .filter_map(|e| e.structure_id)
                .collect();
            let mut structures = section_body();
            for entry in project.structures().unwrap_or_default() {
                let id = entry.id;
                let label = match &entry.source {
                    _ if self.loading.contains_key(&id) => format!("{}  ·  reading…", entry.name),
                    _ if self.load_queue.iter().any(|(q, _)| *q == id) => {
                        format!("{}  ·  queued", entry.name)
                    }
                    Some(src) if src.starts_with("rcsb:") => format!("{}  ·  PDB", entry.name),
                    Some(src) if src.starts_with("pose:") => format!("{}  ·  pose", entry.name),
                    _ => entry.name.clone(),
                };
                structures = structures.child(
                    list_row(("structure", id as usize), label, in_scene.contains(&id)).on_click(cx.listener(
                        move |this, event: &ClickEvent, _, cx| {
                            // Off the UI thread: a trajectory takes seconds to parse.
                            this.load_structure_async(id, event.modifiers().shift, cx);
                        },
                    )),
                );
            }
            structures = structures
                .child(
                    div()
                        .flex()
                        .gap_1()
                        .child(
                            Button::new("add-structure")
                                .label("Add files…")
                                .small()
                                .on_click(cx.listener(|this, _, _, cx| this.prompt_add_structure(cx))),
                        )
                        .child(
                            Button::new("add-folder")
                                .label("Add folder…")
                                .small()
                                .on_click(cx.listener(|this, _, _, cx| this.prompt_add_folder(cx))),
                        ),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(muted())
                        .child("Click a row to add it to the scene, shift-click to replace."),
                );
            panel = panel.child(heading("Structures")).child(structures);
        }

        if tab == LeftTab::Scene {
            if self.viewer.read(cx).entities().is_empty() {
                panel = panel.child(div().text_xs().text_color(muted()).child("The scene is empty. Add a structure from the Project tab."));
            } else {
                let scene = self.render_scene_section(cx);
                panel = panel.child(scene);
            }
        }

        if tab == LeftTab::Measure {
            let measure = self.render_measure_tab(cx);
            panel = panel.child(measure);
        }

        if tab == LeftTab::Jobs {
            if self.project.is_none() {
                panel = panel.child(div().text_xs().text_color(muted()).child("Open a project to run jobs."));
            } else {
                let jobs = self.render_jobs_section(cx);
                panel = panel.child(jobs);
            }
        }

        if tab == LeftTab::Results && self.project.is_some() {
            let mut any = false;
            if let Some(cavities) = self.render_cavities_section(cx) {
                panel = panel.child(cavities);
                any = true;
            }
            if let Some(ensemble) = self.render_ensemble_section(cx) {
                panel = panel.child(ensemble);
                any = true;
            }
            if let Some(docking) = self.render_docking_section(cx) {
                panel = panel.child(docking);
                any = true;
            }
            if !any {
                panel = panel.child(div().text_xs().text_color(muted()).child("No results yet. Run a job from the Jobs tab."));
            }
        }

        if let (LeftTab::Views, Some(project)) = (tab, &self.project) {
            // Views.
            let mut views = section_body();
            for view in project.views().unwrap_or_default() {
                let id = view.id;
                views = views.child(
                    div()
                        .flex()
                        .items_center()
                        .child(div().flex_1().child(list_row(("view", id as usize), view.name.clone(), false).on_click(
                            cx.listener(move |this, _, _, cx| {
                                if let Err(e) = this.apply_view(view.clone(), cx) {
                                    this.status = Some(format!("{e:#}"));
                                    cx.notify();
                                }
                            }),
                        )))
                        .child(
                            Button::new(("delete-view", id as usize))
                                .label("×")
                                .ghost()
                                .xsmall()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    if let Some(p) = &this.project {
                                        p.delete_view(id).ok();
                                    }
                                    cx.notify();
                                })),
                        ),
                );
            }
            views = views.child(
                Button::new("save-view")
                    .label("Save current view")
                    .small()
                    .on_click(cx.listener(|this, _, _, cx| this.quick_save_view(cx))),
            );
            panel = panel.child(views);

            let notes = project.notes().unwrap_or_default();
            if !notes.is_empty() {
                let mut section_notes = section_body();
                for note in notes.iter().rev().take(8) {
                    section_notes = section_notes.child(
                        div()
                            .flex()
                            .flex_col()
                            .child(div().text_sm().child(note.title.clone()))
                            .child(div().text_xs().text_color(muted()).child(note.body.chars().take(140).collect::<String>())),
                    );
                }
                panel = panel.child(heading("Notes")).child(section_notes);
            }
        }

        // Representation of the active entity.
        let viewer = self.viewer.read(cx);
        if tab == LeftTab::Scene && viewer.structure().is_some() {
            let mut scene = section_body();
            let presets = [
                ("default", "Default"),
                ("cartoon+surface", "Cartoon + surface"),
                ("surface", "Surface"),
                ("secondary", "Sec. structure"),
                ("ball", "Ball & stick"),
                ("spacefill", "Spacefill"),
                ("backbone", "Backbone"),
            ];
            let mut row = div().flex().flex_wrap().gap_1();
            for (key, label) in presets {
                row = row.child(
                    Button::new(SharedString::from(format!("preset-{key}")))
                        .label(label)
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| this.apply_preset(key, cx))),
                );
            }
            scene = scene.child(row);
            for (i, c) in viewer.components().iter().enumerate() {
                scene = scene.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .text_xs()
                        .child(format!("{} · {} · {}", c.target.label(), c.representation.label(), c.coloring.label()))
                        .child(
                            Button::new(("remove-component", i))
                                .label("×")
                                .ghost()
                                .xsmall()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.viewer.update(cx, |v, cx| v.remove_component(i, cx));
                                })),
                        ),
                );
            }
            let water = viewer.water_visible();
            scene = scene.child(
                div()
                    .flex()
                    .gap_1()
                    .child({
                        let b = Button::new("water").label("Water").small().on_click(cx.listener(move |this, _, _, cx| {
                            this.viewer.update(cx, |v, cx| v.set_water_visible(!water, cx));
                        }));
                        if water { b.primary() } else { b }
                    })
                    .child(
                        Button::new("reset-view")
                            .label("Reset view")
                            .small()
                            .on_click(cx.listener(|this, _, _, cx| this.viewer.update(cx, |v, cx| v.reset_view(cx)))),
                    )
                    .when(!viewer.selection().is_empty(), |el| {
                        el.child(
                            Button::new("clear-selection")
                                .label("Clear selection")
                                .small()
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.viewer.update(cx, |v, cx| v.set_selection(Vec::new(), cx));
                                })),
                        )
                    }),
            );
            panel = panel.child(heading("Representation (active)")).child(scene);
        }

        let status = self
            .status
            .clone()
            .map(|s| div().p_2().border_t_1().border_color(rgb(0x2a2d35)).text_xs().text_color(rgb(0xc9ccd3)).child(s));

        // Rail: one button per panel; a dot marks running jobs and results.
        let running = !self.running.is_empty();
        let has_results = self.cavities.is_some() || self.docking.is_some() || self.ensemble.is_some();
        let mut rail = div()
            .flex()
            .flex_col()
            .gap_1()
            .w(px(56.0))
            .flex_none()
            .h_full()
            .p_1()
            .bg(rgb(0x14161a))
            .border_r_1()
            .border_color(rgb(0x2a2d35));
        for t in LeftTab::ALL {
            let dot = match t {
                LeftTab::Jobs if running => Some(0x8fb8de),
                LeftTab::Results if has_results => Some(0x7fd18b),
                _ => None,
            };
            rail = rail.child(
                div()
                    .id(SharedString::from(format!("rail-{}", t.label())))
                    .flex()
                    .items_center()
                    .justify_center()
                    .gap_1()
                    .py_1()
                    .rounded_md()
                    .text_xs()
                    .cursor_pointer()
                    .when(t == tab, |d| d.bg(rgb(0x24324d)).text_color(rgb(0xe8eefc)))
                    .when(t != tab, |d| d.hover(|s| s.bg(rgb(0x2c3039))))
                    .child(t.label())
                    .children(dot.map(|c| div().size(px(5.0)).rounded_full().bg(rgb(c))))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.config.left_tab = t;
                        this.config.save();
                        cx.notify();
                    })),
            );
        }

        div()
            .flex()
            .size_full()
            .bg(rgb(0x181a1f))
            .border_r_1()
            .border_color(rgb(0x2a2d35))
            .child(rail)
            .child(div().flex().flex_col().flex_1().min_w_0().h_full().child(panel).children(status))
    }
}

impl Render for Workspace {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Filter text set while a job finished: an input can only be written
        // where a Window is in hand, which is here.
        if let Some(text) = self.pending_filter_text.take() {
            self.pocket_filter
                .update(cx, |input, cx| input.set_value(text, window, cx));
        }
        let toggle = |id: &'static str, label: &'static str| Button::new(id).label(label).ghost().xsmall();
        let toolbar = div()
            .absolute()
            .top_2()
            .left_2()
            .right_2()
            .flex()
            .justify_between()
            .child(toggle("toggle-left", if self.left_open { "◀ Projects" } else { "▶ Projects" }).on_click(
                cx.listener(|this, _, _, cx| {
                    this.left_open = !this.left_open;
                    cx.notify();
                }),
            ))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(toggle("open-settings", "⚙ Settings").on_click(cx.listener(|this, _, window, cx| {
                        this.open_settings(crate::settings::Tab::Llm, window, cx);
                    })))
                    .child(toggle("toggle-right", if self.right_open { "Assistant ▶" } else { "◀ Assistant" }).on_click(
                        cx.listener(|this, _, _, cx| {
                            this.right_open = !this.right_open;
                            cx.notify();
                        }),
                    )),
            );

        let left = resizable_panel()
            .size(px(LEFT_WIDTH))
            .size_range(px(SIDE_MIN)..px(SIDE_MAX))
            .visible(self.left_open)
            .child(
                div()
                    .size_full()
                    .bg(rgb(0x181a1f))
                    .border_r_1()
                    .border_color(rgb(0x2a2d35))
                    .child(self.render_left(cx)),
            );

        let centre = resizable_panel().child(
            div()
                .relative()
                .size_full()
                .child(self.viewer.clone())
                .child(toolbar),
        );

        let right = resizable_panel()
            .size(px(RIGHT_WIDTH))
            .size_range(px(SIDE_MIN)..px(SIDE_MAX))
            .visible(self.right_open)
            .child(
                div()
                    .size_full()
                    .bg(rgb(0x181a1f))
                    .border_l_1()
                    .border_color(rgb(0x2a2d35))
                    .child(self.chat.clone()),
            );

        div()
            .id("workspace")
            .size_full()
            .bg(rgb(0x131418))
            .text_color(rgb(0xc9ccd3))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                this.add_files(paths.paths().to_vec(), cx);
            }))
            .child(
                h_resizable("columns")
                    .with_state(&self.layout)
                    .child(left)
                    .child(centre)
                    .child(right),
            )
            .children(gpui_component::Root::render_dialog_layer(window, cx))
    }
}
