//! Window layout: projects/scene drawer (left), viewport (centre), assistant
//! drawer (right). Owns the open project and the user config.

use crate::chat::ChatPanel;
use crate::config::Config;
use crate::jobs_ui::CavityResults;
use crate::tools;
use crate::viewer::{EntityState, ViewState, Viewer, ViewerEvent};
use anyhow::{Context as _, Result};
use gpui::{
    AppContext, ClickEvent, Context, Entity, ExternalPaths, Focusable, FontWeight, Hsla, IntoElement, ParentElement,
    PathPromptOptions, Render, SharedString, Styled, Subscription, Window, div, prelude::*, px, rgb,
};
use gpui_component::{
    Sizable,
    button::{Button, ButtonVariants},
    input::{Input, InputEvent, InputState},
};
use molecule::Structure;
use molecule::align::AlignOptions;
use project::{Id, Project, ViewEntry};
use render::{Coloring, Component, Representation, Target};
use std::collections::HashMap;
use std::path::PathBuf;

const LEFT_WIDTH: f32 = 300.0;
const RIGHT_WIDTH: f32 = 400.0;

pub struct Workspace {
    pub viewer: Entity<Viewer>,
    pub chat: Entity<ChatPanel>,
    pub project: Option<Project>,
    pub config: Config,
    new_project_name: Entity<InputState>,
    left_open: bool,
    right_open: bool,
    pub(crate) status: Option<String>,
    /// `docker version` result (checked once at startup).
    pub(crate) docker: Option<Result<String, String>>,
    /// Running jobs → container name (for cancellation).
    pub(crate) running: HashMap<Id, String>,
    pub(crate) cavities: Option<CavityResults>,
    pub(crate) kv_ligand_adjust: bool,
    _subscriptions: Vec<Subscription>,
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
            left_open: true,
            right_open: true,
            status: None,
            docker: None,
            running: HashMap::new(),
            cavities: None,
            kv_ligand_adjust: false,
            _subscriptions: subscriptions,
        };
        this.check_docker(cx);
        let model = this.config.provider().map(|p| p.model.clone()).unwrap_or_default();
        this.chat.update(cx, |c, cx| c.set_model_field(model, window, cx));

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

    fn set_project(&mut self, project: Project, cx: &mut Context<Self>) {
        self.config.touch_recent(project.root().to_path_buf());
        let first = project.structures().ok().and_then(|s| s.first().map(|e| e.id));
        self.project = Some(project);
        self.running.clear();
        self.restore_cavity_results();
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
        self.viewer.update(cx, |v, cx| v.clear(cx));
        self.chat.update(cx, |c, cx| c.new_conversation(cx));
        cx.notify();
    }

    // ------------------------------------------------------------ structures

    fn read_structure(&self, id: Id) -> Result<Structure> {
        let project = self.project.as_ref().context("no project is open")?;
        let entry = project
            .structure(id)?
            .with_context(|| format!("no structure with id {id}"))?;
        let mut structure = Structure::load(project.structure_path(&entry))?;
        structure.name = entry.name.clone();
        Ok(structure)
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
            if let Err(e) = self.add_structure_to_scene(id, cx) {
                self.status = Some(format!("{e:#}"));
            }
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
        let mut scene = section("Scene");
        for (i, e) in viewer.entities().iter().enumerate() {
            let is_active = Some(i) == active;
            let visible = e.visible;
            let mut row = div()
                .flex()
                .items_center()
                .gap_1()
                .child(
                    Button::new(("vis", i))
                        .label(if visible { "●" } else { "○" })
                        .ghost()
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.viewer.update(cx, |v, cx| v.set_visible(i, !visible, cx));
                        })),
                )
                .child(div().flex_1().child(
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
        let mut panel = div()
            .id("left-drawer")
            .flex()
            .flex_col()
            .gap_4()
            .w(px(LEFT_WIDTH))
            .h_full()
            .p_3()
            .pt_10()
            .overflow_y_scroll()
            .bg(rgb(0x181a1f))
            .border_r_1()
            .border_color(rgb(0x2a2d35));

        // Project.
        let mut project_section = section("Project");
        match &self.project {
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
                                    .label("Open…")
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
        panel = panel.child(project_section);

        if let Some(project) = &self.project {
            // Structures: click adds to the scene, shift-click replaces it.
            let in_scene: Vec<Id> = self
                .viewer
                .read(cx)
                .entities()
                .iter()
                .filter_map(|e| e.structure_id)
                .collect();
            let mut structures = section("Structures");
            for entry in project.structures().unwrap_or_default() {
                let id = entry.id;
                let label = match &entry.source {
                    Some(src) if src.starts_with("rcsb:") => format!("{}  ·  PDB", entry.name),
                    Some(src) if src.starts_with("pose:") => format!("{}  ·  pose", entry.name),
                    _ => entry.name.clone(),
                };
                structures = structures.child(
                    list_row(("structure", id as usize), label, in_scene.contains(&id)).on_click(cx.listener(
                        move |this, event: &ClickEvent, _, cx| {
                            let result = if event.modifiers().shift {
                                this.load_structure(id, cx)
                            } else {
                                this.add_structure_to_scene(id, cx).map(|_| ())
                            };
                            if let Err(e) = result {
                                this.status = Some(format!("{e:#}"));
                            }
                        },
                    )),
                );
            }
            structures = structures
                .child(
                    Button::new("add-structure")
                        .label("Add file…")
                        .small()
                        .on_click(cx.listener(|this, _, _, cx| this.prompt_add_structure(cx))),
                )
                .child(div().text_xs().text_color(muted()).child("Click adds to the scene · Shift-click replaces it"));
            panel = panel.child(structures);
        }

        if !self.viewer.read(cx).entities().is_empty() {
            let scene = self.render_scene_section(cx);
            panel = panel.child(scene);
        }

        if self.project.is_some() {
            let jobs = self.render_jobs_section(cx);
            panel = panel.child(jobs);
            if let Some(cavities) = self.render_cavities_section(cx) {
                panel = panel.child(cavities);
            }
        }

        if let Some(project) = &self.project {
            // Views.
            let mut views = section("Views");
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
                let mut section_notes = section("Notes");
                for note in notes.iter().rev().take(8) {
                    section_notes = section_notes.child(
                        div()
                            .flex()
                            .flex_col()
                            .child(div().text_sm().child(note.title.clone()))
                            .child(div().text_xs().text_color(muted()).child(note.body.chars().take(140).collect::<String>())),
                    );
                }
                panel = panel.child(section_notes);
            }
        }

        // Representation of the active entity.
        let viewer = self.viewer.read(cx);
        if viewer.structure().is_some() {
            let mut scene = section("Representation (active)");
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
            panel = panel.child(scene);
        }

        if let Some(status) = &self.status {
            panel = panel.child(div().text_xs().text_color(rgb(0xc9ccd3)).child(status.clone()));
        }

        let viewer = self.viewer.read(cx);
        panel.child(div().flex_1()).child(
            section("Renderer")
                .child(div().text_xs().text_color(muted()).child(viewer.adapter_name()))
                .child(div().text_xs().text_color(muted()).child(format!(
                    "{:.2} ms / frame · drag: rotate · right/shift-drag: pan · wheel: zoom · G/R: move/rotate active",
                    viewer.last_render().as_secs_f64() * 1000.0
                ))),
        )
    }
}

impl Render for Workspace {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
            .child(toggle("toggle-right", if self.right_open { "Assistant ▶" } else { "◀ Assistant" }).on_click(
                cx.listener(|this, _, _, cx| {
                    this.right_open = !this.right_open;
                    cx.notify();
                }),
            ));

        let centre = div()
            .relative()
            .flex_1()
            .h_full()
            .child(self.viewer.clone())
            .child(toolbar);

        div()
            .id("workspace")
            .flex()
            .size_full()
            .bg(rgb(0x131418))
            .text_color(rgb(0xc9ccd3))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                this.add_files(paths.paths().to_vec(), cx);
            }))
            .when(self.left_open, |el| el.child(self.render_left(cx)))
            .child(centre)
            .when(self.right_open, |el| {
                el.child(
                    div()
                        .w(px(RIGHT_WIDTH))
                        .h_full()
                        .bg(rgb(0x181a1f))
                        .border_l_1()
                        .border_color(rgb(0x2a2d35))
                        .child(self.chat.clone()),
                )
            })
    }
}
