//! Configuration dialogs for the jobs the app starts: one dialog per job with
//! all its parameters, validated on Run and remembered in
//! `Config.job_defaults` for next time (and for the assistant's tools).

use crate::config::JobDefaults;
use jobs::autodock::{self, Covalent, Engine, Warhead};
use crate::workspace::{Workspace, muted};
use gpui::{
    AppContext, Context, Entity, IntoElement, ParentElement, Render, SharedString, Styled, WeakEntity, Window, div,
    px, rgb,
};
use gpui_component::{
    Disableable, Sizable, WindowExt,
    button::{Button, ButtonVariants},
    checkbox::Checkbox,
    input::{Input, InputState},
};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum JobKind {
    Cavities,
    Ensemble,
    Dock,
    AutoDock,
}

impl JobKind {
    fn title(self) -> &'static str {
        match self {
            JobKind::Cavities => "Detect cavities (pyKVFinder)",
            JobKind::Ensemble => "Pockets across ensemble (pyKVFinder)",
            JobKind::Dock => "Dock (FRODOCK)",
            JobKind::AutoDock => "Dock (AutoDock Vina / AutoDock4)",
        }
    }
}

pub struct JobDialog {
    workspace: WeakEntity<Workspace>,
    kind: JobKind,
    probe_out: Entity<InputState>,
    removal_distance: Entity<InputState>,
    volume_cutoff: Entity<InputState>,
    stride: Entity<InputState>,
    top_n: Entity<InputState>,
    cluster_rmsd: Entity<InputState>,
    ligand_adjust: bool,
    around_ligand: bool,
    // AutoDock.
    box_center: [Entity<InputState>; 3],
    box_size: [Entity<InputState>; 3],
    exhaustiveness: Entity<InputState>,
    modes: Entity<InputState>,
    residue: Entity<InputState>,
    engine: Engine,
    covalent: bool,
    warhead: Warhead,
    restrain: bool,
    error: Option<String>,
}

fn number<T: std::str::FromStr + PartialOrd + std::fmt::Display + Copy>(
    input: &Entity<InputState>,
    cx: &Context<JobDialog>,
    what: &str,
    min: T,
) -> Result<T, String> {
    match input.read(cx).value().trim().parse::<T>() {
        Ok(v) if v >= min => Ok(v),
        _ => Err(format!("{what} must be a number, at least {min}")),
    }
}

impl JobDialog {
    fn new(
        workspace: WeakEntity<Workspace>,
        kind: JobKind,
        defaults: JobDefaults,
        search_box: ([f32; 3], [f32; 3]),
        residue: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let field = |window: &mut Window, cx: &mut Context<Self>, value: String, hint: &'static str| {
            cx.new(|cx| InputState::new(window, cx).default_value(value).placeholder(hint))
        };
        let kv = &defaults.kvfinder;
        let axis = |window: &mut Window, cx: &mut Context<Self>, v: f32, hint: &'static str| {
            field(window, cx, format!("{v:.1}"), hint)
        };
        let (center, size) = search_box;
        Self {
            box_center: [axis(window, cx, center[0], "x"), axis(window, cx, center[1], "y"), axis(window, cx, center[2], "z")],
            box_size: [axis(window, cx, size[0], "x Å"), axis(window, cx, size[1], "y Å"), axis(window, cx, size[2], "z Å")],
            exhaustiveness: field(window, cx, defaults.ad_exhaustiveness.to_string(), "search effort, ≥ 1"),
            modes: field(window, cx, defaults.ad_modes.to_string(), "≥ 1"),
            residue: field(window, cx, residue, "A:CYS:481"),
            engine: defaults.ad_engine,
            covalent: defaults.ad_covalent,
            warhead: defaults.ad_warhead,
            workspace,
            kind,
            probe_out: field(window, cx, kv.probe_out.to_string(), "Å, > 0"),
            removal_distance: field(window, cx, kv.removal_distance.to_string(), "Å, > 0"),
            volume_cutoff: field(window, cx, kv.volume_cutoff.to_string(), "Å³, ≥ 0"),
            stride: field(window, cx, defaults.ensemble_stride.to_string(), "1 = every frame"),
            top_n: field(window, cx, defaults.dock_top_n.to_string(), "poses to keep, ≥ 1"),
            cluster_rmsd: field(window, cx, defaults.dock_cluster_rmsd.to_string(), "Å, > 0"),
            ligand_adjust: kv.ligand_adjust,
            around_ligand: defaults.dock_around_ligand,
            restrain: defaults.dock_restrain_to_selection,
            error: None,
        }
    }

    fn label(text: &'static str) -> gpui::Div {
        div().text_xs().text_color(muted()).child(text)
    }

    fn field(label: &'static str, input: &Entity<InputState>) -> gpui::Div {
        div()
            .flex()
            .flex_col()
            .gap_1()
            .flex_1()
            .child(Self::label(label))
            .child(Input::new(input).small())
    }

    /// Validate, remember the values and start the job.
    fn run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.try_run(cx) {
            Ok(()) => window.close_dialog(cx),
            Err(e) => {
                self.error = Some(e);
                cx.notify();
            }
        }
    }

    fn try_run(&mut self, cx: &mut Context<Self>) -> Result<(), String> {
        let Some(ws) = self.workspace.upgrade() else { return Err("the workspace is gone".into()) };
        let kind = self.kind;
        // Read and check everything before changing anything.
        let kv = if matches!(kind, JobKind::Cavities | JobKind::Ensemble) {
            let probe = number::<f32>(&self.probe_out, cx, "Probe out", 0.1)?;
            let removal = number::<f32>(&self.removal_distance, cx, "Removal distance", 0.1)?;
            let volume = number::<f32>(&self.volume_cutoff, cx, "Minimum volume", 0.0)?;
            Some((probe, removal, volume))
        } else {
            None
        };
        let stride = if kind == JobKind::Ensemble {
            Some(number::<usize>(&self.stride, cx, "Stride", 1)?)
        } else {
            None
        };
        let dock = if kind == JobKind::Dock {
            Some((
                number::<usize>(&self.top_n, cx, "Poses to keep", 1)?,
                number::<f32>(&self.cluster_rmsd, cx, "Cluster RMSD", 0.1)?,
            ))
        } else {
            None
        };
        let autodock = if kind == JobKind::AutoDock {
            let mut centre = [0.0f32; 3];
            let mut size = [0.0f32; 3];
            for i in 0..3 {
                let axis = ["x", "y", "z"][i];
                centre[i] = number::<f32>(&self.box_center[i], cx, &format!("Box centre {axis}"), -10000.0)?;
                size[i] = number::<f32>(&self.box_size[i], cx, &format!("Box size {axis}"), 4.0)?;
            }
            let exhaustiveness = number::<u32>(&self.exhaustiveness, cx, "Exhaustiveness", 1)?;
            let modes = number::<u32>(&self.modes, cx, "Poses / runs", 1)?;
            let covalent = if self.covalent {
                let c = Covalent { residue: self.residue.read(cx).value().trim().to_string(), warhead: self.warhead };
                c.parse().map_err(|e| format!("{e:#}"))?;
                Some(c)
            } else {
                None
            };
            Some((centre, size, exhaustiveness, modes, covalent))
        } else {
            None
        };
        let (ligand_adjust, around, restrain) = (self.ligand_adjust, self.around_ligand, self.restrain);
        let engine = self.engine;
        let warhead = self.warhead;

        if kind == JobKind::Dock {
            let ready = ws.read(cx).config.frodock_dir.is_some();
            if !ready {
                return Err("set your FRODOCK folder first (Settings → Tools)".into());
            }
        }
        ws.update(cx, |ws, cx| {
            if let Some((centre, size, exhaustiveness, modes, covalent)) = &autodock {
                let d = &mut ws.config.job_defaults;
                d.ad_engine = engine;
                d.ad_exhaustiveness = *exhaustiveness;
                d.ad_modes = *modes;
                d.ad_covalent = covalent.is_some();
                d.ad_warhead = warhead;
                // The box stays in the scene, drawn, and is reused next time.
                ws.viewer.update(cx, |v, cx| v.set_dock_box(*centre, *size, cx));
            }
            let d = &mut ws.config.job_defaults;
            if let Some((probe, removal, volume)) = kv {
                d.kvfinder.probe_out = probe;
                d.kvfinder.removal_distance = removal;
                d.kvfinder.volume_cutoff = volume;
                if kind == JobKind::Cavities {
                    d.kvfinder.ligand_adjust = ligand_adjust;
                }
            }
            if let Some(stride) = stride {
                d.ensemble_stride = stride;
            }
            if let Some((top_n, cluster)) = dock {
                d.dock_top_n = top_n;
                d.dock_cluster_rmsd = cluster;
                d.dock_around_ligand = around;
                d.dock_restrain_to_selection = restrain;
                ws.dock.top_n = top_n;
                ws.dock.cluster_rmsd = cluster;
                ws.dock.around_ligand = around;
                ws.dock.restrain_to_selection = restrain;
            }
            ws.config.save();
            match kind {
                JobKind::Cavities => ws.run_cavities(cx),
                JobKind::Ensemble => ws.run_ensemble(cx),
                JobKind::Dock => ws.run_dock(cx),
                JobKind::AutoDock => {
                    if let Some((centre, size, exhaustiveness, modes, covalent)) = autodock {
                        ws.run_autodock(
                            autodock::Settings { engine, center: centre, size, exhaustiveness, num_modes: modes, covalent, ..Default::default() },
                            cx,
                        );
                    }
                }
            }
        });
        Ok(())
    }

    fn kv_fields(&self) -> gpui::Div {
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(Self::field("Probe out (Å)", &self.probe_out))
                    .child(Self::field("Removal distance (Å)", &self.removal_distance))
                    .child(Self::field("Minimum volume (Å³)", &self.volume_cutoff)),
            )
            .child(div().text_xs().text_color(muted()).child(
                "A larger probe-out finds shallower pockets; the volume cutoff drops small cavities.",
            ))
    }

    fn dock_roles(&self, cx: &mut Context<Self>) -> gpui::Div {
        struct Row {
            name: String,
            detail: String,
            radius: f32,
        }
        let Some(ws) = self.workspace.upgrade() else { return div() };
        let rows: Vec<Row> = {
            let viewer = ws.read(cx).viewer.read(cx);
            viewer
                .entities()
                .iter()
                .map(|e| Row {
                    name: e.name.clone(),
                    detail: format!(
                        "{} atoms · {} chain{} · {:.0} Å radius{}",
                        e.structure.atoms.len(),
                        e.structure.chain_count(),
                        if e.structure.chain_count() == 1 { "" } else { "s" },
                        e.radius(),
                        // A multi-frame entity is docked in the conformation on screen.
                        if e.structure.frame_count() > 1 {
                            format!(" · frame {} of {}", e.frame + 1, e.structure.frame_count())
                        } else {
                            String::new()
                        }
                    ),
                    radius: e.radius(),
                })
                .collect()
        };
        let pair = ws.read(cx).dock_pair(cx);
        let mut list = div().flex().flex_col().gap_1().child(Self::label(
            "FRODOCK docks two bodies; the wider one is the receptor unless you choose. Anything else in the scene is left out.",
        ));
        for (i, row) in rows.iter().enumerate() {
            let role = pair.and_then(|(r, l)| if i == r { Some("receptor") } else if i == l { Some("ligand") } else { None });
            let (name_r, name_l) = (row.name.clone(), row.name.clone());
            let (ws_r, ws_l) = (self.workspace.clone(), self.workspace.clone());
            list = list.child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(div().text_sm().child(row.name.clone()))
                            .child(div().text_xs().text_color(muted()).child(row.detail.clone())),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child({
                                let b = Button::new(("job-receptor", i)).label("receptor").xsmall().on_click(
                                    cx.listener(move |_, _, _, cx| {
                                        let n = name_r.clone();
                                        ws_r.update(cx, |ws, cx| {
                                            if ws.dock.ligand.as_deref() == Some(n.as_str()) {
                                                ws.dock.ligand = None;
                                            }
                                            ws.dock.receptor = Some(n);
                                            cx.notify();
                                        })
                                        .ok();
                                        cx.notify();
                                    }),
                                );
                                if role == Some("receptor") { b.primary() } else { b }
                            })
                            .child({
                                let b = Button::new(("job-ligand", i)).label("ligand").xsmall().on_click(
                                    cx.listener(move |_, _, _, cx| {
                                        let n = name_l.clone();
                                        ws_l.update(cx, |ws, cx| {
                                            if ws.dock.receptor.as_deref() == Some(n.as_str()) {
                                                ws.dock.receptor = None;
                                            }
                                            ws.dock.ligand = Some(n);
                                            cx.notify();
                                        })
                                        .ok();
                                        cx.notify();
                                    }),
                                );
                                if role == Some("ligand") { b.primary() } else { b }
                            }),
                    ),
            );
        }
        if let Some((r, l)) = pair
            && rows[l].radius > rows[r].radius
        {
            list = list.child(
                div()
                    .text_xs()
                    .text_color(rgb(0xd8a657))
                    .child("the ligand is wider than the receptor — they will be swapped when the run starts"),
            );
        }
        list
    }
}

impl Render for JobDialog {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let kind = self.kind;
        let mut body = div().flex().flex_col().gap_3();
        match kind {
            JobKind::Cavities => {
                body = body.child(self.kv_fields()).child(
                    Checkbox::new("job-ligand-adjust")
                        .label("Ligand-adjusted: only report cavity space around the bound ligand")
                        .checked(self.ligand_adjust)
                        .on_click(cx.listener(|this, checked: &bool, _, cx| {
                            this.ligand_adjust = *checked;
                            cx.notify();
                        })),
                );
            }
            JobKind::Ensemble => {
                body = body
                    .child(self.kv_fields())
                    .child(Self::field("Analyse every Nth frame", &self.stride))
                    .child(div().text_xs().text_color(muted()).child(
                        "Runs pyKVFinder on each frame, merges pockets that share lining residues and scores them by persistence. \
                         Filters can be edited afterwards in the Ensemble pockets panel.",
                    ));
            }
            JobKind::AutoDock => {
                let mut engines = div().flex().gap_1();
                for e in [Engine::Vina, Engine::AutoDock4] {
                    let b = Button::new(SharedString::from(format!("ad-engine-{}", e.label())))
                        .label(e.label())
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.engine = e;
                            cx.notify();
                        }));
                    engines = engines.child(if self.engine == e { b.primary() } else { b });
                }
                body = body
                    .child(self.dock_roles(cx))
                    .child(engines)
                    .child(Self::label("Search box: centre (Å) and size (Å). Drawn in the viewport; Measure → Docking box can set it from a selection."))
                    .child(div().flex().gap_2().children(self.box_center.iter().map(|i| div().flex_1().child(Input::new(i).small()))))
                    .child(div().flex().gap_2().children(self.box_size.iter().map(|i| div().flex_1().child(Input::new(i).small()))))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(Self::field(
                                if self.engine == Engine::Vina { "Exhaustiveness" } else { "Evaluations are fixed; runs →" },
                                &self.exhaustiveness,
                            ))
                            .child(Self::field(if self.engine == Engine::Vina { "Poses to keep" } else { "Independent runs" }, &self.modes)),
                    )
                    .child(
                        Checkbox::new("ad-covalent")
                            .label("Covalent docking (bond the ligand to a residue)")
                            .checked(self.covalent)
                            .on_click(cx.listener(|this, checked: &bool, _, cx| {
                                this.covalent = *checked;
                                cx.notify();
                            })),
                    );
                if self.covalent {
                    // Only the groups that can bond to the residue typed above.
                    let residue = self.residue.read(cx).value().to_string();
                    let name = residue.split(':').nth(1).unwrap_or_default().to_string();
                    let usable = Warhead::for_residue(&name);
                    let mut warheads = div().flex().flex_wrap().gap_1();
                    for w in usable.iter().copied() {
                        let b = Button::new(SharedString::from(format!("ad-warhead-{}", w.key())))
                            .label(w.label())
                            .xsmall()
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.warhead = w;
                                cx.notify();
                            }));
                        warheads = warheads.child(if self.warhead == w { b.primary() } else { b });
                    }
                    body = body
                        .child(Self::field(
                            "Receptor residue: Cys, Lys, Ser or Tyr (chain:NAME:number, e.g. A:CYS:481)",
                            &self.residue,
                        ))
                        .child(Self::label("Reactive group on the ligand:"))
                        .child(if usable.is_empty() {
                            div().text_xs().text_color(rgb(0xd8a657)).child("type a residue such as A:LYS:258")
                        } else {
                            warheads
                        })
                        .child(div().text_xs().text_color(muted()).child(
                            "The ligand is bonded to the residue's reactive atom and docked as its flexible side chain. Poses are ranked without a binding energy.",
                        ));
                }
            }
            JobKind::Dock => {
                let (selected, frodock) = self
                    .workspace
                    .upgrade()
                    .map(|ws| {
                        let ws = ws.read(cx);
                        let viewer = ws.viewer.read(cx);
                        let selected: usize = ws
                            .dock_pair(cx)
                            .map(|(r, l)| viewer.entities()[r].selection.len() + viewer.entities()[l].selection.len())
                            .unwrap_or(0);
                        (selected, ws.config.frodock_dir.is_some())
                    })
                    .unwrap_or((0, false));
                body = body
                    .child(self.dock_roles(cx))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(Self::field("Poses to keep", &self.top_n))
                            .child(Self::field("Cluster RMSD (Å)", &self.cluster_rmsd)),
                    )
                    .child(
                        Checkbox::new("job-around")
                            .label("Local search around the ligand's current position")
                            .checked(self.around_ligand)
                            .on_click(cx.listener(|this, checked: &bool, _, cx| {
                                this.around_ligand = *checked;
                                cx.notify();
                            })),
                    )
                    .child(
                        Checkbox::new("job-restrain")
                            .label(format!("Restrain the interface to the viewer selection ({selected} atoms)"))
                            .checked(self.restrain)
                            .on_click(cx.listener(|this, checked: &bool, _, cx| {
                                this.restrain = *checked;
                                cx.notify();
                            })),
                    );
                if !frodock {
                    body = body.child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xd8a657))
                            .child("FRODOCK is not redistributed: set your own copy in Settings → Tools."),
                    );
                }
            }
        }
        if let Some(e) = &self.error {
            body = body.child(div().text_xs().text_color(rgb(0xff9b94)).child(e.clone()));
        }
        let running_blocked = kind == JobKind::Dock && self.workspace.upgrade().is_some_and(|ws| ws.read(cx).config.frodock_dir.is_none());
        body.child(
            div()
                .flex()
                .justify_end()
                .gap_2()
                .child(
                    Button::new("job-cancel")
                        .label("Cancel")
                        .small()
                        .on_click(|_, window, cx| window.close_dialog(cx)),
                )
                .child(
                    Button::new("job-run")
                        .label("Run")
                        .small()
                        .primary()
                        .disabled(running_blocked)
                        .on_click(cx.listener(|this, _, window, cx| this.run(window, cx))),
                ),
        )
    }
}

impl Workspace {
    pub fn open_job_dialog(&mut self, kind: JobKind, window: &mut Window, cx: &mut Context<Self>) {
        let weak = cx.entity().downgrade();
        let defaults = self.config.job_defaults.clone();
        // What the AutoDock form opens with, worked out here because the
        // dialog cannot read the workspace while this click is being handled.
        let (search_box, residue) = match self.dock_pair(cx) {
            Some((r, l)) => {
                let covalent = defaults.ad_covalent.then(|| Covalent {
                    residue: self.suggested_residue(cx),
                    warhead: defaults.ad_warhead,
                });
                (self.autodock_box(r, l, covalent.as_ref(), cx), self.suggested_residue(cx))
            }
            None => (([0.0; 3], [22.0; 3]), "A:CYS:".to_string()),
        };
        let view = cx.new(|cx| JobDialog::new(weak, kind, defaults, search_box, residue, window, cx));
        window.open_dialog(cx, move |dialog, _, _| {
            dialog
                .title(SharedString::from(kind.title()))
                .w(px(560.0))
                .close_button(true)
                .child(view.clone())
        });
    }
}
