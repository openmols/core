//! Docker jobs in the workspace: starting/cancelling jobs, the Tools & Jobs
//! panel, and the pyKVFinder cavity workflow (inputs → job → cavity entity +
//! results table).

use crate::workspace::{Workspace, muted, section};
use anyhow::{Context as _, Result, anyhow};
use futures::channel::oneshot;
use glam::Mat4;
use gpui::{Context, ParentElement, Styled, div, prelude::*, rgb};
use gpui_component::{
    Disableable, Sizable,
    button::{Button, ButtonVariants},
};
use jobs::kvfinder::{self, Cavity, Settings};
use molecule::AtomKind;
use project::Id;
use render::{Coloring, Component, Representation, Target};
use serde_json::{Value, json};
use std::time::Duration;

/// Results of the most recent cavity detection, shown in the Cavities panel.
#[derive(Clone, Debug)]
pub struct CavityResults {
    pub job: Id,
    /// Name of the receptor entity and of the cavity entity in the scene.
    pub receptor: String,
    pub cavity_entity: String,
    pub cavities: Vec<Cavity>,
}

/// What a finished job hands back: a text summary (for the chat) and JSON
/// stored as the job result.
type Finish = Box<dyn FnOnce(&mut Workspace, &mut Context<Workspace>) -> Result<(String, Value)>>;

impl Workspace {
    pub fn check_docker(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let (tx, rx) = oneshot::channel();
            std::thread::spawn(move || {
                let _ = tx.send(jobs::docker_version().map_err(|e| format!("{e:#}")));
            });
            if let Ok(result) = rx.await {
                this.update(cx, |this, cx| {
                    this.docker = Some(result);
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    /// Run a prepared job in Docker (building the image first if needed).
    /// Resolves to the finish step's summary once the job is over.
    fn run_job(&mut self, id: Id, spec: jobs::RunSpec, finish: Finish, cx: &mut Context<Self>) -> oneshot::Receiver<Result<String>> {
        let (tx, rx) = oneshot::channel();
        if let Some(project) = &self.project {
            project.set_job_status(id, "running", None).ok();
        }
        self.running.insert(id, spec.name.clone());
        cx.spawn(async move |this, cx| {
            let (done_tx, mut done_rx) = oneshot::channel();
            let worker_spec = spec.clone();
            std::thread::spawn(move || {
                let result = (|| -> Result<i32> {
                    jobs::docker_version()?;
                    if !jobs::image_present(&worker_spec.image) {
                        jobs::build_image(&worker_spec.image, &worker_spec.host_dir().join("build.log"))?;
                    }
                    jobs::run(&worker_spec)
                })();
                let _ = done_tx.send(result);
            });
            // Poll so the panel's log tail stays fresh while the job runs.
            let result = loop {
                match done_rx.try_recv() {
                    Ok(Some(r)) => break r,
                    Ok(None) => {}
                    Err(_) => break Err(anyhow!("job worker stopped unexpectedly")),
                }
                cx.background_executor().timer(Duration::from_millis(500)).await;
                if this.update(cx, |_, cx| cx.notify()).is_err() {
                    return;
                }
            };
            this.update(cx, |ws, cx| {
                let cancelled = ws.running.remove(&id).is_none();
                let outcome = match result {
                    _ if cancelled => Err(anyhow!("job {id} was cancelled")),
                    Ok(0) => finish(ws, cx),
                    Ok(code) => Err(anyhow!("job {id} exited with code {code}; see jobs/{id}/log.txt")),
                    Err(e) => Err(e),
                };
                let (status, stored, reply) = match outcome {
                    Ok((summary, value)) => ("done", value, Ok(summary)),
                    Err(e) => (
                        if cancelled { "cancelled" } else { "failed" },
                        json!({ "error": format!("{e:#}") }),
                        Err(e),
                    ),
                };
                if let Some(project) = &ws.project {
                    project.set_job_status(id, status, Some(&stored)).ok();
                }
                cx.notify();
                let _ = tx.send(reply);
            })
            .ok();
        })
        .detach();
        cx.notify();
        rx
    }

    pub fn cancel_job(&mut self, id: Id, cx: &mut Context<Self>) {
        if let Some(name) = self.running.remove(&id) {
            std::thread::spawn(move || {
                jobs::stop(&name).ok();
            });
            cx.notify();
        }
    }

    // ------------------------------------------------------------ pyKVFinder

    /// Detect cavities in an entity with pyKVFinder.
    pub fn start_kvfinder(
        &mut self,
        entity: usize,
        settings: Settings,
        cx: &mut Context<Self>,
    ) -> Result<oneshot::Receiver<Result<String>>> {
        let project = self.project.as_ref().context("open or create a project first")?;
        let (name, structure, transform, structure_id) = {
            let viewer = self.viewer.read(cx);
            let e = viewer.entities().get(entity).context("no such entity")?;
            (e.name.clone(), e.structure.clone(), e.transform, e.structure_id)
        };
        let polymer = |i: usize| structure.kind(i) == AtomKind::Polymer;
        anyhow::ensure!((0..structure.atoms.len()).any(polymer), "{name} has no polymer atoms to search");
        let job = project.create_job(
            "pykvfinder",
            &json!({ "entity": name, "structure_id": structure_id, "settings": settings }),
        )?;
        let dir = project.job_path(&job);
        std::fs::write(dir.join("receptor.pdb"), molecule::write_pdb(&structure, Mat4::IDENTITY, polymer))?;
        if settings.ligand_adjust {
            let ligand = |i: usize| structure.kind(i) == AtomKind::Ligand;
            anyhow::ensure!(
                (0..structure.atoms.len()).any(ligand),
                "{name} has no ligand for ligand-adjusted detection"
            );
            std::fs::write(dir.join("ligand.pdb"), molecule::write_pdb(&structure, Mat4::IDENTITY, ligand))?;
        }
        let spec = jobs::RunSpec {
            image: jobs::PYKVFINDER,
            project_root: project.root().to_path_buf(),
            job_dir: job.dir(),
            command: kvfinder::command(&settings),
            name: format!("openmols-job-{}-{}", job.id, std::process::id()),
        };
        let job_id = job.id;
        let finish: Finish = Box::new(move |ws, cx| ws.finish_kvfinder(job_id, name, transform, cx));
        Ok(self.run_job(job.id, spec, finish, cx))
    }

    /// `detect_cavities` tool entry point: resolves the entity and settings.
    pub fn start_kvfinder_tool(&mut self, args: &Value, cx: &mut Context<Self>) -> Result<oneshot::Receiver<Result<String>>> {
        let viewer = self.viewer.read(cx);
        let index = match args["entity"].as_str().map(str::to_string).or_else(|| args["entity"].as_u64().map(|n| n.to_string())) {
            Some(key) => viewer.find_entity(&key).with_context(|| format!("no entity '{key}'"))?,
            None => viewer.active_index().context("the scene is empty")?,
        };
        let defaults = Settings::default();
        let settings = Settings {
            ligand_adjust: args["ligand_adjust"].as_bool().unwrap_or(false),
            probe_out: args["probe_out"].as_f64().map(|v| v as f32).unwrap_or(defaults.probe_out),
            volume_cutoff: args["volume_cutoff"].as_f64().map(|v| v as f32).unwrap_or(defaults.volume_cutoff),
            ..defaults
        };
        self.start_kvfinder(index, settings, cx)
    }

    fn finish_kvfinder(
        &mut self,
        job: Id,
        receptor: String,
        transform: Mat4,
        cx: &mut Context<Self>,
    ) -> Result<(String, Value)> {
        let project = self.project.as_ref().context("project closed")?;
        let dir = project.job_path(&project.job(job)?.context("job record missing")?);
        let cavities = kvfinder::read_results(&dir)?;
        let pdb = kvfinder::read_cavity_pdb(&dir)?;
        let cavity_name = format!("{receptor} cavities");
        let entry = project.add_structure_bytes(
            &format!("cavities_job{job}.pdb"),
            pdb.as_bytes(),
            &cavity_name,
            Some(&format!("kvfinder:job{job}")),
        )?;
        let index = self.add_structure_to_scene(entry.id, cx)?;
        let entity_name = self.viewer.read(cx).entities()[index].name.clone();
        self.viewer.update(cx, |v, cx| {
            // Cavities were computed in the receptor's own coordinates.
            v.set_transform(index, transform, cx);
            v.set_components(
                vec![Component::new(Target::All, Representation::GaussianSurface, Coloring::Residue)],
                cx,
            );
        });

        let mut summary = format!("{} cavities found in {receptor}:\n", cavities.len());
        for c in cavities.iter().take(8) {
            summary.push_str(&cavity_line(c));
            summary.push('\n');
        }
        if let Some(top) = cavities.first() {
            summary.push_str(&format!("{} lining residues: {}\n", top.id, top.residue_labels()));
        }
        let value = json!({
            "cavities": cavities,
            "cavity_structure_id": entry.id,
            "cavity_entity": entity_name,
            "receptor": receptor,
        });
        self.cavities = Some(CavityResults {
            job,
            receptor,
            cavity_entity: entity_name,
            cavities,
        });
        Ok((summary, value))
    }

    /// Restore the cavities panel from the latest finished pyKVFinder job.
    pub fn restore_cavity_results(&mut self) {
        self.cavities = self.project.as_ref().and_then(|p| {
            let job = p
                .jobs()
                .ok()?
                .into_iter()
                .find(|j| j.tool == "pykvfinder" && j.status == "done")?;
            let result = job.result?;
            Some(CavityResults {
                job: job.id,
                receptor: result["receptor"].as_str()?.to_string(),
                cavity_entity: result["cavity_entity"].as_str().unwrap_or_default().to_string(),
                cavities: serde_json::from_value(result["cavities"].clone()).ok()?,
            })
        });
    }

    pub fn cavity_summary(&self) -> String {
        match &self.cavities {
            None => "No cavity detection has been run in this project.".into(),
            Some(r) => {
                let mut out = format!("cavities of {} (job {}):\n", r.receptor, r.job);
                for c in &r.cavities {
                    out.push_str(&cavity_line(c));
                    out.push_str(&format!("\n  lining: {}\n", c.residue_labels()));
                }
                out
            }
        }
    }

    /// Select and focus the residues lining a cavity in the receptor entity.
    pub fn focus_cavity(&mut self, cavity: usize, cx: &mut Context<Self>) {
        let Some(results) = &self.cavities else { return };
        let Some(c) = results.cavities.get(cavity).cloned() else { return };
        let receptor = results.receptor.clone();
        let cavity_entity = results.cavity_entity.clone();
        let viewer = self.viewer.read(cx);
        let cavity_index = viewer.find_entity(&cavity_entity);
        let Some(index) = viewer.find_entity(&receptor) else {
            self.status = Some(format!("Add {receptor} to the scene to inspect its cavities"));
            cx.notify();
            return;
        };
        let s = viewer.entities()[index].structure.clone();
        let atoms: Vec<usize> = s
            .residues
            .iter()
            .filter(|r| {
                c.residues
                    .iter()
                    .any(|l| l.chain == r.chain && l.number == r.number.to_string())
            })
            .flat_map(|r| r.atoms.start as usize..r.atoms.end as usize)
            .collect();
        self.viewer.update(cx, |v, cx| {
            if let Some(ci) = cavity_index {
                v.set_visible(ci, true, cx);
            }
            v.set_active(index, cx);
            v.set_selection(atoms.clone(), cx);
            v.focus_atoms(&atoms, cx);
        });
        self.status = Some(format!("{} · {}", cavity_line(&c), c.residue_labels()));
        cx.notify();
    }

    // ---------------------------------------------------------------- panels

    pub fn render_jobs_section(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut panel = section("Tools & jobs");
        let docker = match &self.docker {
            None => "Docker: checking…".to_string(),
            Some(Ok(v)) => format!("Docker {v} ✓"),
            Some(Err(e)) => format!("Docker unavailable: {e}"),
        };
        panel = panel.child(div().text_xs().text_color(muted()).child(docker));

        let has_entity = self.viewer.read(cx).active_index().is_some();
        let ligand_adjust = self.kv_ligand_adjust;
        panel = panel.child(
            div()
                .flex()
                .flex_wrap()
                .gap_1()
                .child(
                    Button::new("detect-cavities")
                        .label("Detect cavities")
                        .primary()
                        .xsmall()
                        .disabled(!has_entity || self.project.is_none())
                        .on_click(cx.listener(|this, _, _, cx| {
                            let Some(index) = this.viewer.read(cx).active_index() else { return };
                            let settings = Settings {
                                ligand_adjust: this.kv_ligand_adjust,
                                ..Default::default()
                            };
                            match this.start_kvfinder(index, settings, cx) {
                                Ok(rx) => {
                                    cx.spawn(async move |this, cx| {
                                        if let Ok(Err(e)) = rx.await {
                                            this.update(cx, |this, cx| {
                                                this.status = Some(format!("{e:#}"));
                                                cx.notify();
                                            })
                                            .ok();
                                        }
                                    })
                                    .detach();
                                }
                                Err(e) => this.status = Some(format!("{e:#}")),
                            }
                            cx.notify();
                        })),
                )
                .child({
                    let b = Button::new("ligand-adjust")
                        .label("Ligand-adjusted")
                        .xsmall()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.kv_ligand_adjust = !this.kv_ligand_adjust;
                            cx.notify();
                        }));
                    if ligand_adjust { b.primary() } else { b }
                }),
        );

        let Some(project) = &self.project else { return panel };
        for job in project.jobs().unwrap_or_default().into_iter().take(6) {
            let id = job.id;
            let label = jobs::tool(&job.tool).map(|t| t.label).unwrap_or(&job.tool);
            let target = job.params["entity"].as_str().unwrap_or_default();
            let running = self.running.contains_key(&id);
            let status = if running { "running" } else { job.status.as_str() };
            let color = match status {
                "done" => 0x7fd18b,
                "running" | "queued" => 0x8fb8de,
                _ => 0xff9b94,
            };
            let mut row = div().flex().flex_col().gap_0p5().child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .text_xs()
                    .child(format!("#{id} {label} · {target}"))
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .items_center()
                            .child(div().text_color(rgb(color)).child(status.to_string()))
                            .when(running, |el| {
                                el.child(
                                    Button::new(("cancel-job", id as usize))
                                        .label("Cancel")
                                        .ghost()
                                        .xsmall()
                                        .on_click(cx.listener(move |this, _, _, cx| this.cancel_job(id, cx))),
                                )
                            }),
                    ),
            );
            if running {
                let tail = jobs::log_tail(&project.job_path(&job).join("log.txt"), 3);
                if !tail.trim().is_empty() {
                    row = row.child(div().text_xs().text_color(muted()).child(tail));
                }
            } else if let Some(error) = job.result.as_ref().and_then(|r| r["error"].as_str()) {
                // First line only; the full message is in the job record/log.
                let short: String = error.lines().next().unwrap_or_default().chars().take(140).collect();
                row = row.child(div().text_xs().text_color(rgb(0xff9b94)).child(short));
            }
            panel = panel.child(row);
        }
        panel
    }

    pub fn render_cavities_section(&mut self, cx: &mut Context<Self>) -> Option<gpui::Div> {
        let results = self.cavities.clone()?;
        let mut panel = section("Cavities").child(
            div()
                .text_xs()
                .text_color(muted())
                .child(format!("{} · job #{} · click to inspect lining residues", results.receptor, results.job)),
        );
        for (i, c) in results.cavities.iter().enumerate().take(12) {
            panel = panel.child(
                div()
                    .id(("cavity", i))
                    .flex()
                    .justify_between()
                    .px_2()
                    .py_0p5()
                    .rounded_md()
                    .text_xs()
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x2c3039)))
                    .child(div().font_weight(gpui::FontWeight::SEMIBOLD).child(c.id.clone()))
                    .child(format!(
                        "{:.0} Å³ · depth {:.1} Å · hyd {:.2}",
                        c.volume,
                        c.max_depth.unwrap_or(0.0),
                        c.avg_hydropathy.unwrap_or(0.0)
                    ))
                    .on_click(cx.listener(move |this, _, _, cx| this.focus_cavity(i, cx))),
            );
        }
        Some(panel)
    }
}

pub fn cavity_line(c: &Cavity) -> String {
    format!(
        "{}: volume {:.0} Å³, area {:.0} Å², max depth {:.1} Å, avg hydropathy {:.2}, {} lining residues",
        c.id,
        c.volume,
        c.area,
        c.max_depth.unwrap_or(0.0),
        c.avg_hydropathy.unwrap_or(0.0),
        c.residues.len()
    )
}
