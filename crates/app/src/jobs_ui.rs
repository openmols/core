//! Docker jobs in the workspace: starting/cancelling jobs, the Tools & Jobs
//! panel, and the pyKVFinder cavity workflow (inputs → job → cavity entity +
//! results table).

use crate::workspace::{Workspace, muted, section, section_body, section_rows, stat_row};
use anyhow::{Context as _, Result, anyhow};
use futures::channel::oneshot;
use glam::Mat4;
use gpui::{Context, ParentElement, Styled, div, prelude::*, rgb};
use gpui_component::{
    Disableable, Sizable,
    button::{Button, ButtonVariants},

    input::Input,
};
use jobs::frodock::{self, Settings as FrodockSettings, Solution};
use jobs::kvfinder::{
    self, Cavity, ConsensusPocket, Criterion, EnsembleFilter, FrameCavity, Metric, ResidueProperties,
    Settings,
};
use molecule::AtomKind;
use project::Id;
use render::{Coloring, Component, Representation, Target};
use serde_json::{Value, json};
use std::time::Duration;

/// Per-residue measures stored beside a finished ensemble job, so filters on
/// them work in a later session without recomputing RMSF over every frame.
const PROPERTIES_FILE: &str = "residue_properties.json";

/// Results of the most recent cavity detection, shown in the Cavities panel.
#[derive(Clone, Debug)]
pub struct CavityResults {
    pub job: Id,
    /// Name of the receptor entity and of the cavity entity in the scene.
    pub receptor: String,
    pub cavity_entity: String,
    pub cavities: Vec<Cavity>,
}

/// Results of the most recent ensemble pocket run.
///
/// Detection is the expensive half (a thousand frames takes an hour), so
/// every cavity pyKVFinder found is kept here and the filters are applied on
/// top. Changing a filter re-merges from `found` in a few tens of
/// milliseconds instead of running the container again.
#[derive(Clone, Debug)]
pub struct EnsemblePockets {
    pub job: Id,
    pub entity: String,
    pub frames: usize,
    pub total_cavities: usize,
    pub pockets: Vec<ConsensusPocket>,
    /// The filter the pockets above came from.
    pub filter: EnsembleFilter,
    /// Every cavity found, over every frame. Empty while the background read
    /// of a restored job is still running.
    pub found: Vec<FrameCavity>,
    /// Per-residue measures the filters can average (rmsf, bfactor, …).
    pub properties: ResidueProperties,
    pub stats: Option<kvfinder::EnsembleStats>,
}

impl EnsemblePockets {
    /// Can the filters be changed without re-running detection?
    pub fn can_refilter(&self) -> bool {
        !self.found.is_empty()
    }
}

/// Just the figures the Ensemble pockets panel draws.
///
/// The panel runs on every frame, and [`EnsemblePockets`] holds every cavity
/// found — tens of megabytes for a thousand-frame run — so the panel must
/// never copy the whole thing.
struct EnsembleSummary {
    entity: String,
    frames: usize,
    total_cavities: usize,
    pocket_count: usize,
    can_refilter: bool,
    stats: Option<kvfinder::EnsembleStats>,
    /// `(rank, persistence, max volume, core residue count)` per listed row.
    rows: Vec<(usize, f64, f64, usize)>,
}

/// What the Dock button will send to FRODOCK.
///
/// FRODOCK docks exactly two bodies, so the pair has to be chosen rather
/// than inferred: before this existed the ligand was whichever entity
/// happened to be active and the receptor was the first other one, which is
/// easy to get backwards and silently ignores anything else in the scene.
///
/// Entities are held by name so the choice survives others being added or
/// removed.
#[derive(Clone, Debug)]
pub struct DockSetup {
    pub receptor: Option<String>,
    pub ligand: Option<String>,
    /// Clustered solutions to keep as poses.
    pub top_n: usize,
    /// RMSD between clusters, Å.
    pub cluster_rmsd: f32,
    /// Search only around the ligand's current position.
    pub around_ligand: bool,
    /// Restrain the interface to the residues selected in the viewer.
    pub restrain_to_selection: bool,
}

impl Default for DockSetup {
    fn default() -> Self {
        let defaults = FrodockSettings::default();
        Self {
            receptor: None,
            ligand: None,
            top_n: defaults.top_n,
            cluster_rmsd: defaults.cluster_rmsd,
            around_ligand: false,
            restrain_to_selection: false,
        }
    }
}

/// One row of the poses table.
#[derive(Clone, Debug)]
pub struct PoseRow {
    /// 1-based; also the frame in the pose entity.
    pub rank: usize,
    /// `score 123.4`, `-10.3 kcal/mol`, …
    pub label: String,
}

/// Results of the most recent docking run (any engine), shown in the Docking panel.
#[derive(Clone, Debug)]
pub struct DockingResults {
    pub job: Id,
    /// `FRODOCK`, `AutoDock Vina`, `AutoDock4 (covalent)`, …
    pub engine: String,
    pub receptor: String,
    pub ligand: String,
    /// Scene entity holding the poses as frames.
    pub pose_entity: String,
    pub poses: Vec<PoseRow>,
}

impl DockingResults {
    pub fn from_frodock(job: Id, receptor: String, ligand: String, pose_entity: String, solutions: &[Solution]) -> Self {
        Self {
            job,
            engine: "FRODOCK".into(),
            receptor,
            ligand,
            pose_entity,
            poses: solutions
                .iter()
                .map(|s| PoseRow { rank: s.rank, label: format!("score {:.1}", s.score) })
                .collect(),
        }
    }
}

/// Per-residue measures a pocket filter can average over its lining
/// residues. RMSF comes from the ensemble's own frames; B-factor from the
/// file. Add further MD-derived measures here and they become filterable by
/// name, with no change to the filter code.
fn residue_properties(structure: &molecule::Structure) -> ResidueProperties {
    let mut properties = ResidueProperties::new();
    let key = |r: &molecule::Residue| (r.chain.clone(), r.number.to_string());

    if structure.frame_count() > 1 {
        // Fit the frames first, so this is internal motion, not tumbling.
        let mut fitted = structure.clone();
        fit_frames_in_place(&mut fitted);
        let rmsf = molecule::analysis::residue_rmsf(&fitted);
        properties.insert(
            "rmsf".into(),
            structure
                .residues
                .iter()
                .zip(&rmsf)
                .map(|(r, v)| (key(r), *v as f64))
                .collect(),
        );
    }

    let b_factors = structure
        .residues
        .iter()
        .map(|r| {
            let range = r.atoms.start as usize..r.atoms.end as usize;
            let mean = if range.is_empty() {
                0.0
            } else {
                let n = range.len() as f32;
                structure.atoms[range].iter().map(|a| a.b_factor).sum::<f32>() / n
            };
            (key(r), mean as f64)
        })
        .collect();
    properties.insert("bfactor".into(), b_factors);
    properties
}

/// Superpose every frame onto the first on Cα atoms.
fn fit_frames_in_place(structure: &mut molecule::Structure) {
    let ca: Vec<usize> = (0..structure.residues.len())
        .filter_map(|r| structure.residue_atom(r, "CA"))
        .collect();
    if ca.len() < 3 || structure.frames.len() < 2 {
        return;
    }
    let target: Vec<glam::Vec3> = ca.iter().map(|&a| structure.frames[0][a]).collect();
    for frame in &mut structure.frames {
        let mobile: Vec<glam::Vec3> = ca.iter().map(|&a| frame[a]).collect();
        let (transform, _) = molecule::align::superpose(&mobile, &target);
        for p in frame.iter_mut() {
            *p = transform.transform_point3(*p);
        }
    }
}

/// Read a `filters` argument: a list of `{metric, min, max}` objects, where
/// `metric` is a built-in name (volume, area, hydropathy, residues,
/// max_depth, persistence) or a per-residue property such as `rmsf`.
fn parse_filters(args: &Value, mut filter: EnsembleFilter) -> EnsembleFilter {
    let Some(list) = args["filters"].as_array() else {
        return filter;
    };
    if args["replace_defaults"].as_bool().unwrap_or(false) {
        filter = EnsembleFilter { criteria: Vec::new(), ..filter };
    }
    for entry in list {
        let Some(name) = entry["metric"].as_str() else { continue };
        let metric = Metric::parse(name);
        if entry["remove"].as_bool().unwrap_or(false) {
            filter = filter.without(&metric);
            continue;
        }
        filter = filter.with(Criterion {
            metric,
            min: entry["min"].as_f64(),
            max: entry["max"].as_f64(),
        });
    }
    filter
}

/// Entity argument as a string, accepting numbers too.
fn entity_key(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// What a finished job hands back: a text summary (for the chat) and JSON
/// stored as the job result.
pub(crate) type Finish = Box<dyn FnOnce(&mut Workspace, &mut Context<Workspace>) -> Result<(String, Value)>>;

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
    pub(crate) fn run_job(&mut self, id: Id, spec: jobs::RunSpec, finish: Finish, cx: &mut Context<Self>) -> oneshot::Receiver<Result<String>> {
        let (tx, rx) = oneshot::channel();
        // Images built from software the user supplies need its folder.
        let context = spec.image.needs_context.then(|| self.config.frodock_dir.clone()).flatten();
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
                        jobs::ensure_image(
                            &worker_spec.image,
                            context.as_deref(),
                            &worker_spec.host_dir().join("build.log"),
                        )?;
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
        let defaults = self.config.job_defaults.kvfinder.clone();
        let settings = Settings {
            ligand_adjust: args["ligand_adjust"].as_bool().unwrap_or(defaults.ligand_adjust),
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

    // ------------------------------------------------------ ensemble pockets

    /// Detect pockets across every frame of an ensemble and merge them into
    /// consensus pockets with a persistence score.
    ///
    /// This is the pocket half of the AR-V7 pipeline (Karatzas, Brotzakis &
    /// Sarimveis, JCTC 2025): pyKVFinder per frame, filter, then group by
    /// shared lining residues.
    pub fn start_ensemble_pockets(
        &mut self,
        entity: usize,
        settings: Settings,
        filter: EnsembleFilter,
        stride: usize,
        cx: &mut Context<Self>,
    ) -> Result<oneshot::Receiver<Result<String>>> {
        let project = self.project.as_ref().context("open or create a project first")?;
        let (name, structure, structure_id) = {
            let viewer = self.viewer.read(cx);
            let e = viewer.entities().get(entity).context("no such entity")?;
            (e.name.clone(), e.structure.clone(), e.structure_id)
        };
        let total = structure.frame_count();
        anyhow::ensure!(
            total > 1,
            "{name} has a single conformation — load an ensemble or trajectory first"
        );
        let polymer = |i: usize| structure.kind(i) == AtomKind::Polymer;
        anyhow::ensure!((0..structure.atoms.len()).any(polymer), "{name} has no polymer atoms");

        let stride = stride.max(1);
        // Per-residue measures the filters can use (mean RMSF over a pocket's
        // lining residues, mean B-factor, ...).
        let properties = residue_properties(&structure);

        let job = project.create_job(
            "pykvfinder_ensemble",
            &json!({
                "entity": name,
                "structure_id": structure_id,
                "frames": total,
                "stride": stride,
                "settings": settings,
                "filter": filter,
                "properties": properties.keys().collect::<Vec<_>>(),
            }),
        )?;
        let dir = project.job_path(&job);
        let frames_dir = dir.join("frames");
        std::fs::create_dir_all(&frames_dir)?;
        // One PDB per sampled frame; the container loops over them.
        let mut scratch = (*structure).clone();
        let mut written = 0usize;
        for frame in (0..total).step_by(stride) {
            scratch.set_frame(frame);
            std::fs::write(
                frames_dir.join(format!("{}.pdb", kvfinder::frame_name(frame))),
                molecule::write_pdb(&scratch, Mat4::IDENTITY, polymer),
            )?;
            written += 1;
        }

        let spec = jobs::RunSpec {
            image: jobs::PYKVFINDER,
            project_root: project.root().to_path_buf(),
            job_dir: job.dir(),
            command: kvfinder::ensemble_command(&settings),
            name: format!("openmols-job-{}-{}", job.id, std::process::id()),
        };
        let job_id = job.id;
        let finish: Finish = Box::new(move |ws, cx| {
            ws.finish_ensemble_pockets(job_id, name, filter, properties, written, cx)
        });
        Ok(self.run_job(job.id, spec, finish, cx))
    }

    fn finish_ensemble_pockets(
        &mut self,
        job: Id,
        entity: String,
        filter: EnsembleFilter,
        properties: ResidueProperties,
        expected_frames: usize,
        _cx: &mut Context<Self>,
    ) -> Result<(String, Value)> {
        let project = self.project.as_ref().context("project closed")?;
        let dir = project.job_path(&project.job(job)?.context("job record missing")?);
        let (found, analysed) =
            kvfinder::read_ensemble(&dir).context("no results — see the job log")?;
        anyhow::ensure!(analysed > 0, "pyKVFinder produced no per-frame results (see jobs/{job}/log.txt)");
        // Keep the per-residue measures with the job, so the filters can be
        // changed in a later session without recomputing RMSF over every frame.
        if let Ok(text) = serde_json::to_string(&properties) {
            let _ = std::fs::write(dir.join(PROPERTIES_FILE), text);
        }
        let total_cavities = found.len();
        let pockets = kvfinder::consensus_pockets_with(&found, analysed, &filter, &properties);
        let stats = kvfinder::ensemble_stats(&found, analysed, &filter, &properties);

        let mut summary = format!(
            "Ensemble pocket detection on {entity}: {total_cavities} cavities across {analysed} frames \
             (of {expected_frames} submitted) → {} consensus pockets after filtering \
             ({} of {total_cavities} cavities passed: {}).\n\
             Every cavity is kept, so the filters can be changed from the panel or with \
             refilter_ensemble_pockets without running detection again.\n",
            pockets.len(),
            stats.passing_all,
            filter.describe(),
        );
        for p in pockets.iter().take(10) {
            summary.push_str(&format!(
                "  #{} persistence {:.0}% ({} frames), max volume {:.0} Å³, hydropathy {:.2} — {}\n",
                p.rank,
                p.persistence * 100.0,
                p.frames.len(),
                p.max_volume,
                p.mean_hydropathy.unwrap_or(0.0),
                p.residue_labels(8),
            ));
        }
        let value = json!({
            "pockets": pockets,
            "entity": entity,
            "frames": analysed,
            "cavities": total_cavities,
            "filter": filter,
        });
        self.ensemble = Some(EnsemblePockets {
            job,
            entity,
            frames: analysed,
            total_cavities,
            pockets,
            filter,
            found,
            properties,
            stats: Some(stats),
        });
        self.set_filter_text(None);
        Ok((summary, value))
    }

    /// Apply a different filter to the cavities already found.
    ///
    /// Detection is not repeated: the cavities are re-merged into consensus
    /// pockets, which takes tens of milliseconds even for an ensemble of a
    /// thousand frames. The new result replaces the job's stored one.
    pub fn refilter_ensemble(&mut self, filter: EnsembleFilter, cx: &mut Context<Self>) -> Result<String> {
        let r = self.ensemble.as_ref().context("no ensemble pocket run to filter")?;
        anyhow::ensure!(
            r.can_refilter(),
            "still reading the cavities of job {} — try again in a moment",
            r.job
        );
        // A name that is not a built-in metric is read as a per-residue
        // property, so a misspelling would reject every cavity and look like
        // a pocket that vanished. Say so instead.
        let unknown = filter.unknown_properties(&r.properties);
        if !unknown.is_empty() {
            let builtin = Metric::builtin()
                .iter()
                .map(Metric::filter_name)
                .collect::<Vec<_>>()
                .join(", ");
            let properties = r
                .properties
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "unknown metric{}: {} — use one of {builtin}, persistence{}",
                if unknown.len() > 1 { "s" } else { "" },
                unknown.join(", "),
                if properties.is_empty() {
                    String::new()
                } else {
                    format!(", or a per-residue property: {properties}")
                }
            );
        }
        // Taken out and put back, so the cavities are never copied.
        let mut r = self.ensemble.take().expect("checked above");
        let frames = r.frames;
        r.pockets = kvfinder::consensus_pockets_with(&r.found, frames, &filter, &r.properties);
        let stats = kvfinder::ensemble_stats(&r.found, frames, &filter, &r.properties);
        let summary = format!(
            "{} of {} cavities pass ({}) → {} pockets",
            stats.passing_all,
            r.found.len(),
            filter.describe(),
            r.pockets.len()
        );

        // Keep the choice: the panel shows it again next time the project opens.
        if let Some(project) = self.project.as_ref() {
            let value = json!({
                "pockets": r.pockets,
                "entity": r.entity,
                "frames": frames,
                "cavities": r.found.len(),
                "filter": filter,
            });
            let _ = project.set_job_status(r.job, "done", Some(&value));
        }
        r.total_cavities = r.found.len();
        r.filter = filter;
        r.stats = Some(stats);
        self.ensemble = Some(r);
        self.status = Some(summary.clone());
        cx.notify();
        Ok(summary)
    }

    /// `detect_pockets_ensemble` tool entry point.
    pub fn start_ensemble_tool(&mut self, args: &Value, cx: &mut Context<Self>) -> Result<oneshot::Receiver<Result<String>>> {
        let index = {
            let viewer = self.viewer.read(cx);
            match entity_key(&args["entity"]) {
                Some(key) => viewer.find_entity(&key).with_context(|| format!("no entity '{key}'"))?,
                None => viewer.active_index().context("the scene is empty")?,
            }
        };
        let mut filter = parse_filters(args, EnsembleFilter::default());
        if let Some(overlap) = args["residue_overlap"].as_f64() {
            filter.residue_overlap = overlap.clamp(0.05, 1.0);
        }
        let defaults = &self.config.job_defaults;
        let stride = args["stride"].as_u64().map(|n| n as usize).unwrap_or(defaults.ensemble_stride);
        let settings = defaults.kvfinder.clone();
        self.start_ensemble_pockets(index, settings, filter, stride, cx)
    }

    pub fn ensemble_summary(&self) -> String {
        match &self.ensemble {
            None => "No ensemble pocket detection has been run in this project.".into(),
            Some(r) => {
                let mut out = format!(
                    "ensemble pockets of {} (job {}, {} frames, {} raw cavities):\n",
                    r.entity, r.job, r.frames, r.total_cavities
                );
                for p in &r.pockets {
                    out.push_str(&format!(
                        "  #{} persistence {:.0}% · max volume {:.0} Å³ · mean area {:.0} Å² · hydropathy {:.2}\n     lining: {}\n",
                        p.rank,
                        p.persistence * 100.0,
                        p.max_volume,
                        p.mean_area,
                        p.mean_hydropathy.unwrap_or(0.0),
                        p.residue_labels(30),
                    ));
                }
                out
            }
        }
    }

    /// Restore the pocket panel from the latest finished ensemble job, then
    /// read its cavities in the background so the filters can be changed.
    pub fn restore_ensemble_results(&mut self, cx: &mut Context<Self>) {
        self.ensemble = self.project.as_ref().and_then(|p| {
            let job = p
                .jobs()
                .ok()?
                .into_iter()
                .find(|j| j.tool == "pykvfinder_ensemble" && j.status == "done")?;
            let result = job.result.clone()?;
            // A job from before filters were stored falls back to the
            // criteria its params recorded, then to the defaults.
            let filter = serde_json::from_value(result["filter"].clone())
                .or_else(|_| serde_json::from_value(job.params["filter"].clone()))
                .unwrap_or_default();
            Some(EnsemblePockets {
                job: job.id,
                entity: result["entity"].as_str()?.to_string(),
                frames: result["frames"].as_u64().unwrap_or(0) as usize,
                total_cavities: result["cavities"].as_u64().unwrap_or(0) as usize,
                pockets: serde_json::from_value(result["pockets"].clone()).ok()?,
                filter,
                found: Vec::new(),
                properties: ResidueProperties::new(),
                stats: None,
            })
        });
        self.set_filter_text(None);
        self.load_ensemble_cavities(cx);
    }

    /// Read a finished job's per-frame results off the UI thread.
    ///
    /// A thousand frames is a thousand small TOML files — about a second,
    /// which is too long to spend on the UI thread at startup.
    fn load_ensemble_cavities(&mut self, cx: &mut Context<Self>) {
        let Some(r) = &self.ensemble else { return };
        if r.can_refilter() {
            return;
        }
        let (job, frames) = (r.job, r.frames);
        let Some(project) = self.project.as_ref() else { return };
        let Ok(Some(record)) = project.job(job) else { return };
        let dir = project.job_path(&record);

        let (tx, rx) = oneshot::channel();
        std::thread::spawn(move || {
            let found = kvfinder::read_ensemble(&dir).map(|(found, _)| found);
            // RMSF over a thousand frames is slow to recompute, so the run
            // wrote it next to the results.
            let properties = std::fs::read_to_string(dir.join(PROPERTIES_FILE))
                .ok()
                .and_then(|text| serde_json::from_str::<ResidueProperties>(&text).ok())
                .unwrap_or_default();
            let _ = tx.send((found.map_err(|e| format!("{e:#}")), properties));
        });

        cx.spawn(async move |this, cx| {
            let Ok((found, properties)) = rx.await else { return };
            this.update(cx, |ws, cx| {
                let Some(r) = ws.ensemble.as_mut() else { return };
                // A different job may have finished while this was reading.
                if r.job != job {
                    return;
                }
                match found {
                    Ok(found) => {
                        r.stats = Some(kvfinder::ensemble_stats(&found, frames, &r.filter, &properties));
                        r.total_cavities = found.len();
                        r.found = found;
                        r.properties = properties;
                    }
                    Err(e) => ws.status = Some(format!("could not read the cavities of job {job}: {e}")),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Jump to the frame where a consensus pocket is largest and select it.
    pub fn focus_ensemble_pocket(&mut self, rank: usize, cx: &mut Context<Self>) {
        let Some(r) = &self.ensemble else { return };
        let Some(p) = r.pockets.iter().find(|p| p.rank == rank).cloned() else { return };
        let entity = r.entity.clone();
        let viewer = self.viewer.read(cx);
        let Some(index) = viewer.find_entity(&entity) else {
            self.status = Some(format!("Add '{entity}' to the scene to inspect its pockets"));
            cx.notify();
            return;
        };
        let s = viewer.entities()[index].structure.clone();
        let core: Vec<(String, String)> = p
            .core_residues(0.5)
            .iter()
            .map(|r| (r.chain.clone(), r.number.clone()))
            .collect();
        let atoms: Vec<usize> = s
            .residues
            .iter()
            .filter(|r| core.iter().any(|(c, n)| *c == r.chain && *n == r.number.to_string()))
            .flat_map(|r| r.atoms.start as usize..r.atoms.end as usize)
            .collect();
        let frame = p.best_frame;
        self.viewer.update(cx, |v, cx| {
            v.set_active(index, cx);
            v.set_frame(index, frame, cx);
            v.set_selection(atoms.clone(), cx);
            v.focus_atoms(&atoms, cx);
        });
        self.status = Some(format!(
            "Pocket #{rank}: {:.0}% persistent, {:.0} Å³ at frame {}",
            p.persistence * 100.0,
            p.max_volume,
            frame + 1
        ));
        cx.notify();
    }

    /// `refilter_ensemble_pockets` tool entry point.
    pub fn refilter_ensemble_tool(&mut self, args: &Value, cx: &mut Context<Self>) -> Result<String> {
        let current = self
            .ensemble
            .as_ref()
            .context("no ensemble pocket run to filter — run detect_pockets_ensemble first")?
            .filter
            .clone();
        let mut filter = parse_filters(args, current);
        if let Some(overlap) = args["residue_overlap"].as_f64() {
            filter.residue_overlap = overlap.clamp(0.05, 1.0);
        }
        let summary = self.refilter_ensemble(filter, cx)?;
        // The panel's editor should show what the assistant just applied.
        self.set_filter_text(None);
        Ok(summary)
    }

    /// Show `text` in the filter editor, or the current filter when `None`.
    pub(crate) fn set_filter_text(&mut self, text: Option<&str>) {
        self.pending_filter_text = Some(match text {
            Some(text) => text.to_string(),
            None => self
                .ensemble
                .as_ref()
                .map(|r| r.filter.to_text())
                .unwrap_or_else(|| EnsembleFilter::default().to_text()),
        });
    }

    /// Re-filter using whatever is in the filter editor.
    fn apply_filter_text(&mut self, cx: &mut Context<Self>) {
        let text = self.pocket_filter.read(cx).value().to_string();
        let (filter, problems) = EnsembleFilter::parse_text(&text);
        if !problems.is_empty() {
            self.status = Some(problems.join("; "));
            cx.notify();
            return;
        }
        if let Err(e) = self.refilter_ensemble(filter, cx) {
            self.status = Some(format!("{e:#}"));
            cx.notify();
        }
    }

    /// The distribution of every metric over the cavities that were found,
    /// as text, so a threshold can be read off the data.
    pub fn cavity_statistics(&self) -> String {
        let Some(r) = &self.ensemble else {
            return "No ensemble pocket detection has been run in this project.".into();
        };
        let Some(stats) = &r.stats else {
            return format!("Still reading the cavities of job {}.", r.job);
        };
        let mut out = format!(
            "{}: {} cavities over {} frames ({} frames held at least one, up to {} in a frame, {:.1} on average).\n\
             Every cavity is kept; the filters below are applied on top and can be changed freely.\n\n\
             metric        n       min    p10    median p90    max    missing\n",
            r.entity, stats.cavities, stats.frames, stats.frames_with_cavities, stats.max_per_frame, stats.mean_per_frame
        );
        for m in &stats.metrics {
            out.push_str(&format!(
                "{:<13} {:<7} {:<6.2} {:<6.2} {:<6.2} {:<6.2} {:<6.2} {}\n",
                m.name, m.n, m.min, m.percentiles[0], m.percentiles[2], m.percentiles[4], m.max, m.missing
            ));
        }
        out.push_str(&format!("\nfilters: {}\n", r.filter.describe()));
        for c in &stats.criteria {
            out.push_str(&format!(
                "  {:<24} keeps {} of {} ({:.0}%)\n",
                c.rule,
                c.passing,
                stats.cavities,
                100.0 * c.passing as f64 / stats.cavities.max(1) as f64
            ));
        }
        out.push_str(&format!(
            "  {:<24} keeps {} of {} ({:.0}%) → {} consensus pockets at {:.0}% residue overlap\n",
            "all together",
            stats.passing_all,
            stats.cavities,
            100.0 * stats.passing_all as f64 / stats.cavities.max(1) as f64,
            r.pockets.len(),
            r.filter.residue_overlap * 100.0
        ));
        out
    }

    pub fn render_ensemble_section(&mut self, cx: &mut Context<Self>) -> Option<gpui::Div> {
        // Only what the panel draws, never the whole result: the cavities
        // and the per-residue properties run to tens of megabytes, and this
        // is on the render path.
        let r = {
            let full = self.ensemble.as_ref()?;
            EnsembleSummary {
                entity: full.entity.clone(),
                frames: full.frames,
                total_cavities: full.total_cavities,
                pocket_count: full.pockets.len(),
                can_refilter: full.can_refilter(),
                stats: full.stats.clone(),
                rows: full
                    .pockets
                    .iter()
                    .take(15)
                    .map(|p| (p.rank, p.persistence, p.max_volume, p.core_residues(0.5).len()))
                    .collect(),
            }
        };
        let mut panel = section("Ensemble pockets").child(
            div().text_xs().text_color(muted()).child(format!(
                "{} · {} frames · {} cavities → {} pockets · click to show the best frame",
                r.entity, r.frames, r.total_cavities, r.pocket_count
            )),
        );

        // What the detector found, so thresholds can be chosen from the data
        // rather than guessed.
        match &r.stats {
            None => {
                panel = panel.child(
                    div()
                        .text_xs()
                        .text_color(muted())
                        .child("reading every cavity, so the filters can be changed…"),
                );
            }
            Some(stats) => {
                let mut table = section_rows("Cavities found").child(stat_row(
                    "per frame",
                    format!(
                        "{:.1} avg · {} max · {} of {} frames",
                        stats.mean_per_frame, stats.max_per_frame, stats.frames_with_cavities, stats.frames
                    ),
                ));
                for m in &stats.metrics {
                    table = table.child(stat_row(
                        m.name.clone(),
                        if m.n == 0 {
                            format!("no values ({} cavities missing it)", m.missing)
                        } else {
                            format!(
                                "median {:.1}  ·  {:.1}–{:.1}  ·  p10 {:.1} p90 {:.1}{}",
                                m.percentiles[2],
                                m.min,
                                m.max,
                                m.percentiles[0],
                                m.percentiles[4],
                                if m.missing == 0 {
                                    String::new()
                                } else {
                                    format!("  ·  {} missing", m.missing)
                                }
                            )
                        },
                    ));
                }
                panel = panel.child(table);

                // Which rule is doing the damage.
                let mut report = section_rows("Filters");
                for c in &stats.criteria {
                    report = report.child(stat_row(
                        c.rule.clone(),
                        format!(
                            "{} ({:.0}%)",
                            c.passing,
                            100.0 * c.passing as f64 / stats.cavities.max(1) as f64
                        ),
                    ));
                }
                report = report.child(stat_row(
                    "all together".to_string(),
                    format!(
                        "{} of {} ({:.0}%) → {} pockets",
                        stats.passing_all,
                        stats.cavities,
                        100.0 * stats.passing_all as f64 / stats.cavities.max(1) as f64,
                        r.pocket_count
                    ),
                ));
                panel = panel.child(report);
            }
        }

        // The filter editor: any metric above, one rule per line.
        panel = panel
            .child(
                div()
                    .text_xs()
                    .text_color(muted())
                    .child("One rule per line: `metric >= n`, `metric <= n`, `metric a .. b`, `overlap 0.6`."),
            )
            .child(Input::new(&self.pocket_filter).small())
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(
                        Button::new("apply-filters")
                            .label("Apply filters")
                            .primary()
                            .small()
                            .disabled(!r.can_refilter)
                            .on_click(cx.listener(|this, _, _, cx| this.apply_filter_text(cx))),
                    )
                    .child(
                        Button::new("filters-none")
                            .label("Show all")
                            .small()
                            .disabled(!r.can_refilter)
                            .on_click(cx.listener(|this, _, _, cx| {
                                // Everything pyKVFinder found, unfiltered.
                                let keep = EnsembleFilter::unfiltered();
                                this.set_filter_text(Some(&keep.to_text()));
                                if let Err(e) = this.refilter_ensemble(keep, cx) {
                                    this.status = Some(format!("{e:#}"));
                                }
                            })),
                    )
                    .child(
                        Button::new("filters-default")
                            .label("Paper defaults")
                            .small()
                            .disabled(!r.can_refilter)
                            .on_click(cx.listener(|this, _, _, cx| {
                                let paper = EnsembleFilter::default();
                                this.set_filter_text(Some(&paper.to_text()));
                                if let Err(e) = this.refilter_ensemble(paper, cx) {
                                    this.status = Some(format!("{e:#}"));
                                }
                            })),
                    ),
            );
        for (rank, persistence, max_volume, residues) in r.rows {
            panel = panel.child(
                div()
                    .id(("pocket", rank))
                    .flex()
                    .justify_between()
                    .px_2()
                    .py_0p5()
                    .rounded_md()
                    .text_xs()
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x2c3039)))
                    .child(
                        div()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            // Persistent pockets stand out.
                            .text_color(rgb(if persistence >= 0.5 { 0x9cd09a } else { 0xc9ccd3 }))
                            .child(format!("#{rank}  {:.0}%", persistence * 100.0)),
                    )
                    .child(format!("{max_volume:.0} Å³ · {residues} res"))
                    .on_click(cx.listener(move |this, _, _, cx| this.focus_ensemble_pocket(rank, cx))),
            );
        }
        Some(panel)
    }

    // --------------------------------------------------------------- FRODOCK

    /// Dock the `ligand` entity onto the `receptor` entity with FRODOCK.
    ///
    /// Both are exported in their current world positions, so the returned
    /// poses are in the same frame as the receptor as it sits in the scene.
    pub fn start_frodock(
        &mut self,
        receptor: usize,
        ligand: usize,
        settings: FrodockSettings,
        cx: &mut Context<Self>,
    ) -> Result<oneshot::Receiver<Result<String>>> {
        anyhow::ensure!(receptor != ligand, "pick two different structures");
        // FRODOCK searches the ligand around the receptor and needs the
        // ligand to be the smaller body; the other way round it still runs
        // for hours and returns scores that cannot be compared. Correct the
        // order rather than waste the run, and say so.
        let (receptor, ligand, swapped) = {
            let viewer = self.viewer.read(cx);
            let entities = viewer.entities();
            let (r, l) = (
                entities.get(receptor).context("no such entity")?.radius(),
                entities.get(ligand).context("no such entity")?.radius(),
            );
            if l > r {
                (ligand, receptor, true)
            } else {
                (receptor, ligand, false)
            }
        };
        // The image is pulled on first use; only if that fails is your own
        // copy of FRODOCK (Settings → Tools) needed to build it.
        let project = self.project.as_ref().context("open or create a project first")?;

        let export = |index: usize| -> Result<(String, String)> {
            let viewer = self.viewer.read(cx);
            let e = viewer.entities().get(index).context("no such entity")?;
            // Water would be treated as part of the interface.
            let dry = |i: usize| e.structure.kind(i) != AtomKind::Water;
            anyhow::ensure!(
                (0..e.structure.atoms.len()).any(dry),
                "{} has no non-water atoms",
                e.name
            );
            Ok((e.name.clone(), molecule::write_pdb(&e.structure, e.transform, dry)))
        };
        let (receptor_name, receptor_pdb) = export(receptor)?;
        let (ligand_name, ligand_pdb) = export(ligand)?;
        if swapped {
            self.dock.receptor = Some(receptor_name.clone());
            self.dock.ligand = Some(ligand_name.clone());
            self.status = Some(format!(
                "{ligand_name} is the smaller body, so it is the ligand and {receptor_name} the receptor"
            ));
        }

        let job = project.create_job(
            "frodock",
            &json!({
                "entity": format!("{ligand_name} → {receptor_name}"),
                "receptor": receptor_name,
                "ligand": ligand_name,
                "settings": settings,
            }),
        )?;
        let dir = project.job_path(&job);
        std::fs::write(dir.join(frodock::RECEPTOR), receptor_pdb)?;
        std::fs::write(dir.join(frodock::LIGAND), ligand_pdb)?;
        if !settings.restraints.is_empty() {
            std::fs::write(
                dir.join(frodock::RESTRAINTS_FILE),
                frodock::restraints_file(&settings.restraints),
            )?;
        }
        let spec = jobs::RunSpec {
            image: jobs::FRODOCK,
            project_root: project.root().to_path_buf(),
            job_dir: job.dir(),
            command: frodock::command(&settings),
            name: format!("openmols-job-{}-{}", job.id, std::process::id()),
        };
        let job_id = job.id;
        let finish: Finish = Box::new(move |ws, cx| ws.finish_frodock(job_id, receptor_name, ligand_name, cx));
        Ok(self.run_job(job.id, spec, finish, cx))
    }

    fn finish_frodock(
        &mut self,
        job: Id,
        receptor: String,
        ligand: String,
        cx: &mut Context<Self>,
    ) -> Result<(String, Value)> {
        let project = self.project.as_ref().context("project closed")?;
        let dir = project.job_path(&project.job(job)?.context("job record missing")?);
        let solutions = frodock::read_solutions(&dir)?;
        let poses = frodock::pose_paths(&dir, &solutions);
        anyhow::ensure!(!poses.is_empty(), "FRODOCK wrote no pose PDBs (see jobs/{job}/log.txt)");
        // One multi-model PDB, so the poses become frames of a single
        // entity. The receptor goes into every model: FRODOCK writes only the
        // body it moved, and a pose on its own does not show the interface.
        let combined = frodock::combine_poses(&poses, Some(&dir.join(frodock::RECEPTOR)))?;
        let entry = project.add_structure_bytes(
            &format!("poses_job{job}.pdb"),
            combined.as_bytes(),
            &format!("{ligand} on {receptor} poses"),
            Some(&format!("frodock:job{job}")),
        )?;
        // A docking run can take hours, so a problem adding the poses to the
        // scene must not throw it away: they are saved in the project either
        // way, and the poses table works without them being loaded.
        let loaded = match self.add_structure_to_scene(entry.id, cx) {
            Ok(index) => Ok(self.viewer.read(cx).entities()[index].name.clone()),
            Err(error) => Err(error),
        };
        let entity = loaded.as_ref().map(String::as_str).unwrap_or(&entry.name).to_string();

        let mut summary = match &loaded {
            Ok(entity) => format!(
                "FRODOCK docked {ligand} onto {receptor}: {} poses, loaded as '{entity}' ({} frames — use the frame slider or set_frame).\n",
                solutions.len(),
                poses.len()
            ),
            Err(error) => format!(
                "FRODOCK docked {ligand} onto {receptor}: {} poses, saved as '{}' in the project, but they could not be added to the scene: {error}\n",
                solutions.len(),
                entry.name
            ),
        };
        for s in solutions.iter().take(10) {
            summary.push_str(&format!(
                "  rank {}: score {:.1}, centre {:.1},{:.1},{:.1}\n",
                s.rank, s.score, s.position[0], s.position[1], s.position[2]
            ));
        }
        let value = json!({
            "solutions": solutions,
            "pose_structure_id": entry.id,
            "pose_entity": entity,
            "receptor": receptor,
            "ligand": ligand,
        });
        self.docking = Some(DockingResults::from_frodock(job, receptor, ligand, entity, &solutions));
        Ok((summary, value))
    }

    /// The entity pair the Dock button will use, resolved by name.
    ///
    /// Roles the panel has not set are assigned automatically: the wider
    /// body is the receptor and the next widest is the ligand. FRODOCK
    /// searches the ligand's orientations around the receptor and assumes
    /// the ligand's sphere is the smaller one, so this is the only ordering
    /// that gives meaningful scores.
    pub fn dock_pair(&self, cx: &gpui::App) -> Option<(usize, usize)> {
        let viewer = self.viewer.read(cx);
        let entities = viewer.entities();
        if entities.len() < 2 {
            return None;
        }
        let by_name = |name: &Option<String>| {
            name.as_ref()
                .and_then(|n| entities.iter().position(|e| e.name == *n))
        };
        let mut widest: Vec<usize> = (0..entities.len()).collect();
        widest.sort_by(|&a, &b| entities[b].radius().total_cmp(&entities[a].radius()));

        let receptor = by_name(&self.dock.receptor).unwrap_or(widest[0]);
        let ligand = by_name(&self.dock.ligand)
            .filter(|l| *l != receptor)
            .or_else(|| widest.iter().copied().find(|&i| i != receptor))?;
        Some((receptor, ligand))
    }

    /// Turn the panel's choices into FRODOCK settings.
    pub(crate) fn dock_settings(&self, receptor: usize, ligand: usize, cx: &Context<Self>) -> FrodockSettings {
        let viewer = self.viewer.read(cx);
        let entities = viewer.entities();
        let around = self.dock.around_ligand.then(|| {
            let c = entities[ligand].world_center();
            [c.x, c.y, c.z]
        });
        // One restraint per selected residue pair: the selection on the
        // receptor is required to end up near the selection on the ligand.
        let restraints = if self.dock.restrain_to_selection {
            let residues = |index: usize| -> Vec<frodock::RestraintResidue> {
                let e = &entities[index];
                let mut seen = Vec::new();
                for &atom in &e.selection {
                    let r = e.structure.residue_of(atom);
                    let entry = frodock::RestraintResidue {
                        name: r.name.clone(),
                        chain: r.chain.clone(),
                        number: r.number as i32,
                    };
                    if !seen.contains(&entry) {
                        seen.push(entry);
                    }
                }
                seen
            };
            let (a, b) = (residues(receptor), residues(ligand));
            a.into_iter()
                .zip(b)
                .map(|(receptor, ligand)| frodock::Restraint {
                    receptor,
                    ligand,
                    distance: 10.0,
                })
                .collect()
        } else {
            Vec::new()
        };
        FrodockSettings {
            top_n: self.dock.top_n,
            cluster_rmsd: self.dock.cluster_rmsd,
            around,
            restraints,
            ..FrodockSettings::default()
        }
    }

    /// `run_frodock` tool entry point.
    pub fn start_frodock_tool(&mut self, args: &Value, cx: &mut Context<Self>) -> Result<oneshot::Receiver<Result<String>>> {
        let find = |key: &Value, what: &str| -> Result<usize> {
            let viewer = self.viewer.read(cx);
            let key = entity_key(key).with_context(|| format!("{what} is required"))?;
            viewer
                .find_entity(&key)
                .with_context(|| format!("no entity '{key}' in the scene"))
        };
        let receptor = find(&args["receptor"], "receptor")?;
        let ligand = find(&args["ligand"], "ligand")?;
        let saved = self.config.job_defaults.clone();
        let defaults = FrodockSettings {
            top_n: saved.dock_top_n,
            cluster_rmsd: saved.dock_cluster_rmsd,
            ..FrodockSettings::default()
        };
        let around = args["around_ligand"].as_bool().unwrap_or(saved.dock_around_ligand).then(|| {
            let c = self.viewer.read(cx).entities()[ligand].world_center();
            [c.x, c.y, c.z]
        });
        let settings = FrodockSettings {
            interaction: match args["interaction"].as_str() {
                Some("antibody") => frodock::Interaction::Antibody,
                Some("enzyme") => frodock::Interaction::Enzyme,
                _ => frodock::Interaction::Other,
            },
            top_n: args["top_n"].as_u64().unwrap_or(defaults.top_n as u64) as usize,
            around,
            ..defaults
        };
        self.start_frodock(receptor, ligand, settings, cx)
    }

    pub fn docking_summary(&self) -> String {
        match &self.docking {
            None => "No docking run has finished in this project.".into(),
            Some(d) => {
                let mut out = format!(
                    "{} job {}: {} onto {}, poses in entity '{}'\n",
                    d.engine, d.job, d.ligand, d.receptor, d.pose_entity
                );
                for s in &d.poses {
                    out.push_str(&format!("  rank {} (frame {}): {}\n", s.rank, s.rank, s.label));
                }
                out
            }
        }
    }

    /// Restore the docking panel from the latest FRODOCK job that produced
    /// solutions.
    ///
    /// A run that searched for hours and then failed while loading its poses
    /// still has everything on disk, so a job whose record holds no solutions
    /// falls back to reading its directory.
    pub fn restore_docking_results(&mut self) {
        self.docking = self.project.as_ref().and_then(|p| {
            let jobs = p.jobs().ok()?;
            jobs.iter()
                .filter(|j| j.tool == "frodock" || j.tool == "autodock")
                .find_map(|job| Self::docking_from_job(p, job))
        });
    }

    fn docking_from_job(project: &project::Project, job: &project::Job) -> Option<DockingResults> {
        let result = job.result.clone().unwrap_or_default();
        let name = |key: &str| {
            result[key]
                .as_str()
                .or_else(|| job.params[key].as_str())
                .unwrap_or("?")
                .to_string()
        };
        let ligand = name("ligand");
        let pose_entity = result["pose_entity"]
            .as_str()
            .map(str::to_string)
            // What `finish_frodock` names the structure it saves.
            .unwrap_or_else(|| format!("{ligand} poses"));
        if job.tool == "autodock" {
            let poses: Vec<jobs::autodock::Pose> = serde_json::from_value(result["poses"].clone()).ok()?;
            return Some(DockingResults {
                job: job.id,
                engine: result["engine"].as_str().unwrap_or("AutoDock").to_string(),
                receptor: name("receptor"),
                pose_entity,
                ligand,
                poses: crate::autodock_ui::pose_rows(&poses),
            });
        }
        let solutions: Vec<Solution> = serde_json::from_value(result["solutions"].clone())
            .ok()
            .or_else(|| frodock::read_solutions(&project.job_path(job)).ok())?;
        Some(DockingResults::from_frodock(job.id, name("receptor"), ligand, pose_entity, &solutions))
    }

    /// Show a docking pose (frame) in the viewer.
    pub fn show_pose(&mut self, rank: usize, cx: &mut Context<Self>) {
        let Some(d) = &self.docking else { return };
        let entity = d.pose_entity.clone();
        let Some(index) = self.viewer.read(cx).find_entity(&entity) else {
            self.status = Some(format!("Add '{entity}' to the scene to see the poses"));
            cx.notify();
            return;
        };
        self.viewer.update(cx, |v, cx| {
            v.set_active(index, cx);
            v.set_frame(index, rank.saturating_sub(1), cx);
        });
        self.status = Some(format!("Showing docking pose {rank}"));
        cx.notify();
    }

    // ---------------------------------------------------------------- panels

    pub fn render_jobs_section(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut panel = section_body();
        // Healthy Docker needs no space here; Settings → Tools has the details.
        if let Some(Err(e)) = &self.docker {
            panel = panel.child(
                div()
                    .text_xs()
                    .text_color(rgb(0xd8a657))
                    .child(format!("Docker unavailable: {e}")),
            );
        }

        let viewer = self.viewer.read(cx);
        let has_entity = viewer.active_index().is_some();
        let has_ensemble = viewer.active().is_some_and(|e| e.structure.frame_count() > 1);
        let entities = viewer.entities().len();
        let has_project = self.project.is_some();
        panel = panel.child(
            div()
                .flex()
                .flex_wrap()
                .gap_1()
                .child(
                    Button::new("detect-cavities")
                        .label("Detect cavities…")
                        .primary()
                        .xsmall()
                        .disabled(!has_entity || !has_project)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.open_job_dialog(crate::job_dialogs::JobKind::Cavities, window, cx)
                        })),
                )
                .child(
                    Button::new("ensemble-pockets")
                        .label("Pockets across ensemble…")
                        .xsmall()
                        .disabled(!has_ensemble || !has_project)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.open_job_dialog(crate::job_dialogs::JobKind::Ensemble, window, cx)
                        })),
                )
                .child(
                    Button::new("dock-autodock")
                        .label("Dock (AutoDock)…")
                        .primary()
                        .xsmall()
                        .disabled(entities < 2 || !has_project)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.open_job_dialog(crate::job_dialogs::JobKind::AutoDock, window, cx)
                        })),
                )
                .child(
                    Button::new("dock")
                        .label("Dock (FRODOCK)…")
                        .xsmall()
                        .disabled(entities < 2 || !has_project)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.open_job_dialog(crate::job_dialogs::JobKind::Dock, window, cx)
                        })),
                ),
        );
        // Say why a button is greyed out, and offer the way out.
        let blockers = self.docking_blockers(cx);
        for line in &blockers {
            panel = panel.child(div().text_xs().text_color(rgb(0xd8a657)).child(line.clone()));
        }
        if !has_entity {
            panel = panel.child(div().text_xs().text_color(muted()).child("Detect cavities needs a structure in the scene."));
        } else if !has_ensemble {
            panel = panel.child(div().text_xs().text_color(muted()).child(
                "Pockets across ensemble needs an entity with several frames (an ensemble or trajectory).",
            ));
        }
        if !blockers.is_empty() {
            panel = panel.child(
                Button::new("jobs-load-example")
                    .label("Load the example (ready to dock)")
                    .xsmall()
                    .on_click(cx.listener(|this, _, _, cx| this.load_example(cx))),
            );
        } else {
            panel = panel.child(
                Button::new("docking-view")
                    .label("Docking view")
                    .xsmall()
                    .on_click(cx.listener(|this, _, _, cx| {
                        if let Err(e) = this.docking_view(cx) {
                            this.status = Some(format!("{e:#}"));
                        }
                        cx.notify();
                    })),
            );
        }

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

    pub(crate) fn pick_frodock_dir(&mut self, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Select your FRODOCK folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = paths.await
                && let Some(dir) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| {
                    let has_tools = dir.join("bin").is_dir() || dir.join("compile_frodock.sh").is_file();
                    this.status = Some(if has_tools {
                        format!("FRODOCK folder: {}", dir.display())
                    } else {
                        format!("{} has no bin/ or compile_frodock.sh — is this the FRODOCK repository?", dir.display())
                    });
                    this.config.frodock_dir = Some(dir);
                    this.config.save();
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    pub fn render_docking_section(&mut self, cx: &mut Context<Self>) -> Option<gpui::Div> {
        let d = self.docking.clone()?;
        let mut panel = section("Docking poses").child(
            div()
                .text_xs()
                .text_color(muted())
                .child(format!("{} · {} onto {} · job #{} · click a pose to show it", d.engine, d.ligand, d.receptor, d.job)),
        );
        for s in d.poses.iter().take(12) {
            let rank = s.rank;
            panel = panel.child(
                div()
                    .id(("pose", rank))
                    .flex()
                    .justify_between()
                    .px_2()
                    .py_0p5()
                    .rounded_md()
                    .text_xs()
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x2c3039)))
                    .child(div().font_weight(gpui::FontWeight::SEMIBOLD).child(format!("#{rank}")))
                    .child(s.label.clone())
                    .on_click(cx.listener(move |this, _, _, cx| this.show_pose(rank, cx))),
            );
        }
        let job = d.job;
        panel = panel.child(
            div()
                .flex()
                .gap_1()
                .child(
                    Button::new("dock-load-poses")
                        .label("Load poses")
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| this.load_job_poses(job, cx))),
                )
                .child(
                    Button::new("dock-open-folder")
                        .label("Open job folder")
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| this.open_job_folder(job, cx))),
                ),
        );
        Some(panel)
    }

    /// Add a finished run's poses to the scene as one multi-frame entity.
    fn load_job_poses(&mut self, job: Id, cx: &mut Context<Self>) {
        let name = format!("poses_job{job}.pdb");
        let found = self.project.as_ref().and_then(|p| {
            p.structures()
                .ok()?
                .into_iter()
                .find(|s| s.file.ends_with(&name))
        });
        match found {
            Some(entry) => self.load_structure_async(entry.id, false, cx),
            None => {
                self.status = Some(format!("job {job} has no saved poses file"));
                cx.notify();
            }
        }
    }

    /// Show a job's folder in the file manager, where its raw output lives.
    fn open_job_folder(&mut self, job: Id, cx: &mut Context<Self>) {
        let path = self
            .project
            .as_ref()
            .and_then(|p| p.job(job).ok().flatten().map(|j| p.job_path(&j)));
        let Some(path) = path else {
            self.status = Some(format!("job {job} has no folder"));
            cx.notify();
            return;
        };
        let opener = if cfg!(target_os = "windows") {
            "explorer"
        } else if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        // `explorer` reports failure even when it opens the window, so the
        // status line reflects the path rather than the exit code.
        let _ = std::process::Command::new(opener).arg(&path).spawn();
        self.status = Some(format!("{}", path.display()));
        cx.notify();
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

impl Workspace {
    /// Report a failed background run in the status line.
    pub(crate) fn report_start<T: 'static>(
        &mut self,
        started: Result<oneshot::Receiver<Result<T>>>,
        cx: &mut Context<Self>,
    ) {
        match started {
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
            Err(e) => self.status = Some(format!("{e:#}")),
        }
        cx.notify();
    }

    /// Start cavity detection on the active entity with the saved settings.
    pub(crate) fn run_cavities(&mut self, cx: &mut Context<Self>) {
        let Some(index) = self.viewer.read(cx).active_index() else { return };
        let settings = self.config.job_defaults.kvfinder.clone();
        let started = self.start_kvfinder(index, settings, cx);
        self.report_start(started, cx);
    }

    /// Start the ensemble pocket run on the active entity with the saved settings.
    pub(crate) fn run_ensemble(&mut self, cx: &mut Context<Self>) {
        let Some(index) = self.viewer.read(cx).active_index() else { return };
        let defaults = &self.config.job_defaults;
        let (settings, stride) = (defaults.kvfinder.clone(), defaults.ensemble_stride);
        let started = self.start_ensemble_pockets(index, settings, EnsembleFilter::default(), stride, cx);
        self.report_start(started, cx);
    }

    /// Dock with the roles and settings in `self.dock`.
    pub(crate) fn run_dock(&mut self, cx: &mut Context<Self>) {
        let Some((receptor, ligand)) = self.dock_pair(cx) else {
            self.status = Some("add two structures to the scene first".into());
            cx.notify();
            return;
        };
        let settings = self.dock_settings(receptor, ligand, cx);
        let started = self.start_frodock(receptor, ligand, settings, cx);
        self.report_start(started, cx);
    }
}
