//! AutoDock Vina / AutoDock4 jobs: exporting the two bodies and the search
//! box, starting the container, and loading the poses as a multi-frame entity.

use crate::jobs_ui::{DockingResults, Finish, PoseRow};
use crate::workspace::Workspace;
use anyhow::{Context as _, Result};
use futures::channel::oneshot;
use gpui::Context;
use jobs::autodock::{self, Covalent, Engine, Settings, Warhead};
use molecule::AtomKind;
use serde_json::{Value, json};

/// Padding around the ligand when the box is made automatically, Å.
const BOX_PADDING: f32 = 8.0;
/// Smallest automatic box edge, Å.
const BOX_MIN: f32 = 20.0;

/// Table rows for poses: energies in kcal/mol, or "covalent pose" when the engine gives none.
pub fn pose_rows(poses: &[autodock::Pose]) -> Vec<PoseRow> {
    poses
        .iter()
        .map(|p| PoseRow {
            rank: p.rank,
            label: match p.energy {
                Some(e) => format!("{e:.2} kcal/mol"),
                None => "covalent pose".to_string(),
            },
        })
        .collect()
}

/// Display name of a run, e.g. `AutoDock Vina (covalent)`.
pub fn engine_label(settings: &Settings) -> String {
    match &settings.covalent {
        Some(_) => format!("{} (covalent)", settings.engine.label()),
        None => settings.engine.label().to_string(),
    }
}

impl Workspace {
    /// A residue to suggest for covalent docking: the one Cys, Lys, Ser or Tyr selected on the receptor, as `A:CYS:481`.
    pub(crate) fn suggested_residue(&self, cx: &gpui::App) -> String {
        let fallback = "A:CYS:".to_string();
        let Some((receptor, _)) = self.dock_pair(cx) else { return fallback };
        let viewer = self.viewer.read(cx);
        let e = &viewer.entities()[receptor];
        let mut residues: Vec<usize> = e.selection.iter().map(|&a| e.structure.atoms[a].residue as usize).collect();
        residues.sort_unstable();
        residues.dedup();
        match residues[..] {
            [r] if autodock::RESIDUES.contains(&e.structure.residues[r].name.to_ascii_uppercase().as_str()) => {
                let r = &e.structure.residues[r];
                format!("{}:{}:{}", r.chain, r.name.to_ascii_uppercase(), r.number)
            }
            _ => fallback,
        }
    }

    /// The search box: the one in the scene, else around the ligand (and, for
    /// covalent runs, the target cysteine's sulfur).
    pub(crate) fn autodock_box(
        &self,
        receptor: usize,
        ligand: usize,
        covalent: Option<&Covalent>,
        cx: &gpui::App,
    ) -> ([f32; 3], [f32; 3]) {
        let viewer = self.viewer.read(cx);
        if let Some(found) = viewer.dock_box() {
            return found;
        }
        let entities = viewer.entities();
        let l = &entities[ligand];
        let mut points: Vec<glam::Vec3> = (0..l.structure.atoms.len())
            .filter(|&i| l.structure.kind(i) != AtomKind::Water)
            .map(|i| l.transform.transform_point3(l.structure.atoms[i].position))
            .collect();
        if let Some((chain, number)) = covalent.and_then(|c| c.parse().ok()) {
            let r = &entities[receptor];
            let sulfur = r
                .structure
                .residues
                .iter()
                .position(|x| x.number == number as isize && x.chain.eq_ignore_ascii_case(&chain))
                .and_then(|res| r.structure.residue_atom(res, "SG"));
            if let Some(a) = sulfur {
                points.push(r.transform.transform_point3(r.structure.atoms[a].position));
            }
        }
        match crate::annotations::box_around(&points, BOX_PADDING) {
            Some(crate::annotations::Annotation::DockBox { center, size }) => {
                (center, size.map(|s| s.max(BOX_MIN)))
            }
            _ => ([0.0; 3], [BOX_MIN; 3]),
        }
    }

    /// Start a docking run on the pair chosen in the dock setup.
    pub(crate) fn run_autodock(&mut self, settings: Settings, cx: &mut Context<Self>) {
        let Some((receptor, ligand)) = self.dock_pair(cx) else {
            self.status = Some("add a receptor and a ligand to the scene first".into());
            cx.notify();
            return;
        };
        let started = self.start_autodock(receptor, ligand, settings, cx);
        self.report_start(started, cx);
    }

    /// Dock `ligand` into `receptor` with AutoDock Vina or AutoDock4.
    ///
    /// Both bodies are exported at their current positions, so the poses
    /// come back in the frame the receptor has in the scene.
    pub fn start_autodock(
        &mut self,
        receptor: usize,
        ligand: usize,
        mut settings: Settings,
        cx: &mut Context<Self>,
    ) -> Result<oneshot::Receiver<Result<String>>> {
        anyhow::ensure!(receptor != ligand, "pick two different structures");
        let project = self.project.as_ref().context("open or create a project first")?;
        if let Some(c) = &settings.covalent {
            c.parse()?;
        }
        let export = |index: usize, keep: &dyn Fn(AtomKind) -> bool| -> Result<(String, String)> {
            let viewer = self.viewer.read(cx);
            let e = viewer.entities().get(index).context("no such entity")?;
            let want = |i: usize| keep(e.structure.kind(i));
            anyhow::ensure!((0..e.structure.atoms.len()).any(want), "{} has nothing to export", e.name);
            Ok((e.name.clone(), molecule::write_pdb(&e.structure, e.transform, want)))
        };
        // The receptor is the polymer: other molecules (waters, a bound
        // ligand, unusual hets) would not survive Meeko's templates anyway.
        let (receptor_name, receptor_pdb) = export(receptor, &|k| k == AtomKind::Polymer)?;
        let (ligand_name, ligand_pdb) = export(ligand, &|k| k != AtomKind::Water)?;

        let (center, size) = self.autodock_box(receptor, ligand, settings.covalent.as_ref(), cx);
        settings.center = center;
        settings.size = size;
        let command = autodock::command(&settings)?;
        let engine = engine_label(&settings);

        let job = project.create_job(
            "autodock",
            &json!({
                "entity": format!("{ligand_name} → {receptor_name}"),
                "receptor": receptor_name,
                "ligand": ligand_name,
                "engine": engine,
                "settings": settings,
            }),
        )?;
        let dir = project.job_path(&job);
        std::fs::write(dir.join(autodock::RECEPTOR), receptor_pdb)?;
        std::fs::write(dir.join(autodock::LIGAND), ligand_pdb)?;
        std::fs::write(dir.join(autodock::COVALENT_PREP_FILE), autodock::COVALENT_PREP)?;
        let spec = jobs::RunSpec {
            image: jobs::AUTODOCK,
            project_root: project.root().to_path_buf(),
            job_dir: job.dir(),
            command,
            name: format!("openmols-job-{}-{}", job.id, std::process::id()),
        };
        let job_id = job.id;
        let finish: Finish =
            Box::new(move |ws, cx| ws.finish_autodock(job_id, receptor_name, ligand_name, settings, cx));
        Ok(self.run_job(job.id, spec, finish, cx))
    }

    fn finish_autodock(
        &mut self,
        job: project::Id,
        receptor: String,
        ligand: String,
        settings: Settings,
        cx: &mut Context<Self>,
    ) -> Result<(String, Value)> {
        let project = self.project.as_ref().context("project closed")?;
        let dir = project.job_path(&project.job(job)?.context("job record missing")?);
        let covalent = settings.covalent.is_some();
        let poses = match settings.engine {
            Engine::Vina => {
                let text = std::fs::read_to_string(dir.join(autodock::VINA_OUT)).context("reading Vina's output")?;
                // Vina leaves a flexible residue's energy out of its figure.
                autodock::parse_vina(&text, !covalent)?
            }
            Engine::AutoDock4 => {
                let text = std::fs::read_to_string(dir.join(autodock::AD4_OUT)).context("reading AutoDock4's log")?;
                autodock::parse_dlg(&text, covalent)?
            }
        };
        let engine = engine_label(&settings);
        // Saved in the project first: a failure adding them to the scene must not lose the run.
        let entry = project.add_structure_bytes(
            &format!("poses_job{job}.pdb"),
            poses.pdb.as_bytes(),
            &format!("{ligand} poses ({engine})"),
            Some(&format!("autodock:job{job}")),
        )?;
        let loaded = self
            .add_structure_to_scene(entry.id, cx)
            .map(|index| self.viewer.read(cx).entities()[index].name.clone());
        let entity = loaded.as_ref().map(String::as_str).unwrap_or(&entry.name).to_string();

        let mut summary = match &loaded {
            Ok(name) => format!(
                "{engine} docked {ligand} into {receptor}: {} poses, loaded as '{name}' (frames, best first).\n",
                poses.poses.len()
            ),
            Err(e) => format!(
                "{engine} docked {ligand} into {receptor}: {} poses, saved as '{}' but not added to the scene: {e}\n",
                poses.poses.len(),
                entry.name
            ),
        };
        for p in poses.poses.iter().take(10) {
            match p.energy {
                Some(e) => summary.push_str(&format!("  rank {}: {e:.2} kcal/mol\n", p.rank)),
                None => summary.push_str(&format!("  rank {}\n", p.rank)),
            }
        }
        if covalent {
            summary.push_str(
                "Covalent poses are ranked by the search but carry no binding energy: the engines leave the bonded residue out of their score.\n",
            );
        }
        let value = json!({
            "poses": poses.poses,
            "engine": engine,
            "pose_structure_id": entry.id,
            "pose_entity": entity,
            "receptor": receptor,
            "ligand": ligand,
        });
        self.docking = Some(DockingResults {
            job,
            engine,
            receptor,
            ligand,
            pose_entity: entity,
            poses: pose_rows(&poses.poses),
        });
        Ok((summary, value))
    }

    /// `run_autodock` tool.
    pub fn start_autodock_tool(&mut self, args: &Value, cx: &mut Context<Self>) -> Result<oneshot::Receiver<Result<String>>> {
        let find = |key: &Value, what: &str| -> Result<usize> {
            let key = key.as_str().with_context(|| format!("{what} is required"))?;
            self.viewer
                .read(cx)
                .find_entity(key)
                .with_context(|| format!("no entity '{key}' in the scene"))
        };
        let (receptor, ligand) = (find(&args["receptor"], "receptor")?, find(&args["ligand"], "ligand")?);
        let saved = &self.config.job_defaults;
        let covalent = args["covalent_residue"].as_str().map(|residue| Covalent {
            residue: residue.to_string(),
            warhead: Warhead::ALL
                .into_iter()
                .find(|w| Some(w.key()) == args["warhead"].as_str())
                .unwrap_or(saved.ad_warhead),
        });
        let settings = Settings {
            engine: match args["engine"].as_str() {
                Some("autodock4") => Engine::AutoDock4,
                Some("vina") => Engine::Vina,
                _ => saved.ad_engine,
            },
            exhaustiveness: args["exhaustiveness"].as_u64().map(|n| n as u32).unwrap_or(saved.ad_exhaustiveness),
            num_modes: args["num_modes"].as_u64().map(|n| n as u32).unwrap_or(saved.ad_modes),
            covalent,
            ..Settings::default()
        };
        self.start_autodock(receptor, ligand, settings, cx)
    }
}
