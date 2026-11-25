//! Getting to a docking-ready scene: a bundled example, splitting a ligand off
//! a complex, and a representation that makes a receptor and ligand legible.

use crate::config::LeftTab;
use crate::workspace::Workspace;
use anyhow::{Context as _, Result};
use glam::Mat4;
use gpui::Context;
use molecule::{AtomKind, Structure};
use project::Project;
use render::{Coloring, Component, Representation, Target};

/// HIV-1 protease with the inhibitor indinavir (MK1), PDB 1HSG.
const EXAMPLE_PDB: &[u8] = include_bytes!("../../../assets/1hsg.pdb");
/// Residues of the receptor within this distance of the ligand are drawn as sticks.
const POCKET_RADIUS: f32 = 5.0;
/// Edge padding of the docking box made for the example, Å.
const EXAMPLE_BOX_PADDING: f32 = 8.0;

/// The biggest non-water hetero group: the ligand of a complex.
fn largest_ligand(structure: &Structure) -> Option<usize> {
    structure
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| r.atoms.start < r.atoms.end && structure.kind(r.atoms.start as usize) == AtomKind::Ligand)
        .filter(|(_, r)| r.atoms.len() >= 4)
        .max_by_key(|(_, r)| r.atoms.len())
        .map(|(i, _)| i)
}

/// `(everything but the ligand, the ligand)` as PDB text in world coordinates.
fn split_pdb(structure: &Structure, transform: Mat4, ligand: usize) -> (String, String) {
    let in_ligand = |i: usize| structure.atoms[i].residue as usize == ligand;
    (
        molecule::write_pdb(structure, transform, |i| !in_ligand(i)),
        molecule::write_pdb(structure, transform, in_ligand),
    )
}

impl Workspace {
    /// Why the docking buttons cannot be used yet; empty when they can.
    pub(crate) fn docking_blockers(&self, cx: &gpui::App) -> Vec<String> {
        let mut why = Vec::new();
        if self.project.is_none() {
            why.push("Open or create a project first (Project tab).".to_string());
        }
        if self.viewer.read(cx).entities().len() < 2 {
            why.push(
                "Docking needs two entities in the scene: the receptor (protein) and the ligand (small molecule). \
                 A structure that holds both, like 1HSG, can be split in the Scene tab."
                    .to_string(),
            );
        }
        why
    }

    /// Load the bundled example: protease and inhibitor as two entities, a
    /// search box around the ligand, ready for **Dock (AutoDock)**.
    pub fn load_example(&mut self, cx: &mut Context<Self>) {
        self.status = Some(match self.try_load_example(cx) {
            Ok(message) => message,
            Err(e) => format!("Could not load the example: {e:#}"),
        });
        cx.notify();
    }

    fn try_load_example(&mut self, cx: &mut Context<Self>) -> Result<String> {
        if self.project.is_none() {
            let base = self.config.projects_dir.join("example-hiv-protease");
            let (mut root, mut n) = (base.clone(), 2);
            while root.exists() {
                root = self.config.projects_dir.join(format!("example-hiv-protease-{n}"));
                n += 1;
            }
            let project = Project::create(&root, "HIV protease example")?;
            self.set_project(project, cx);
        }
        let project = self.project.as_ref().context("no project")?;
        let full = project.add_structure_bytes("1hsg.pdb", EXAMPLE_PDB, "1HSG", Some("example:1hsg"))?;
        let structure = self.read_structure(full.id)?;
        let ligand = largest_ligand(&structure).context("the example has no ligand")?;
        let (protein_pdb, ligand_pdb) = split_pdb(&structure, Mat4::IDENTITY, ligand);
        let project = self.project.as_ref().context("no project")?;
        let protein = project.add_structure_bytes("1hsg_protease.pdb", protein_pdb.as_bytes(), "HIV-1 protease", Some("example:1hsg"))?;
        let inhibitor = project.add_structure_bytes("1hsg_mk1.pdb", ligand_pdb.as_bytes(), "Indinavir (MK1)", Some("example:1hsg"))?;

        self.viewer.update(cx, |v, cx| v.clear(cx));
        let receptor = self.add_structure_to_scene(protein.id, cx)?;
        let ligand_index = self.add_structure_to_scene(inhibitor.id, cx)?;
        let names = {
            let v = self.viewer.read(cx);
            (v.entities()[receptor].name.clone(), v.entities()[ligand_index].name.clone())
        };
        self.dock.receptor = Some(names.0.clone());
        self.dock.ligand = Some(names.1.clone());
        self.docking_view(cx)?;
        self.box_around_ligand(cx);
        self.config.left_tab = LeftTab::Jobs;
        self.config.save();
        Ok(format!(
            "Example ready: {} (receptor) and {} (ligand), with a search box around the ligand. \
             Now click Dock (AutoDock)… and Run.",
            names.0, names.1
        ))
    }

    /// Put the search box around the dock pair's ligand.
    fn box_around_ligand(&mut self, cx: &mut Context<Self>) {
        let Some((_, ligand)) = self.dock_pair(cx) else { return };
        let points: Vec<glam::Vec3> = {
            let e = &self.viewer.read(cx).entities()[ligand];
            e.structure.atoms.iter().map(|a| e.transform.transform_point3(a.position)).collect()
        };
        if let Some(crate::annotations::Annotation::DockBox { center, size }) =
            crate::annotations::box_around(&points, EXAMPLE_BOX_PADDING)
        {
            self.viewer.update(cx, |v, cx| v.set_dock_box(center, size.map(|s| s.max(20.0)), cx));
        }
    }

    /// Receptor as cartoon with its pocket residues as sticks, ligand as ball and stick.
    pub fn docking_view(&mut self, cx: &mut Context<Self>) -> Result<()> {
        let (receptor, ligand) = self.dock_pair(cx).context("the scene needs a receptor and a ligand")?;
        let pocket: Vec<usize> = {
            let viewer = self.viewer.read(cx);
            let (r, l) = (&viewer.entities()[receptor], &viewer.entities()[ligand]);
            let lig: Vec<glam::Vec3> = l.structure.atoms.iter().map(|a| l.transform.transform_point3(a.position)).collect();
            let cutoff = POCKET_RADIUS * POCKET_RADIUS;
            let mut atoms = Vec::new();
            for res in &r.structure.residues {
                if res.atoms.start == res.atoms.end || res.is_water() {
                    continue;
                }
                let near = (res.atoms.start..res.atoms.end).any(|a| {
                    let p = r.transform.transform_point3(r.structure.atoms[a as usize].position);
                    lig.iter().any(|q| q.distance_squared(p) <= cutoff)
                });
                if near {
                    atoms.extend(res.atoms.start as usize..res.atoms.end as usize);
                }
            }
            atoms
        };
        self.viewer.update(cx, |v, cx| {
            v.set_active(receptor, cx);
            v.set_components(vec![Component::new(Target::Polymer, Representation::Cartoon, Coloring::Chain)], cx);
            v.set_selection(pocket, cx);
            v.set_active(ligand, cx);
            v.set_components(vec![Component::new(Target::All, Representation::BallAndStick, Coloring::Element)], cx);
            v.reset_view(cx);
        });
        Ok(())
    }

    /// Replace a complex by its protein-side entity and a separate ligand entity.
    pub fn split_ligand(&mut self, index: usize, cx: &mut Context<Self>) {
        self.status = Some(match self.try_split_ligand(index, cx) {
            Ok(message) => message,
            Err(e) => format!("Could not split: {e:#}"),
        });
        cx.notify();
    }

    fn try_split_ligand(&mut self, index: usize, cx: &mut Context<Self>) -> Result<String> {
        let (structure, transform, name) = {
            let v = self.viewer.read(cx);
            let e = v.entities().get(index).context("no such entity")?;
            (e.structure.clone(), e.transform, e.name.clone())
        };
        let ligand = largest_ligand(&structure).context("this entity has no ligand to split off")?;
        anyhow::ensure!(
            (0..structure.atoms.len()).any(|i| structure.atoms[i].residue as usize != ligand),
            "there is nothing but the ligand here"
        );
        let ligand_name = structure.residues[ligand].name.clone();
        let (rest_pdb, ligand_pdb) = split_pdb(&structure, transform, ligand);
        let project = self.project.as_ref().context("open or create a project first")?;
        let rest = project.add_structure_bytes(
            &format!("{}_without_{}.pdb", name.replace(' ', "_"), ligand_name),
            rest_pdb.as_bytes(),
            &format!("{name} (no {ligand_name})"),
            Some("split"),
        )?;
        let lig = project.add_structure_bytes(
            &format!("{}_{}.pdb", name.replace(' ', "_"), ligand_name),
            ligand_pdb.as_bytes(),
            &format!("{ligand_name} (ligand)"),
            Some("split"),
        )?;
        self.viewer.update(cx, |v, cx| v.remove_entity(index, cx));
        self.add_structure_to_scene(rest.id, cx)?;
        self.add_structure_to_scene(lig.id, cx)?;
        self.dock.receptor = None;
        self.dock.ligand = None;
        if self.dock_pair(cx).is_some() {
            self.docking_view(cx)?;
        }
        Ok(format!(
            "Split {ligand_name} off {name}: it is now its own entity, so it can be docked or measured against the protein."
        ))
    }
}
