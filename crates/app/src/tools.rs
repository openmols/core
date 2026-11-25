//! Tools the LLM can call to drive the viewer and the project.
//!
//! Each tool takes JSON arguments and returns a short text result that is fed
//! back to the model. Network tools (`fetch_pdb`) are split into a background
//! download step and a UI-thread apply step, see [`crate::chat`].

use crate::viewer::ViewState;
use crate::workspace::Workspace;
use anyhow::{Context as _, Result, anyhow, bail};
use gpui::Context;
use llm::ToolSpec;
use project::Id;
use molecule::{AtomKind, SecondaryStructure, Structure};
use render::{Coloring, Component, Representation, Target};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const SYSTEM_PROMPT: &str = "You are the assistant inside openmols, a desktop drug-discovery workbench. \
The user works in projects (one per discovery campaign) containing protein/ligand structures, saved views and notes. \
You can drive the 3D viewer and the project with the provided tools. Prefer calling tools over describing what the user \
should click. Call get_project_overview or get_structure_info before answering questions about the current data. \
Representations available: cartoon, ball_and_stick, spacefill, licorice, backbone, gaussian_surface. \
Targets: all, polymer, ligand, water, selection. Colorings: element, chain, secondary. \
The scene can hold several entities (structures placed with rigid transforms); most tools act on the active entity \
and accept an optional `entity` argument (name or index) to target another one. Use align_entities to superpose \
structures, move_entity to pose a partner (e.g. as a docking starting position) and export_pose to save posed coordinates. \
Keep answers concise and scientific; cite residue names as CHAIN:RESNAME NUMBER.";

fn spec(name: &str, description: &str, parameters: Value) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: description.into(),
        parameters,
    }
}

fn no_args() -> Value {
    json!({ "type": "object", "properties": {} })
}

pub fn specs() -> Vec<ToolSpec> {
    let mut specs = base_specs();
    // Every tool that acts on structures can target a specific entity.
    const SCENE_LEVEL: &[&str] = &[
        "get_project_overview", "fetch_pdb", "load_structure", "add_structure_to_scene",
        "list_entities", "align_entities", "add_note", "save_view", "load_view", "reset_view", "clear_scene",
        "run_frodock", "run_autodock", "list_docking_poses", "list_ensemble_pockets",
        "cavity_statistics", "refilter_ensemble_pockets",
    ];
    for spec in &mut specs {
        if SCENE_LEVEL.contains(&spec.name.as_str()) {
            continue;
        }
        if let Some(props) = spec.parameters.get_mut("properties").and_then(Value::as_object_mut) {
            props.insert(
                "entity".into(),
                json!({ "type": "string", "description": "Entity name or index; defaults to the active entity" }),
            );
        }
    }
    specs
}

fn base_specs() -> Vec<ToolSpec> {
    let component = json!({
        "type": "object",
        "properties": {
            "target": { "type": "string", "enum": ["all", "polymer", "ligand", "water", "selection"] },
            "representation": { "type": "string", "enum": ["cartoon", "ball_and_stick", "spacefill", "licorice", "backbone", "gaussian_surface"] },
            "coloring": { "type": "string", "enum": ["element", "chain", "secondary", "residue"] }
        },
        "required": ["target", "representation"]
    });
    vec![
        spec("get_project_overview", "List the open project's structures, saved views and notes, and which structure is loaded.", no_args()),
        spec("get_structure_info", "Summarise the loaded structure: chains, residue counts, ligands, secondary structure, current representations and selection.", no_args()),
        spec("fetch_pdb", "Download a structure from the RCSB PDB by 4-character id, add it to the project and load it. PDB format is used where available, falling back to mmCIF only for entries too large for it.",
            json!({ "type": "object", "properties": { "pdb_id": { "type": "string" } }, "required": ["pdb_id"] })),
        spec("load_structure", "Replace the whole scene with one project structure.",
            json!({ "type": "object", "properties": { "structure_id": { "type": "integer" } }, "required": ["structure_id"] })),
        spec("add_structure_to_scene", "Add a project structure to the scene as a new entity (keeps the others) and make it active.",
            json!({ "type": "object", "properties": { "structure_id": { "type": "integer" } }, "required": ["structure_id"] })),
        spec("list_entities", "List the entities in the scene with their index, structure, frame count and whether they are moved.", no_args()),
        spec("remove_entity", "Remove one entity from the scene (the active one unless `entity` says otherwise).", no_args()),
        spec("clear_scene", "Remove every entity from the scene, leaving it empty. The project's structures are untouched.", no_args()),
        spec("align_entities", "Superpose the mobile entity onto the target (sequence alignment + Cα Kabsch fit with outlier rejection, like PyMOL align). Returns RMSD.",
            json!({ "type": "object", "properties": {
                "mobile": { "type": "string", "description": "Entity name or index to move" },
                "target": { "type": "string", "description": "Entity name or index to keep fixed" },
                "pairing": { "type": "string", "enum": ["sequence", "residue_number"], "description": "How residues are paired (default sequence)" },
                "mobile_chain": { "type": "string" },
                "target_chain": { "type": "string" }
            }, "required": ["mobile", "target"] })),
        spec("move_entity", "Rigidly move an entity: translate by [x,y,z] Å and/or rotate by [x,y,z] degrees about world axes through its centre.",
            json!({ "type": "object", "properties": {
                "translate": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                "rotate": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }
            } })),
        spec("export_pose", "Write an entity's current (moved/aligned) coordinates into the project as a new PDB structure.", no_args()),
        spec("set_frame", "Show a given frame (1-based) of an ensemble, trajectory or docking-pose entity.",
            json!({ "type": "object", "properties": { "frame": { "type": "integer" } }, "required": ["frame"] })),
        spec("fit_frames", "Superpose all frames of a multi-frame entity onto its first frame (Cα). Returns the mean RMSD.", no_args()),
        spec("set_representations", "Replace all representations with the given list (applied in order).",
            json!({ "type": "object", "properties": { "components": { "type": "array", "items": component } }, "required": ["components"] })),
        spec("add_representation", "Add one representation on top of the current ones.", component),
        spec("show_water", "Show or hide water molecules.",
            json!({ "type": "object", "properties": { "visible": { "type": "boolean" } }, "required": ["visible"] })),
        spec("select_residues", "Select residues (shown as ball & stick) and focus the camera on them.",
            json!({ "type": "object", "properties": {
                "chain": { "type": "string", "description": "Chain id; omit for all chains" },
                "residues": { "type": "array", "items": { "type": "string" }, "description": "Residue numbers or ranges, e.g. [\"25\", \"80-84\"]" }
            }, "required": ["residues"] })),
        spec("select_binding_site", "Select a ligand and all polymer residues within `cutoff` Å of it, focus on them, and return the residue list.",
            json!({ "type": "object", "properties": {
                "ligand": { "type": "string", "description": "Ligand residue name, e.g. MK1. Omit to use the first ligand." },
                "cutoff": { "type": "number", "description": "Distance in Å (default 5)" }
            } })),
        spec("clear_selection", "Clear the selection.", no_args()),
        spec("focus", "Point the camera at the selection, a ligand, or the whole structure.",
            json!({ "type": "object", "properties": { "target": { "type": "string", "description": "'selection', 'all', or a ligand residue name" } }, "required": ["target"] })),
        spec("reset_view", "Reset the camera to show the whole structure.", no_args()),
        spec("save_view", "Save the current camera, representations and selection as a named view in the project.",
            json!({ "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] })),
        spec("load_view", "Restore a saved view by name.",
            json!({ "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] })),
        spec("detect_cavities", "Detect and characterise cavities/pockets of an entity with pyKVFinder (runs in Docker; takes seconds). Shows the cavities as surfaces and returns volume, area, depth, hydropathy and lining residues per cavity.",
            json!({ "type": "object", "properties": {
                "ligand_adjust": { "type": "boolean", "description": "Only report cavity space around the entity's bound ligand" },
                "probe_out": { "type": "number", "description": "Probe-out radius in Å (default 4; larger finds shallower pockets)" },
                "volume_cutoff": { "type": "number", "description": "Minimum cavity volume in Å³ (default 5)" }
            } })),
        spec("list_cavities", "Return the results of the latest cavity detection in this project.", no_args()),
        spec("measure", "Measure and show a distance (2 atoms), angle (3, vertex in the middle) or dihedral (4) between named atoms, possibly in different entities. Returns the value in angstrom or degrees.",
            json!({ "type": "object", "properties": {
                "kind": { "type": "string", "enum": ["distance", "angle", "dihedral"] },
                "atoms": { "type": "array", "items": { "type": "object", "properties": { "entity": { "type": "string", "description": "Entity name; omit for the active one" }, "chain": { "type": "string" }, "residue": { "type": "integer" }, "atom": { "type": "string", "description": "Atom name, e.g. CA or OD2" } }, "required": ["residue", "atom"] } }
            }, "required": ["kind", "atoms"] })),
        spec("find_contacts", "Find and draw contacts between two entities (e.g. a ligand and its receptor): H-bond candidates (N/O within 3.5 A), other contacts and clashes. Returns counts and the polar/clash pairs by residue.",
            json!({ "type": "object", "properties": {
                "entity": { "type": "string" }, "against": { "type": "string" },
                "cutoff": { "type": "number", "description": "Distance in A, default 4" },
                "use_selection": { "type": "boolean", "description": "Only the selected atoms of `entity`" }
            }, "required": ["entity", "against"] })),
        spec("set_dock_box", "Place the docking search box (the grid for Vina/AutoDock). Give center and size, or let it wrap the selection / ligand of an entity with padding.",
            json!({ "type": "object", "properties": {
                "center": { "type": "array", "items": { "type": "number" } }, "size": { "type": "array", "items": { "type": "number" } },
                "entity": { "type": "string" }, "around": { "type": "string", "enum": ["selection", "ligand"] },
                "padding": { "type": "number", "description": "Angstrom added on each side (default 5)" }
            } })),
        spec("list_annotations", "List the measurements, contact sets and boxes in the scene.", no_args()),
        spec("clear_annotations", "Remove all measurements, contact sets and boxes.", no_args()),
        spec("detect_pockets_ensemble", "Detect pockets across EVERY frame of an ensemble or trajectory with pyKVFinder, then merge cavities that share lining residues into consensus pockets scored by persistence (the fraction of frames they appear in). Use this instead of detect_cavities when the entity has more than one frame. Runs in Docker; roughly a second per frame. Defaults reproduce the AR-V7 protocol: >5 lining residues, hydropathy <= 0, area 80-2480 A^2, volume >= 120 A^3, merged at 60% shared residues.",
            json!({ "type": "object", "properties": {
                "stride": { "type": "integer", "description": "Analyse every Nth frame (default 1 = all)" },
                "residue_overlap": { "type": "number", "description": "Fraction of lining residues two cavities must share to count as the same pocket (default 0.6)" },
                "replace_defaults": { "type": "boolean", "description": "Start from no filters instead of the AR-V7 defaults" },
                "filters": {
                    "type": "array",
                    "description": "Filters to add, change or remove. Each sets a min and/or max on one metric; giving a metric that already has a filter replaces it.",
                    "items": { "type": "object", "properties": {
                        "metric": { "type": "string", "description": "volume, area, max_depth, avg_depth, hydropathy, residues, persistence, or a per-residue property averaged over the lining residues: rmsf (from the ensemble's own motion, fitted first) or bfactor" },
                        "min": { "type": "number" },
                        "max": { "type": "number" },
                        "remove": { "type": "boolean", "description": "Drop the default filter on this metric" }
                    }, "required": ["metric"] }
                }
            } })),
        spec("list_ensemble_pockets", "Return the consensus pockets from the latest ensemble pocket run, with persistence and lining residues.", no_args()),
        spec("cavity_statistics", "Statistics over EVERY cavity pyKVFinder found in the latest ensemble run, before filtering: how many per frame, and the range, median and percentiles of each metric, plus how many cavities each filter rule keeps on its own. Use this to choose thresholds from the data, or to explain why a filter kept too few pockets.", no_args()),
        spec("refilter_ensemble_pockets", "Apply different filters to the cavities the latest ensemble run already found, without running detection again (fast: no container, no re-detection). Every cavity is kept on disk, so any threshold can be tried. Omit `filters` with replace_defaults to keep everything unfiltered.",
            json!({ "type": "object", "properties": {
                "residue_overlap": { "type": "number", "description": "Fraction of lining residues two cavities must share to count as the same pocket (default keeps the current value)" },
                "replace_defaults": { "type": "boolean", "description": "Start from no filters at all instead of the current ones" },
                "filters": {
                    "type": "array",
                    "description": "Filters to add, change or remove, as in detect_pockets_ensemble.",
                    "items": { "type": "object", "properties": {
                        "metric": { "type": "string", "description": "volume, area, max_depth, avg_depth, hydropathy, residues, persistence, or a per-residue property such as rmsf or bfactor" },
                        "min": { "type": "number" },
                        "max": { "type": "number" },
                        "remove": { "type": "boolean" }
                    }, "required": ["metric"] }
                }
            } })),
        spec("run_frodock", "Dock one structure onto another with FRODOCK (protein-protein docking; runs in Docker and can take many minutes). Both are used in their current scene positions, so pose them first if you want a local search. The top poses load as frames of one entity.",
            json!({ "type": "object", "properties": {
                "receptor": { "type": "string", "description": "Entity name or index to keep fixed (the larger one)" },
                "ligand": { "type": "string", "description": "Entity name or index to dock (the smaller one)" },
                "interaction": { "type": "string", "enum": ["other", "antibody", "enzyme"], "description": "Electrostatic profile (default other)" },
                "top_n": { "type": "integer", "description": "How many clustered poses to keep (default 10)" },
                "around_ligand": { "type": "boolean", "description": "Local docking: search only around the ligand's current position" }
            }, "required": ["receptor", "ligand"] })),
        spec("run_autodock", "Dock a small molecule into a receptor with AutoDock Vina or AutoDock4 (Docker; minutes). Uses the docking box if one is set (set_dock_box), else a box around the ligand. With covalent_residue (Cys, Lys, Ser or Tyr, e.g. A:CYS:481) the ligand is bonded to that residue through its warhead and docked as a flexible side chain; covalent poses carry no energy. Warheads: acrylamide, vinylsulfonamide and chloroacetamide (Cys, Lys), sulfonylfluoride (Lys, Ser, Tyr), aldehyde (Lys, as an imine). Poses load as frames of one entity, best first, and are listed by list_docking_poses.",
            json!({ "type": "object", "properties": {
                "receptor": { "type": "string", "description": "Entity name or index of the protein" },
                "ligand": { "type": "string", "description": "Entity name or index of the small molecule" },
                "engine": { "type": "string", "enum": ["vina", "autodock4"], "description": "Default: the last one used" },
                "exhaustiveness": { "type": "integer", "description": "Vina search effort (default 8; 32 for a careful run)" },
                "num_modes": { "type": "integer", "description": "Poses to keep (Vina) or independent runs (AutoDock4)" },
                "covalent_residue": { "type": "string", "description": "Receptor Cys/Lys/Ser/Tyr as chain:NAME:number, e.g. A:CYS:481 or A:SER:195, for covalent docking" },
                "warhead": { "type": "string", "enum": ["acrylamide", "chloroacetamide", "vinylsulfonamide", "sulfonylfluoride", "aldehyde"], "description": "Reactive group on the ligand (covalent only); must be able to bond to the residue" }
            }, "required": ["receptor", "ligand"] })),
        spec("list_docking_poses", "Return the poses from the latest docking run (FRODOCK, Vina or AutoDock4) in this project.", no_args()),
        spec("add_note", "Add a note to the project (findings, hypotheses, next steps).",
            json!({ "type": "object", "properties": { "title": { "type": "string" }, "body": { "type": "string" } }, "required": ["title", "body"] })),
    ]
}

fn parse_target(s: &str) -> Result<Target> {
    Ok(match s {
        "all" => Target::All,
        "polymer" | "protein" => Target::Polymer,
        "ligand" | "ligands" => Target::Ligand,
        "water" => Target::Water,
        "selection" => Target::Selection,
        _ => bail!("unknown target '{s}'"),
    })
}

fn parse_representation(s: &str) -> Result<Representation> {
    Ok(match s {
        "cartoon" => Representation::Cartoon,
        "ball_and_stick" | "ball-and-stick" => Representation::BallAndStick,
        "spacefill" => Representation::Spacefill,
        "licorice" | "stick" | "sticks" => Representation::Licorice,
        "backbone" => Representation::Backbone,
        "gaussian_surface" | "surface" => Representation::GaussianSurface,
        _ => bail!("unknown representation '{s}'"),
    })
}

fn parse_coloring(s: Option<&str>, representation: Representation) -> Result<Coloring> {
    Ok(match s {
        Some("element") => Coloring::Element,
        Some("chain") => Coloring::Chain,
        Some("secondary") | Some("secondary_structure") => Coloring::Secondary,
        Some("residue") => Coloring::Residue,
        Some(other) => bail!("unknown coloring '{other}'"),
        None => match representation {
            Representation::Cartoon | Representation::Backbone | Representation::GaussianSurface => Coloring::Chain,
            _ => Coloring::Element,
        },
    })
}

fn parse_component(v: &Value) -> Result<Component> {
    let target = parse_target(v["target"].as_str().context("missing target")?)?;
    let representation = parse_representation(v["representation"].as_str().context("missing representation")?)?;
    let coloring = parse_coloring(v["coloring"].as_str(), representation)?;
    Ok(Component::new(target, representation, coloring))
}

fn residue_label(s: &Structure, residue: usize) -> String {
    let r = &s.residues[residue];
    format!("{}:{} {}{}", r.chain, r.name, r.number, r.insertion_code.as_deref().unwrap_or(""))
}

pub fn structure_info(s: &Structure, components: &[Component], selection: &[usize]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "name: {}", s.name);
    let _ = writeln!(out, "atoms: {}, bonds: {}, residues: {}", s.atoms.len(), s.bonds.len(), s.residues.len());
    let mut chains: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for r in &s.residues {
        if r.atoms.start == r.atoms.end {
            continue;
        }
        let entry = chains.entry(&r.chain).or_default();
        match s.kind(r.atoms.start as usize) {
            AtomKind::Polymer => entry.0 += 1,
            AtomKind::Ligand => entry.1 += 1,
            AtomKind::Water => entry.2 += 1,
        }
    }
    for (chain, (poly, lig, wat)) in &chains {
        let _ = writeln!(out, "chain {chain}: {poly} polymer residues, {lig} hetero groups, {wat} waters");
    }
    let ligands: Vec<String> = s
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| r.atoms.start < r.atoms.end && s.kind(r.atoms.start as usize) == AtomKind::Ligand)
        .map(|(i, r)| format!("{} ({} atoms)", residue_label(s, i), r.atoms.len()))
        .collect();
    let _ = writeln!(out, "ligands/hetero groups: {}", if ligands.is_empty() { "none".into() } else { ligands.join(", ") });
    let count = |ss| s.secondary.iter().filter(|x| **x == ss).count();
    let _ = writeln!(
        out,
        "secondary structure: {} helix, {} strand, {} coil residues",
        count(SecondaryStructure::Helix),
        count(SecondaryStructure::Sheet),
        count(SecondaryStructure::Coil)
    );
    let reps: Vec<String> = components
        .iter()
        .map(|c| format!("{} → {} ({})", c.target.label(), c.representation.label(), c.coloring.label()))
        .collect();
    let _ = writeln!(out, "representations: {}", reps.join("; "));
    let mut residues: Vec<usize> = selection.iter().map(|&a| s.atoms[a].residue as usize).collect();
    residues.dedup();
    let _ = writeln!(
        out,
        "selection: {}",
        if residues.is_empty() {
            "none".into()
        } else {
            residues.iter().map(|&r| residue_label(s, r)).collect::<Vec<_>>().join(", ")
        }
    );
    out
}

/// Ligand residue indices matching `name` (or all non-water hetero groups).
fn ligand_residues(s: &Structure, name: Option<&str>) -> Vec<usize> {
    s.residues
        .iter()
        .enumerate()
        .filter(|(_, r)| r.atoms.start < r.atoms.end && s.kind(r.atoms.start as usize) == AtomKind::Ligand)
        .filter(|(_, r)| name.is_none_or(|n| r.name.eq_ignore_ascii_case(n)))
        .map(|(i, _)| i)
        .collect()
}

fn residue_atoms(s: &Structure, residues: impl IntoIterator<Item = usize>) -> Vec<usize> {
    residues
        .into_iter()
        .flat_map(|r| {
            let range = &s.residues[r].atoms;
            range.start as usize..range.end as usize
        })
        .collect()
}

impl Workspace {
    /// Execute a (non-network) tool on the UI thread.
    pub fn run_tool(&mut self, name: &str, args: &Value, cx: &mut Context<Self>) -> Result<String> {
        let viewer = self.viewer.clone();
        // "all" targets the whole scene rather than one entity.
        let all_entities = entity_arg(&args["entity"])
            .is_some_and(|k| matches!(k.to_lowercase().as_str(), "all" | "*" | "everything" | "scene"));
        if name == "clear_scene" || (name == "remove_entity" && all_entities) {
            let count = viewer.read(cx).entities().len();
            viewer.update(cx, |v, cx| v.clear(cx));
            return Ok(match count {
                0 => "the scene was already empty".to_string(),
                n => format!("cleared the scene ({n} entit{} removed)", if n == 1 { "y" } else { "ies" }),
            });
        }
        if let Some(key) = entity_arg(&args["entity"]).filter(|_| !all_entities) {
            let index = viewer
                .read(cx)
                .find_entity(&key)
                .with_context(|| format!("no entity '{key}' in the scene (use list_entities to see them)"))?;
            viewer.update(cx, |v, cx| v.set_active(index, cx));
        }
        let structure = viewer.read(cx).structure().cloned();
        let need_structure = || structure.clone().ok_or_else(|| anyhow!("no structure is loaded"));

        match name {
            "get_project_overview" => {
                let Some(project) = &self.project else {
                    return Ok("No project is open. Ask the user to create or open one.".into());
                };
                let mut out = format!("project: {} ({})\n", project.name(), project.root().display());
                let in_scene: Vec<Option<Id>> = viewer.read(cx).entities().iter().map(|e| e.structure_id).collect();
                out.push_str("structures:\n");
                for s in project.structures()? {
                    let mark = if in_scene.contains(&Some(s.id)) { " [in scene]" } else { "" };
                    let _ = writeln!(out, "  id={} {} ({}){mark}", s.id, s.name, s.source.as_deref().unwrap_or("file"));
                }
                out.push_str("views:\n");
                for v in project.views()? {
                    let _ = writeln!(out, "  {}", v.name);
                }
                out.push_str("notes:\n");
                for n in project.notes()? {
                    let _ = writeln!(out, "  {}", n.title);
                }
                Ok(out)
            }
            "get_structure_info" => {
                let s = need_structure()?;
                let v = viewer.read(cx);
                Ok(structure_info(&s, v.components(), v.selection()))
            }
            "load_structure" => {
                let id = args["structure_id"].as_i64().context("structure_id must be an integer")?;
                self.load_structure(id, cx)?;
                Ok(format!("loaded structure {id}"))
            }
            "add_structure_to_scene" => {
                let id = args["structure_id"].as_i64().context("structure_id must be an integer")?;
                let index = self.add_structure_to_scene(id, cx)?;
                Ok(format!("structure {id} added as entity {index}"))
            }
            "list_entities" => {
                let v = viewer.read(cx);
                if v.entities().is_empty() {
                    return Ok("the scene is empty".into());
                }
                let mut out = String::new();
                for (i, e) in v.entities().iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "{i}: {}{} — structure id {:?}, {} atoms, {} frame(s){}{}",
                        e.name,
                        if Some(i) == v.active_index() { " [active]" } else { "" },
                        e.structure_id,
                        e.structure.atoms.len(),
                        e.structure.frame_count(),
                        if e.transform != glam::Mat4::IDENTITY { ", moved" } else { "" },
                        if e.visible { "" } else { ", hidden" },
                    );
                }
                Ok(out)
            }
            "remove_entity" => {
                let (index, label) = {
                    let v = viewer.read(cx);
                    let i = v.active_index().context("the scene is empty")?;
                    (i, v.entities()[i].name.clone())
                };
                viewer.update(cx, |v, cx| v.remove_entity(index, cx));
                Ok(format!("removed {label} from the scene"))
            }
            "align_entities" => {
                let v = viewer.read(cx);
                let find = |key: &Value| -> Result<usize> {
                    let key = entity_arg(key).context("mobile and target are required")?;
                    v.find_entity(&key).with_context(|| format!("no entity '{key}'"))
                };
                let (mobile, target) = (find(&args["mobile"])?, find(&args["target"])?);
                let chain_residues = |entity: usize, chain: Option<&str>| {
                    chain.map(|c| {
                        v.entities()[entity]
                            .structure
                            .residues
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| r.chain.eq_ignore_ascii_case(c))
                            .map(|(i, _)| i)
                            .collect::<Vec<_>>()
                    })
                };
                let options = molecule::align::AlignOptions {
                    pairing: match args["pairing"].as_str() {
                        Some("residue_number") => molecule::align::Pairing::ResidueNumber,
                        _ => molecule::align::Pairing::Sequence,
                    },
                    mobile_residues: chain_residues(mobile, args["mobile_chain"].as_str()),
                    target_residues: chain_residues(target, args["target_chain"].as_str()),
                    ..Default::default()
                };
                let names = (v.entities()[mobile].name.clone(), v.entities()[target].name.clone());
                let fit = viewer.update(cx, |v, cx| v.align(mobile, target, &options, cx))?;
                Ok(format!(
                    "aligned {} onto {}: RMSD {:.2} Å over {} of {} Cα pairs",
                    names.0, names.1, fit.rmsd, fit.n_aligned, fit.n_pairs
                ))
            }
            "move_entity" => {
                let index = viewer.read(cx).active_index().context("the scene is empty")?;
                let vec3 = |v: &Value| -> glam::Vec3 {
                    let a: Vec<f32> = v
                        .as_array()
                        .map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect())
                        .unwrap_or_default();
                    glam::Vec3::new(
                        a.first().copied().unwrap_or(0.0),
                        a.get(1).copied().unwrap_or(0.0),
                        a.get(2).copied().unwrap_or(0.0),
                    )
                };
                let (t, r) = (vec3(&args["translate"]), vec3(&args["rotate"]));
                viewer.update(cx, |v, cx| v.move_entity(index, t, r, cx));
                Ok(format!("moved entity {index}: translate {t:?} Å, rotate {r:?}°"))
            }
            "export_pose" => {
                let index = viewer.read(cx).active_index().context("the scene is empty")?;
                self.export_pose(index, cx)
            }
            "set_frame" => {
                let index = viewer.read(cx).active_index().context("the scene is empty")?;
                let frame = args["frame"].as_u64().context("frame must be an integer")?.max(1) as usize - 1;
                viewer.update(cx, |v, cx| v.set_frame(index, frame, cx));
                let count = viewer.read(cx).entities()[index].structure.frame_count();
                Ok(format!("showing frame {} of {count}", frame.min(count - 1) + 1))
            }
            "fit_frames" => {
                let index = viewer.read(cx).active_index().context("the scene is empty")?;
                let rmsd = viewer.update(cx, |v, cx| v.fit_frames(index, 0, cx))?;
                Ok(format!("frames fitted onto frame 1; mean Cα RMSD {rmsd:.2} Å"))
            }
            "list_cavities" => Ok(self.cavity_summary()),
            "measure" => self.tool_measure(args, cx),
            "find_contacts" => self.tool_contacts(args, cx),
            "set_dock_box" => self.tool_dock_box(args, cx),
            "list_annotations" => Ok(self.tool_list_annotations(cx)),
            "clear_annotations" => {
                viewer.update(cx, |v, cx| v.clear_annotations(cx));
                Ok("cleared".into())
            }
            "list_ensemble_pockets" => Ok(self.ensemble_summary()),
            "cavity_statistics" => Ok(self.cavity_statistics()),
            "refilter_ensemble_pockets" => self.refilter_ensemble_tool(args, cx),
            "list_docking_poses" => Ok(self.docking_summary()),
            "set_representations" => {
                need_structure()?;
                let components = args["components"]
                    .as_array()
                    .context("components must be an array")?
                    .iter()
                    .map(parse_component)
                    .collect::<Result<Vec<_>>>()?;
                let n = components.len();
                viewer.update(cx, |v, cx| v.set_components(components, cx));
                Ok(format!("{n} representation(s) set"))
            }
            "add_representation" => {
                need_structure()?;
                let component = parse_component(args)?;
                viewer.update(cx, |v, cx| v.add_component(component, cx));
                Ok("added".into())
            }
            "show_water" => {
                let visible = args["visible"].as_bool().unwrap_or(true);
                viewer.update(cx, |v, cx| v.set_water_visible(visible, cx));
                Ok(format!("water {}", if visible { "shown" } else { "hidden" }))
            }
            "select_residues" => {
                let s = need_structure()?;
                let chain = args["chain"].as_str();
                let mut wanted: Vec<(isize, isize)> = Vec::new();
                for item in args["residues"].as_array().context("residues must be an array")? {
                    let text = match item {
                        Value::Number(n) => n.to_string(),
                        Value::String(t) => t.clone(),
                        _ => continue,
                    };
                    let (a, b) = match text.split_once('-') {
                        Some((a, b)) if !a.is_empty() => (a.trim().parse()?, b.trim().parse()?),
                        _ => {
                            let n: isize = text.trim().parse()?;
                            (n, n)
                        }
                    };
                    wanted.push((a.min(b), a.max(b)));
                }
                let residues: Vec<usize> = s
                    .residues
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| chain.is_none_or(|c| r.chain.eq_ignore_ascii_case(c)))
                    .filter(|(_, r)| wanted.iter().any(|(a, b)| (*a..=*b).contains(&r.number)))
                    .map(|(i, _)| i)
                    .collect();
                if residues.is_empty() {
                    bail!("no residues matched");
                }
                let atoms = residue_atoms(&s, residues.iter().copied());
                viewer.update(cx, |v, cx| {
                    v.set_selection(atoms.clone(), cx);
                    v.focus_atoms(&atoms, cx);
                });
                Ok(format!(
                    "selected {}",
                    residues.iter().map(|&r| residue_label(&s, r)).collect::<Vec<_>>().join(", ")
                ))
            }
            "select_binding_site" => {
                let s = need_structure()?;
                let cutoff = args["cutoff"].as_f64().unwrap_or(5.0) as f32;
                let ligands = ligand_residues(&s, args["ligand"].as_str());
                let Some(&ligand) = ligands.first() else {
                    bail!("no matching ligand found");
                };
                let ligand_atoms = residue_atoms(&s, [ligand]);
                let positions: Vec<_> = ligand_atoms.iter().map(|&a| s.atoms[a].position).collect();
                let cutoff2 = cutoff * cutoff;
                let mut site: Vec<usize> = s
                    .residues
                    .iter()
                    .enumerate()
                    .filter(|(i, r)| *i != ligand && r.atoms.start < r.atoms.end && !r.is_water())
                    .filter(|(_, r)| {
                        (r.atoms.start..r.atoms.end).any(|a| {
                            let p = s.atoms[a as usize].position;
                            positions.iter().any(|q| q.distance_squared(p) <= cutoff2)
                        })
                    })
                    .map(|(i, _)| i)
                    .collect();
                site.sort_unstable();
                let mut atoms = ligand_atoms.clone();
                atoms.extend(residue_atoms(&s, site.iter().copied()));
                viewer.update(cx, |v, cx| {
                    v.set_selection(atoms.clone(), cx);
                    v.focus_atoms(&atoms, cx);
                });
                Ok(format!(
                    "ligand {} — {} residues within {cutoff} Å: {}",
                    residue_label(&s, ligand),
                    site.len(),
                    site.iter().map(|&r| residue_label(&s, r)).collect::<Vec<_>>().join(", ")
                ))
            }
            "clear_selection" => {
                viewer.update(cx, |v, cx| v.set_selection(Vec::new(), cx));
                Ok("selection cleared".into())
            }
            "focus" => {
                let s = need_structure()?;
                let target = args["target"].as_str().unwrap_or("all");
                match target {
                    "all" => viewer.update(cx, |v, cx| v.reset_view(cx)),
                    "selection" => {
                        let atoms = viewer.read(cx).selection().to_vec();
                        if atoms.is_empty() {
                            bail!("nothing is selected");
                        }
                        viewer.update(cx, |v, cx| v.focus_atoms(&atoms, cx));
                    }
                    ligand => {
                        let residues = ligand_residues(&s, Some(ligand));
                        if residues.is_empty() {
                            bail!("no ligand named {ligand}");
                        }
                        let atoms = residue_atoms(&s, residues);
                        viewer.update(cx, |v, cx| v.focus_atoms(&atoms, cx));
                    }
                }
                Ok(format!("focused on {target}"))
            }
            "reset_view" => {
                viewer.update(cx, |v, cx| v.reset_view(cx));
                Ok("view reset".into())
            }
            "save_view" => {
                let name = args["name"].as_str().context("name is required")?;
                self.save_view(name, cx)?;
                Ok(format!("saved view '{name}'"))
            }
            "load_view" => {
                let name = args["name"].as_str().context("name is required")?;
                let project = self.project.as_ref().context("no project is open")?;
                let view = project
                    .views()?
                    .into_iter()
                    .find(|v| v.name.eq_ignore_ascii_case(name))
                    .with_context(|| format!("no view named '{name}'"))?;
                self.apply_view(view, cx)?;
                Ok(format!("restored view '{name}'"))
            }
            "add_note" => {
                let project = self.project.as_ref().context("no project is open")?;
                let title = args["title"].as_str().context("title is required")?;
                let body = args["body"].as_str().unwrap_or_default();
                project.add_note(title, body)?;
                cx.notify();
                Ok(format!("note '{title}' added"))
            }
            other => bail!("unknown tool '{other}'"),
        }
    }

    /// Apply a downloaded PDB entry (second half of `fetch_pdb`).
    /// `extension` is whichever format RCSB served, normally `pdb`.
    pub fn add_downloaded_structure(
        &mut self,
        pdb_id: &str,
        bytes: &[u8],
        extension: &str,
        cx: &mut Context<Self>,
    ) -> Result<String> {
        let project = self.project.as_ref().context("no project is open — ask the user to create one first")?;
        let id = pdb_id.to_uppercase();
        let entry = project.add_structure_bytes(
            &format!("{}.{extension}", id.to_lowercase()),
            bytes,
            &id,
            Some(&format!("rcsb:{id}")),
        )?;
        self.add_structure_to_scene(entry.id, cx)?;
        let viewer = self.viewer.read(cx);
        let s = viewer.structure().context("structure failed to load")?;
        Ok(format!(
            "added {id} as structure id={} and to the scene as the active entity\n{}",
            entry.id,
            structure_info(s, viewer.components(), viewer.selection())
        ))
    }
}

/// Blocking download of an entry from RCSB (run off the UI thread).
///
/// PDB format is preferred, since that is what most tools and users expect.
/// Very large entries (big complexes, ribosomes) are only published as
/// mmCIF, so that is the fallback. Returns the bytes and the extension used.
pub fn download_pdb(pdb_id: &str) -> Result<(Vec<u8>, &'static str)> {
    let id = pdb_id.trim();
    if id.len() != 4 || !id.chars().all(|c| c.is_ascii_alphanumeric()) {
        bail!("'{id}' is not a valid 4-character PDB id");
    }
    let id = id.to_uppercase();
    // A missing .pdb is a 404 we handle, not a transport failure, so don't
    // let ureq turn the status into an error.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(15)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut last_error = String::new();
    for extension in ["pdb", "cif"] {
        let url = format!("https://files.rcsb.org/download/{id}.{extension}");
        let response = agent
            .get(&url)
            .call()
            .with_context(|| format!("downloading {url}"))?;
        let status = response.status();
        if status.is_success() {
            let bytes = response
                .into_body()
                .with_config()
                .limit(200 * 1024 * 1024)
                .read_to_vec()?;
            return Ok((bytes, extension));
        }
        // 404 on .pdb means the entry is too large for the format.
        last_error = format!("{url} returned {status}");
    }
    bail!("could not download {id} from RCSB ({last_error})")
}

/// Entity argument as a string, accepting numbers too.
fn entity_arg(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

pub fn view_state_json(state: &ViewState) -> Value {
    serde_json::to_value(state).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hits RCSB; run with `cargo test -p app -- --ignored`.
    #[test]
    #[ignore]
    fn prefers_pdb_and_falls_back_to_cif() {
        // A normal entry is served as PDB.
        let (bytes, extension) = download_pdb("1hsg").unwrap();
        assert_eq!(extension, "pdb");
        assert!(bytes.starts_with(b"HEADER"), "expected a PDB header");

        // A ribosome is too large for the PDB format, so mmCIF is used.
        let (bytes, extension) = download_pdb("4V6X").unwrap();
        assert_eq!(extension, "cif");
        assert!(bytes.starts_with(b"data_"), "expected an mmCIF file");

        assert!(download_pdb("nope!").is_err());
    }
}
