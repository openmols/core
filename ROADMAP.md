# openmols roadmap

An open-source drug-discovery workbench in Rust: a GPUI desktop app with a
Mol\*-quality molecular viewer, projects for discovery campaigns, an LLM
assistant that drives the app, and open-source engines (docking, MD, cavity
detection) run in Docker containers.

Status date: 2026-11-23.

## Architecture

| Crate | Purpose |
| --- | --- |
| `crates/molecule` | Data model, PDB/mmCIF/SDF parsing, bonds, secondary structure, superposition, PDB writing |
| `crates/render` | wgpu renderer (UI-agnostic): impostors, meshes, cartoon, Gaussian surface, picking |
| `crates/project` | One folder + SQLite per campaign: structures, views, chats, notes, jobs |
| `crates/jobs` | Running external tools in Docker; per-tool command building and output parsing |
| `crates/llm` | Streaming chat client for OpenAI-compatible endpoints, with tool calls |
| `crates/app` | The GPUI application: viewport, drawers, chat, tool implementations |

External tools run in Docker images built from `docker/<tool>/Dockerfile`.

---

## Done

### M1 — Multi-structure scene
Several structures in one scene, each with its own rigid transform,
representations, selection and frame. Entity-aware picking. Ensembles and
multi-model files load as frames with playback.

### M2 — Posing (Blender-style)
**G** grabs, **R** rotates, **X/Y/Z** lock an axis, click/Enter confirms,
Esc cancels, Ctrl+Z/Ctrl+Shift+Z undo and redo. "Apply transform" bakes the
move into the coordinates; "Export pose" writes the posed structure into the
project as docking input.

### M3 — Alignment
PyMOL-`align`-style superposition: Needleman–Wunsch sequence alignment
(BLOSUM62, affine gaps) → Cα pairing → Kabsch fit → iterative outlier
rejection. Also pairing by residue number, chain restriction, and per-frame
fitting of trajectories. *Verified: 1HSG chain A onto B, and 3OXC onto 1HSG at
0.61 Å over 197/198 pairs.*

### M4 — Docker job runner
Images built on demand from Dockerfiles embedded in the binary. Each job runs
in `jobs/<id>/` inside the project with the project mounted at `/work`, logging
to `log.txt`. Jobs are recorded in the project database, can be cancelled, and
show live status in the **Tools & jobs** panel.

### M5 — Cavity detection (pyKVFinder)
Detects and characterises pockets: volume, area, depth, hydropathy and lining
residues per cavity. Cavities become their own scene entity rendered as
surfaces, one colour each, plus a sortable table that selects lining residues
on click. *Verified: 1HSG active site at 690 Å³, lined by Asp25/Asp25′.*

### Representations
Cartoon (own DSSP, with strand arrowheads), Gaussian surface, ball & stick,
spacefill, licorice, backbone. Colour by element, chain, secondary structure or
residue. Targets: all, polymer, ligand, water, selection.

---

### M8 — Ensemble pocket detection *(done; detection validated, UI run pending)*
The pocket half of the AR-V7 pipeline (Karatzas, Brotzakis & Sarimveis,
*JCTC* 2025), for any ensemble loaded as frames.

- pyKVFinder runs on **every frame** in a single container (`frames/*.pdb` →
  `out/<frame>/`). Measured: 12 frames in 7 s, so ~350 frames in ~3.5 min.
- Cavities are filtered with that paper's criteria as defaults (>5 lining
  residues, hydropathy < 0, area 80–2480 Å², volume > 120 Å³) and merged into
  **consensus pockets** when they share ≥60% of their lining residues.
- Each pocket reports **persistence** (fraction of frames it appears in),
  max/mean volume, mean area and hydropathy, and residues ranked by how often
  they line it. Frames with no cavities still count towards persistence.
- Panel lists pockets by persistence; clicking one jumps to the frame where it
  is largest and selects its lining residues.
- RMSF, mean positions, radius of gyration and frame RMSD are computed
  straight from the frames (`crates/molecule/src/analysis.rs`) — no trajectory
  reader needed.
- **Every cavity is kept.** Detection is the expensive half (1000 frames of a
  24-chain fibril took ~43 min and found 43,122 cavities), so nothing is
  discarded: the per-frame results stay in the job folder and the filters are
  applied on top. Changing a filter re-merges in ~30 ms (~230 ms with no
  filters at all), with no container and no re-detection. A job from an
  earlier session is re-read in the background when the project opens, so
  runs made before this existed can be re-filtered too.
- **You write the filters.** The panel holds an editable rule list, one per
  line, and `Apply filters` re-merges immediately:

  ```text
  residues >= 6          # any metric, or a per-residue property
  hydropathy <= 0        # such as rmsf or bfactor
  area 80 .. 2480        # both bounds at once
  volume >= 120
  overlap 0.6            # residues two cavities must share to merge
  ```

  `#` starts a comment, `>`/`<` read as `>=`/`<=`, and an empty list means no
  filtering. `Show all` and `Paper defaults` are one click. A name that is
  not a built-in metric is read as a per-residue property, so a misspelling
  is reported with the names available rather than silently rejecting every
  cavity.
- **Statistics before filtering.** The panel reports how many cavities were
  found, how many frames held one, the most in any frame, and for every
  metric its range, median and 10th/90th percentiles — so a threshold comes
  from the data. It also shows what each rule keeps **on its own**, which
  identifies the one doing the damage. On that 1000-frame run: median cavity
  volume is 29 A^3, so `volume >= 120` alone drops 85%; `hydropathy <= 0`
  keeps 38%; all four together keep 7% (2,960 cavities) and give 41 pockets —
  the same count as the paper.
- Metrics: `volume`, `area`, `max_depth`, `avg_depth`, `hydropathy`,
  `residues`, `persistence`, or the mean over a pocket's lining residues of
  any per-residue property the app supplies. Two ship today — `rmsf`
  (computed from the ensemble's own frames, superposed first so it measures
  internal motion) and `bfactor` — and they are stored beside the job, so
  filtering on them later does not recompute RMSF over every frame. Adding
  another MD-derived measure means adding it to `residue_properties()` in
  `crates/app/src/jobs_ui.rs`; it becomes filterable by name with no change
  to the filter code. A metric whose value cannot be determined rejects the
  cavity, so a filter never passes silently.
- Panel lists pockets by persistence; clicking one jumps to the frame where it
  is largest and selects its lining residues.
- RMSF, mean positions, radius of gyration and frame RMSD are computed
  straight from the frames (`crates/molecule/src/analysis.rs`) — no trajectory
  reader needed.
- Assistant tools: `detect_pockets_ensemble`, `refilter_ensemble_pockets`
  (re-filter without re-running), `cavity_statistics` (the distributions
  above) and `list_ensemble_pockets`.
- Dev tool: `cargo run -p jobs --example ensemble -- <job-dir> [filters-file]`,
  which prints the same statistics and accepts the same filter text (`-` for
  stdin).

Still to do: run it once from the UI on a real ensemble. Further MD-derived
filters worth adding to `residue_properties()`: per-residue SASA,
secondary-structure propensity, hydrogen-bond occupancy, and contact
frequency with a chosen partner.

## Next

### M6 — FRODOCK protein–protein docking *(built, not yet run)*
FRODOCK has **no licence file** in its repository, so openmols never downloads
or redistributes it. You point openmols at your own copy ("Set FRODOCK
folder…" in the Tools panel) and the image is built locally from that folder.
It uses the supplied Linux binaries if they are executables, otherwise it
compiles with FRODOCK's own `compile_frodock.sh`.

Done:
- Images that build from a user-supplied folder, and the setting that holds it.
- The pipeline, following FRODOCK's guide: `frodockgrid` receptor maps (van der
  Waals, electrostatics `-m 1 -t A/E/O`, desolvation `-m 3`) and the ligand
  desolvation map → `frodock` → `frodockcluster` → `frodockview -p`.
- Local docking around the ligand's current position (`--around x,y,z`) and
  residue-pair restraints (`frodockonstraints`).
- Solution parsing (rank, Euler angles, position, score) and a poses table.
- The top poses are combined into one multi-model PDB, so they load as **frames
  of a single entity** and the frame slider scrubs through them.
- Assistant tools `run_frodock` and `list_docking_poses`.

The image builds and all six tools compile. Quirks found and handled:
- Only `frodock_gcc` ships prebuilt; the rest are compiled. `compile_frodock.sh`
  also walks the Intel (`icpc`) and MPI (`mpiCC`) variants, which abort the
  run, so the image builds the GCC targets directly in dependency order.
- `-lfftw3f` is required (`libfftw3-dev`).
- `libpdb` has a second target, `Release2_gcc`, producing `libpdb2_gcc.a`,
  which the cluster and constraint tools link against.
- `libfrodockcluster`'s directory is spelled `Relase_gcc` upstream.
- A Windows checkout leaves CRLF endings that bash and make reject.
- **The tools exit non-zero even when they succeed**, so the pipeline checks
  for each output file rather than trusting exit status.
- `frodockgrid -m 3` does write `<input>_ASA.pdb`, as the guide implies.
- FRODOCK's pose PDBs stop at column 66 with no element column, which strict
  readers reject. `molecule`'s loader now pads short `ATOM`/`HETATM` lines and
  fills in an element guessed from the atom name, so tool output loads
  whatever its width.

**Validated** on FRODOCK's own 1MLC antibody/antigen test (unbound–unbound):
the pipeline produced 10 clustered poses, and pose 4 sits **4.86 Å** from the
native complex (`1MLC_l_b.pdb`), measured as ligand RMSD without
superposition over all 1001 shared atoms; every other pose is 16–26 Å away.
Upstream's note says their 3rd solution is the near-native one — the likely
difference is clustering with `--nc 10` here versus `--nc 100` there.
The 10 poses combine into one multi-model PDB and load back as 10 frames,
which is the path the app uses.

Still to do:
- Speed: the image is sequential and took **2.7 h** for 1MLC on one core under
  WSL2. Adding OpenMPI and building the `Release_MPI_GCC` targets would give
  roughly a 4x speed-up, and is close to required for routine use.
- Surface a progress estimate: `frodock` prints `Searching <percent>`, which
  the Jobs panel could show instead of a bare log tail.
- "Export for Rosetta": posed complex plus a `docking_protocol` flags file.
  Rosetta needs your own licensed copy, same arrangement as FRODOCK.

### M7 — Trajectories, ensembles and MD runs
1. Readers: GROMACS `.xtc` (via the `molly` crate), `.dcd`, `.gro`, with
   stride/range/memory limits. Trajectories stored in the project. Also
   "load a folder of PDBs as frames", which several generative pipelines need.
2. Analysis: the measures in `analysis.rs` are done; what remains is the
   `Property` colouring fed from RMSF and the plots in an Analysis panel.
3. Launch GROMACS in Docker with a protein-only protocol: `pdb2gmx`
   (AMBER99SB-ILDN/TIP3P) → solvate → ions → minimise → NVT → NPT →
   production on the GPU, with progress read from `md.log`.
4. On completion, `trjconv -pbc mol -center` then auto-load as a multi-frame
   entity.
5. Protein–ligand MD (ligand parametrisation) is a follow-up, not in v1.

---

## Backlog

### Viewer
- Structures are read on a worker thread and queued one at a time, so opening
  a gigabyte trajectory no longer freezes the window. Assistant and job paths
  (`add_structure_to_scene`) are still synchronous; those files are small.
- Remaining Mol\* representations: molecular (SES) and solvent-accessible
  surfaces, putty, labels, density maps.
- Transparency, so a surface can sit over a cartoon.
- Ambient occlusion and outlines (the main reason Mol\* looks better).
- Restrict cavity detection to a box around the selection (pyKVFinder `-B`).
- Manual check of the G/R posing controls with a real mouse.

### Assistant
- Native Anthropic and Google Gemini clients (OpenRouter covers both for now).
- Strip chat-template tokens such as `<|im_start|>` leaked by some local models.
- A system-prompt pass to stop small models acting before tool results arrive.

### M9 — AutoDock Vina, AutoDock4 and covalent docking *(done; verified in the container)*
`docker/autodock` (conda-forge + bioconda: vina 1.2, autodock 4.2, autogrid,
Meeko 0.8, RDKit, Open Babel). Meeko prepares receptor and ligand; the search
box is the one drawn in the viewport. Poses load as frames of one entity.

- Vina redock of the 1HSG inhibitor: −11.5 kcal/mol; AutoDock4: −10.9.
- **Covalent** (Cys, Lys, Ser, Tyr) uses Meeko's tethered method: the
  ligand's warhead is turned into its adduct with a stand-in for the residue's
  side chain by `covalent_prep.py`, superposed on the residue's CA/CB and
  docked as a flexible side chain with a one-atom dummy ligand. Warheads:
  acrylamide, vinyl sulfonamide, chloroacetamide (Cys, Lys), sulfonyl fluoride
  (Lys sulfonamide, Ser/Tyr sulfonate ester), aldehyde (Lys imine).
  Neither engine includes the bonded residue in its reported energy, so
  covalent poses are ranked (AutoDock4 by inter + internal energy) but shown
  without a score. Checked against crystal complexes (nearest-atom RMSD of the
  best of five poses): BTK Cys481–ibrutinib 5P9J 0.7 Å; chymotrypsin Ser195–PMS
  1EQ9 0.5 Å; Tyr41–PMS 1WB8 1.1 Å; aspartate aminotransferase Lys258–PLP
  1AAW 0.7 Å (best pose; Vina ranks the 0.9 Å pose second, AutoDock4 third).
- Limits: receptor = polymer only; no flexible receptor side chains yet;
  AutoDock4 evaluations fixed at 2.5 M; Thr, His, Asp/Glu and non-covalent
  warheads (boronic acids, β-lactams) not covered.

### M10 — Compact rail and measurement tools *(done)*
Icon-less rail (Project · Scene · Jobs · Results · Measure · Views). Click-to-
measure distances, angles and dihedrals (M/A/D), contact sets (H-bonds,
contacts, clashes) between entities, a docking box, and SASA / van der Waals
volume / buried fraction / pocket overlap. Drawn as a 2D overlay (no depth
test yet). Assistant tools: `measure`, `find_contacts`, `set_dock_box`,
`list_annotations`, `clear_annotations`, `run_autodock`.

### Platform
- First commit and push to `github.com/openmols/core`.
- GPUI 0.3 upgrade (needs gpui-component 0.6 and a newer wgpu).
- Windows note: `target/debug` debug info is capped in `Cargo.toml` because
  full debug info overflows MSVC's PDB limit on this project.

---

## File formats

Structures load from PDB (`.pdb`, biological assemblies `.pdb1`…, `.ent`,
`.brk`), mmCIF (`.cif`, `.mmcif`) and MDL (`.sdf`, `.mol`), each optionally
gzipped. Short `ATOM`/`HETATM` lines from external tools are padded and given
an element guessed from the atom name, so FRODOCK poses and pyKVFinder cavity
grids load as they are.

`fetch_pdb` downloads **PDB format** from RCSB and only falls back to mmCIF
for entries too large for it (1HSG is served as `.pdb`; the 4V6X ribosome
only as `.cif`). Everything openmols writes itself — posed structures,
docking poses, cavity grids — is PDB.

## Conventions

- External tools never link into openmols; they run as processes in containers,
  which also keeps GPL-licensed tools (pyKVFinder) at arm's length.
- `crates/render` must not depend on any UI toolkit.
- Every parser has a unit test against a real captured output file.
- Tools the assistant can call live in `crates/app/src/tools.rs`; each returns a
  short text summary that is fed back to the model.
