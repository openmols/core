# Licences

openmols itself is Apache-2.0 ([LICENSE](../LICENSE)).

## Tools openmols runs

Each tool runs as a separate process inside a Docker container. openmols does not link
against them or ship their code.

| Tool | Licence | How it gets into the image |
| --- | --- | --- |
| [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) | Apache-2.0 | conda-forge `vina` |
| [AutoDock4 / AutoGrid](https://autodock.scripps.edu/) | GPL-2.0 | bioconda `autodock`, `autogrid` |
| [Meeko](https://github.com/forlilab/Meeko) | LGPL-2.1 | conda-forge `meeko` |
| [RDKit](https://www.rdkit.org/) | BSD-3-Clause | conda-forge `rdkit` |
| [Open Babel](https://openbabel.org/) | GPL-2.0 | conda-forge `openbabel` |
| [pyKVFinder](https://github.com/LBC-LNBio/pyKVFinder) | GPL-3.0 | `docker/pykvfinder/Dockerfile` (pip) |
| [FRODOCK](https://github.com/chaconlab/FRODOCK) | GPL-2.0-or-later notices in about half the sources; none in the core docking code; no top-level `LICENSE` | image publishing is switched off until the authors confirm: see below |

### Where the images come from

The pyKVFinder and AutoDock images are built by CI from the Dockerfiles in `docker/` and
published to the GitHub Container Registry (`ghcr.io/openmols/pykvfinder`,
`ghcr.io/openmols/autodock`), so users download them instead of building for several minutes.
If a pull fails the app builds the same image locally from the Dockerfile it carries.

Publishing those images distributes GPL software (AutoDock4, Open Babel, pyKVFinder) in
binary form. Each image's `org.opencontainers.image.licenses` and `.source` labels say so and
point here. The exact package versions are the ones the Dockerfile installs; their source
code is available from the upstream projects and the conda-forge / bioconda package recipes
linked above. Anyone redistributing the images should keep this notice with them.

FRODOCK is a special case (see below).

### FRODOCK

FRODOCK is public on GitHub (<https://github.com/chaconlab/FRODOCK>), but the repository has no
top-level `LICENSE` file. Reading the sources (as of this writing):

- about half the source files (the libraries `libvolume`, `libfrm`, parts of `libpdb`, and
  others) carry **GPL-2.0-or-later** notices, and the bundled TCLAP library carries its own
  permissive notice;
- the core docking programs and libraries (`frodock`, `libfrodock`, `libprotprot`, `libbessel`,
  `frodockcluster`, ...) carry **no licence notice**;
- the bundled `bin/soap.bin` scoring potential has no stated terms.

The intent looks like GPL, but the core files have no explicit grant. So the CI job that would
publish a FRODOCK image (`frodock` in `.github/workflows/containers.yml`) is **switched off**
until the maintainers have confirmed this with the authors (ideally by the authors adding a
`LICENSE` file) and set the repository variable `PUBLISH_FRODOCK` to `true`.

- While it is off, users build the image on their own machine from their own copy of FRODOCK
  (**Settings → Tools**).
- Once it is on, the app pulls `ghcr.io/openmols/frodock` like the other images and the folder
  is only a fallback. The image contains FRODOCK's full source tree (`/opt/frodock`), which
  covers the source-availability part of the GPL for the files that carry it.

Check the authors' terms before using FRODOCK commercially.

## Please cite what you use

- Eberhardt et al., *J. Chem. Inf. Model.* 61, 3891 (2021), AutoDock Vina 1.2.
- Morris et al., *J. Comput. Chem.* 30, 2785 (2009), AutoDock4.
- Meeko, Forli lab (covalent docking uses its tethered method).
- Garzón et al., *Bioinformatics* 25, 2544 (2009), and Ramírez-Aportela et al., *Bioinformatics* 32, 2386 (2016), FRODOCK.
- Karatzas, Brotzakis and Sarimveis, *J. Chem. Theory Comput.* (2025), the ensemble pocket protocol the pocket workflow follows.

## Example data

`assets/1hsg.pdb` is the Protein Data Bank entry 1HSG (HIV-1 protease with indinavir).
PDB data are available under CC0.

## Dependencies

Rust dependencies are listed in `Cargo.lock`. The GUI uses
[GPUI](https://www.gpui.rs/) and `gpui-component` (Apache-2.0).
Run `cargo install cargo-license && cargo license` for the full list.
