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
| [FRODOCK](https://github.com/chaconlab/FRODOCK) | no licence published | **not redistributed**: see below |

Images are built on the user's machine from the Dockerfiles in `docker/`. openmols does
not publish or distribute built images.

### FRODOCK

The FRODOCK repository carries no licence file, so openmols neither downloads nor ships it.
To use it you point **Settings → Tools** at your own copy and the image is built from that
folder locally. Check the terms with its authors before sharing the built image or using it
commercially.

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
