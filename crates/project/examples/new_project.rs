//! Create a project from the command line and optionally add structures.
//!
//! cargo run -p project --example new_project -- <dir> <name> [structure files...]

use anyhow::{Context, Result};
use project::Project;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("usage: new_project <dir> <name> [files...]")?;
    let name = args.next().unwrap_or_else(|| "Untitled".into());
    let project = Project::create(&dir, &name)?;
    for file in args {
        let entry = project.add_structure_file(&file, None)?;
        println!("added {} as id={}", entry.name, entry.id);
    }
    println!("created project '{}' at {}", project.name(), project.root().display());
    Ok(())
}
