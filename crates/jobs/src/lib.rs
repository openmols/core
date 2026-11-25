//! Running external tools in Docker containers.
//!
//! Every tool (pyKVFinder, FRODOCK, GROMACS, …) has a [`ToolImage`] whose
//! Dockerfile is embedded in the binary and built locally on demand. A job runs
//! one shell command in that image with the project folder mounted at `/work`
//! and the job's own folder (`jobs/<id>/`) as the working directory. Output is
//! streamed to `jobs/<id>/log.txt`.
//!
//! Everything here is blocking; callers run it on a background thread.

pub mod autodock;
pub mod frodock;
pub mod kvfinder;

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ToolImage {
    /// Short id used in job records, e.g. `pykvfinder`.
    pub tool: &'static str,
    pub label: &'static str,
    /// Image tag; bump the version when the Dockerfile changes.
    pub tag: &'static str,
    pub dockerfile: &'static str,
    pub gpu: bool,
    /// Built from a folder the user supplies (software we must not download
    /// or redistribute).
    pub needs_context: bool,
    pub description: &'static str,
}

pub const PYKVFINDER: ToolImage = ToolImage {
    tool: "pykvfinder",
    label: "pyKVFinder",
    tag: "openmols/pykvfinder:1",
    dockerfile: include_str!("../../../docker/pykvfinder/Dockerfile"),
    gpu: false,
    needs_context: false,
    description: "Cavity detection and characterisation (GPL-3.0)",
};

pub const FRODOCK: ToolImage = ToolImage {
    tool: "frodock",
    label: "FRODOCK",
    tag: "openmols/frodock:1",
    dockerfile: include_str!("../../../docker/frodock/Dockerfile"),
    gpu: false,
    needs_context: true,
    description: "Protein-protein docking, built from your own copy (no licence published)",
};

pub const AUTODOCK: ToolImage = ToolImage {
    tool: "autodock",
    label: "AutoDock",
    tag: "openmols/autodock:1",
    dockerfile: include_str!("../../../docker/autodock/Dockerfile"),
    gpu: false,
    needs_context: false,
    description: "AutoDock Vina and AutoDock4 docking, with covalent (flexible side-chain) docking (Apache-2.0 / GPL-2.0)",
};

pub const TOOLS: &[ToolImage] = &[PYKVFINDER, FRODOCK, AUTODOCK];

pub fn tool(name: &str) -> Option<&'static ToolImage> {
    TOOLS.iter().find(|t| t.tool == name)
}

fn docker() -> Command {
    let mut cmd = Command::new("docker");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Docker server version, or an error explaining that Docker isn't running.
pub fn docker_version() -> Result<String> {
    let out = docker()
        .args(["version", "--format", "{{.Server.Version}}"])
        .output()
        .context("Docker is not installed or not on PATH")?;
    if !out.status.success() {
        bail!(
            "Docker is not running — start Docker Desktop ({})",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn image_present(image: &ToolImage) -> bool {
    docker()
        .args(["image", "inspect", "--format", "{{.Id}}", image.tag])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// `docker build` the image from its embedded Dockerfile, logging to `log`.
///
/// `context` is the build context folder. Images that set `needs_context`
/// require one: that is the copy of the tool you supply, which openmols never
/// downloads. Other images build from the Dockerfile alone.
pub fn build_image(image: &ToolImage, context: Option<&Path>, log: &Path) -> Result<()> {
    if image.needs_context && context.is_none() {
        bail!(
            "{} is built from your own copy of the software — set its folder in the Tools panel first",
            image.label
        );
    }
    if let Some(dir) = context
        && !dir.is_dir()
    {
        bail!("{} is not a folder", dir.display());
    }
    let log_file = File::create(log).with_context(|| format!("creating {}", log.display()))?;
    let mut cmd = docker();
    cmd.args(["build", "--progress", "plain", "-t", image.tag, "-f", "-"]);
    match context {
        Some(dir) => cmd.arg(dir),
        None => cmd.arg("-"),
    };
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(log_file.try_clone()?)
        .stderr(log_file)
        .spawn()
        .context("starting docker build")?;
    child
        .stdin
        .take()
        .context("docker stdin")?
        .write_all(image.dockerfile.as_bytes())?;
    let status = child.wait()?;
    if !status.success() {
        bail!("docker build of {} failed (see {})", image.tag, log.display());
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RunSpec {
    pub image: ToolImage,
    /// Project folder on the host, mounted at `/work`.
    pub project_root: PathBuf,
    /// Job folder relative to the project root, e.g. `jobs/12`.
    pub job_dir: String,
    /// Shell command run with `sh -c` inside the job folder.
    pub command: String,
    /// Container name, used to cancel the job.
    pub name: String,
}

impl RunSpec {
    pub fn host_dir(&self) -> PathBuf {
        self.project_root.join(&self.job_dir)
    }

    pub fn log_path(&self) -> PathBuf {
        self.host_dir().join("log.txt")
    }
}

/// Run a job to completion, streaming output to `<job>/log.txt`.
/// Returns the container's exit code.
pub fn run(spec: &RunSpec) -> Result<i32> {
    std::fs::create_dir_all(spec.host_dir())?;
    let mut log = File::create(spec.log_path())?;
    writeln!(log, "$ {}\n", spec.command)?;
    let mut cmd = docker();
    cmd.args(["run", "--rm", "--name", &spec.name])
        .arg("--mount")
        .arg(format!(
            "type=bind,source={},target=/work",
            spec.project_root.display()
        ))
        .args(["-w", &format!("/work/{}", spec.job_dir.replace('\\', "/"))]);
    if spec.image.gpu {
        cmd.args(["--gpus", "all"]);
    }
    cmd.args([spec.image.tag, "sh", "-c", &spec.command]);
    let status = cmd
        .stdout(log.try_clone()?)
        .stderr(log.try_clone()?)
        .status()
        .context("starting docker run")?;
    let code = status.code().unwrap_or(-1);
    writeln!(log, "\n[exit code {code}]")?;
    Ok(code)
}

/// Stop a running job's container (cancellation).
pub fn stop(name: &str) -> Result<()> {
    let status = docker()
        .args(["stop", "-t", "2", name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !status.success() {
        bail!("could not stop container {name}");
    }
    Ok(())
}

/// Last `lines` lines of a job log (for the UI).
pub fn log_tail(path: &Path, lines: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dockerfiles_are_embedded() {
        for t in TOOLS {
            assert!(t.dockerfile.contains("FROM "), "{}", t.tool);
            assert!(t.tag.starts_with("openmols/"));
        }
        assert_eq!(tool("pykvfinder"), Some(&PYKVFINDER));
    }

    #[test]
    fn context_images_refuse_to_build_without_one() {
        let log = std::env::temp_dir().join("openmols-frodock-guard.log");
        let err = build_image(&FRODOCK, None, &log).unwrap_err().to_string();
        assert!(err.contains("your own copy"), "{err}");
        // A context that does not exist is reported too.
        let missing = std::path::Path::new("definitely-not-a-folder-12345");
        let err = build_image(&FRODOCK, Some(missing), &log).unwrap_err().to_string();
        assert!(err.contains("not a folder"), "{err}");
    }

    /// Needs Docker and the image; run with `cargo test -p jobs -- --ignored`.
    #[test]
    #[ignore]
    fn runs_a_command_in_the_image() {
        let root = std::env::temp_dir().join("openmols-jobs-test");
        std::fs::create_dir_all(&root).unwrap();
        let spec = RunSpec {
            image: PYKVFINDER,
            project_root: root.clone(),
            job_dir: "jobs/1".into(),
            command: "echo hello > out.txt && pyKVFinder --version".into(),
            name: "openmols-test-job".into(),
        };
        assert_eq!(run(&spec).unwrap(), 0);
        assert_eq!(std::fs::read_to_string(root.join("jobs/1/out.txt")).unwrap().trim(), "hello");
    }
}
