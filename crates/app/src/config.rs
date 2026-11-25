//! Per-user application settings (`%APPDATA%/openmols/config.json` on Windows,
//! `~/.config/openmols/config.json` elsewhere). `OPENMOLS_CONFIG` overrides the path.

use llm::Provider;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Most recent first.
    pub recent_projects: Vec<PathBuf>,
    /// Where "New project" creates folders.
    pub projects_dir: PathBuf,
    pub providers: Vec<Provider>,
    pub active_provider: usize,
    /// Your own copy of FRODOCK (its repository folder). openmols never
    /// downloads it; the Docker image is built from this folder.
    #[serde(default)]
    pub frodock_dir: Option<PathBuf>,
    /// Last-used job parameters (what the job dialogs open with, and what the
    /// assistant's tools fall back to).
    pub job_defaults: JobDefaults,
    /// Panel shown beside the left rail.
    pub left_tab: LeftTab,
}

/// The tabs of the left rail.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeftTab {
    #[default]
    Project,
    Scene,
    Jobs,
    Results,
    Measure,
    Views,
}

impl LeftTab {
    pub const ALL: [LeftTab; 6] = [
        LeftTab::Project,
        LeftTab::Scene,
        LeftTab::Jobs,
        LeftTab::Results,
        LeftTab::Measure,
        LeftTab::Views,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LeftTab::Project => "Project",
            LeftTab::Scene => "Scene",
            LeftTab::Jobs => "Jobs",
            LeftTab::Results => "Results",
            LeftTab::Measure => "Measure",
            LeftTab::Views => "Views",
        }
    }
}

/// Parameters of the jobs started from the app, remembered between runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct JobDefaults {
    pub kvfinder: jobs::kvfinder::Settings,
    /// Analyse every Nth frame in an ensemble pocket run.
    pub ensemble_stride: usize,
    pub dock_top_n: usize,
    pub dock_cluster_rmsd: f32,
    pub dock_around_ligand: bool,
    pub dock_restrain_to_selection: bool,
    /// AutoDock Vina / AutoDock4.
    pub ad_engine: jobs::autodock::Engine,
    pub ad_exhaustiveness: u32,
    /// Poses (Vina) or independent runs (AutoDock4).
    pub ad_modes: u32,
    pub ad_covalent: bool,
    pub ad_warhead: jobs::autodock::Warhead,
}

impl Default for JobDefaults {
    fn default() -> Self {
        let dock = jobs::frodock::Settings::default();
        Self {
            kvfinder: jobs::kvfinder::Settings::default(),
            ensemble_stride: 1,
            dock_top_n: dock.top_n,
            dock_cluster_rmsd: dock.cluster_rmsd,
            dock_around_ligand: false,
            dock_restrain_to_selection: false,
            ad_engine: jobs::autodock::Engine::Vina,
            ad_exhaustiveness: 8,
            ad_modes: 9,
            ad_covalent: false,
            ad_warhead: jobs::autodock::Warhead::Acrylamide,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            recent_projects: Vec::new(),
            projects_dir: home_dir().join("openmols").join("projects"),
            providers: Provider::defaults(),
            active_provider: 0,
            frodock_dir: None,
            job_defaults: JobDefaults::default(),
            left_tab: LeftTab::Project,
        }
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Where per-user files live: `%APPDATA%\openmols`, else `~/.config/openmols`.
pub fn config_dir() -> PathBuf {
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("openmols")
}

fn config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("OPENMOLS_CONFIG") {
        return PathBuf::from(path);
    }
    config_dir().join("config.json")
}

impl Config {
    pub fn load() -> Self {
        let mut config: Config = std::fs::read_to_string(config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let mut migrate = false;
        for p in &mut config.providers {
            match p.api_key.as_deref().filter(|k| !k.is_empty()) {
                // A plaintext key from an older config: move it to the keychain on the next save.
                Some(_) => migrate = true,
                None => p.api_key = keychain::get(&p.name),
            }
        }
        if migrate {
            config.save();
        }
        config
    }

    /// Writes the config. API keys go to the OS keychain and stay out of the
    /// file; if the keychain is unavailable the key stays in the file.
    pub fn save(&self) {
        let path = config_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let mut on_disk = self.clone();
        for p in &mut on_disk.providers {
            let key = p.api_key.clone().filter(|k| !k.is_empty());
            if keychain::set(&p.name, key.as_deref()) {
                p.api_key = None;
            }
        }
        if let Ok(json) = serde_json::to_string_pretty(&on_disk) {
            std::fs::write(path, json).ok();
        }
    }

    pub fn touch_recent(&mut self, root: PathBuf) {
        self.recent_projects.retain(|p| p != &root);
        self.recent_projects.insert(0, root);
        self.recent_projects.truncate(10);
        self.save();
    }

    pub fn provider(&self) -> Option<&Provider> {
        self.providers.get(self.active_provider)
    }
}

/// API keys in the OS credential store, one entry per provider name.
mod keychain {
    const SERVICE: &str = "openmols";

    fn entry(name: &str) -> Option<keyring::Entry> {
        keyring::Entry::new(SERVICE, name).ok()
    }

    pub fn get(name: &str) -> Option<String> {
        entry(name)?.get_password().ok().filter(|k| !k.is_empty())
    }

    /// Store (or delete, for `None`) the key. Returns true when the keychain
    /// now holds exactly what was asked, so the file need not.
    pub fn set(name: &str, key: Option<&str>) -> bool {
        let Some(entry) = entry(name) else { return false };
        match key {
            Some(key) => entry.set_password(key).is_ok() && entry.get_password().ok().as_deref() == Some(key),
            None => {
                // Nothing to delete is fine.
                let _ = entry.delete_credential();
                true
            }
        }
    }
}
