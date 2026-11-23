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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            recent_projects: Vec::new(),
            projects_dir: home_dir().join("openmols").join("projects"),
            providers: Provider::defaults(),
            active_provider: 0,
        }
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("OPENMOLS_CONFIG") {
        return PathBuf::from(path);
    }
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("openmols").join("config.json")
}

impl Config {
    pub fn load() -> Self {
        std::fs::read_to_string(config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let path = config_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
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
